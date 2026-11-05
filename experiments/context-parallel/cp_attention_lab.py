# SPDX-License-Identifier: Apache-2.0
# Uses the installed vLLM-Ascend DCP custom op and group coordinator.
# Group setup follows the upstream test_dcp_a2a.py, with a device communicator
# enabled for the added Q-head all_gather. See README.md for the fixed source.
"""Two-rank A5 attention-shard correctness and profiling experiment.

This invokes CANN FIA for local attention and the production registered
dcp_a2a_fused for output/LSE exchange and merge. It does not instantiate a
full model runner or paged KV cache manager. KV-byte counts describe input
tensor payload, not allocator reservations or process-wide HBM usage.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)
from vllm_ascend.ops.triton.dcp import dcp_a2a
from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import can_exchange
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def cpu_reference(q, k, v):
    # Inputs have already been rounded to BF16, matching the NPU inputs.
    q, k, v = q.double(), k.double(), v.double()
    k = k.expand(-1, -1, q.shape[1], -1)
    v = v.expand(-1, -1, q.shape[1], -1)
    score = torch.einsum('bhd,blhd->bhl', q, k) / math.sqrt(q.shape[-1])
    out = torch.einsum('bhl,blhd->bhd', score.softmax(-1), v)
    return out, score.logsumexp(-1).unsqueeze(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[512, 4096])
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--interleave', type=int, default=128)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert world == 2 and args.heads % world == 0
    torch.npu.set_device(local_rank)
    init_device_properties_triton()
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank,
                                distributed_init_method='env://', backend='hccl')
    group = init_model_parallel_group([list(range(world))], local_rank=local_rank,
                                     backend='hccl', group_name='wiki_cp_attention',
                                     use_device_communicator=True)
    args.output.mkdir(parents=True, exist_ok=True)
    h_local = args.heads // world
    head_slice = slice(rank * h_local, (rank + 1) * h_local)
    seq_q = list(range(1, args.batch + 1))

    def fia(q, k, v, length):
        result = torch_npu.npu_fused_infer_attention_score(
            q, k, v, num_heads=q.shape[1], num_key_value_heads=1,
            input_layout='TND', atten_mask=None, scale=args.head_dim ** -0.5,
            sparse_mode=0, antiquant_mode=0, antiquant_scale=None,
            softmax_lse_flag=True, actual_seq_lengths=seq_q,
            actual_seq_lengths_kv=[length * (i + 1) for i in range(args.batch)])
        out, lse = result
        return out.reshape(args.batch, q.shape[1], args.head_dim), lse.reshape(args.batch, q.shape[1], 1)

    rows = []
    cases = [('weighted_merge', 512)] + [('random', length) for length in args.lengths]
    for repetition in range(2):
        for case, length in cases:
            generator = torch.Generator().manual_seed(20261105 + length)
            q_cpu = torch.randn(args.batch, args.heads, args.head_dim, generator=generator).bfloat16()
            k_cpu = torch.randn(args.batch, length, 1, args.head_dim, generator=generator).bfloat16()
            v_cpu = torch.randn(args.batch, length, 1, args.head_dim, generator=generator).bfloat16()
            positions = torch.arange(length)
            owners = (positions // args.interleave) % world
            local_indices = positions[owners == rank]
            if case == 'weighted_merge':
                q_cpu.zero_(); q_cpu[..., 0] = 1
                k_cpu.zero_(); k_cpu[:, owners == 0, :, 0] = math.log(9.0) * math.sqrt(args.head_dim)
                v_cpu.zero_(); v_cpu[:, owners == 0] = 10
            expected_all, expected_lse = cpu_reference(q_cpu, k_cpu, v_cpu)
            expected = expected_all[:, head_slice]
            q_local = q_cpu[:, head_slice].contiguous().npu()
            k_full = k_cpu.reshape(-1, 1, args.head_dim).contiguous().npu()
            v_full = v_cpu.reshape(-1, 1, args.head_dim).contiguous().npu()
            k_local = k_cpu[:, local_indices].reshape(-1, 1, args.head_dim).contiguous().npu()
            v_local = v_cpu[:, local_indices].reshape(-1, 1, args.head_dim).contiguous().npu()
            local_length = local_indices.numel()

            def tp_only():
                return fia(q_local, k_full, v_full, length)[0]

            def dcp_path():
                # Like the TP-only DCP backend: gather distinct query heads,
                # attend to this rank's sequence shard, scatter/merge heads.
                gathered_q = group.all_gather(q_local, dim=1)
                partial_out, partial_lse = fia(gathered_q, k_local, v_local, local_length)
                return torch.ops.vllm.dcp_a2a_fused(partial_out, partial_lse, world, 1, group.unique_name)

            tp_out = tp_only()
            gathered_q = group.all_gather(q_local, dim=1)
            partial_out, partial_lse = fia(gathered_q, k_local, v_local, local_length)
            merged = torch.ops.vllm.dcp_a2a_fused(partial_out, partial_lse, world, 1, group.unique_name)
            torch.npu.synchronize()
            tp_cpu, merged_cpu = tp_out.cpu().double(), merged.cpu().double()
            torch.testing.assert_close(tp_cpu, expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(merged_cpu, expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(merged_cpu, tp_cpu, atol=0.02, rtol=0.02)
            row = {'case': case, 'repetition': repetition, 'rank': rank,
                   'sequence_length': length, 'local_kv_positions': local_length,
                   'heads_before_gather': h_local, 'heads_after_gather': args.heads,
                   'output_heads_after_merge': merged.shape[1],
                   'tp_full_kv_tensor_bytes': (k_full.numel() + v_full.numel()) * k_full.element_size(),
                   'dcp_local_kv_tensor_bytes': (k_local.numel() + v_local.numel()) * k_local.element_size(),
                   'tp_max_abs_error_vs_fp64': (tp_cpu - expected).abs().max().item(),
                   'dcp_max_abs_error_vs_fp64': (merged_cpu - expected).abs().max().item(),
                   'dcp_max_abs_error_vs_tp': (merged_cpu - tp_cpu).abs().max().item(),
                   'merge_branch': 'sfa_dcp_exchange' if can_exchange(partial_out, partial_lse, world) else 'pack_a2a_combine',
                   'correctness': 'passed'}
            # FP32 result/LSE path validates the general packed merge as well.
            out_with_lse = torch.ops.vllm.dcp_a2a_fused(partial_out.float(), partial_lse,
                                                     world, 1, group.unique_name, return_lse=True)
            out_lse_cpu = out_with_lse.cpu().double()
            torch.testing.assert_close(out_lse_cpu[..., :-1], expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(out_lse_cpu[..., -1:], expected_lse[:, head_slice], atol=0.02, rtol=0.002)
            row['merged_lse_max_abs_error_vs_fp64'] = (out_lse_cpu[..., -1:] - expected_lse[:, head_slice]).abs().max().item()
            if case == 'weighted_merge':
                partials = [torch.empty_like(partial_out) for _ in range(world)]
                dist.all_gather(partials, partial_out, group=group.device_group)
                naive = torch.stack(partials).float().mean(0)[:, head_slice].cpu().double()
                row.update(expected_scalar=expected.flatten()[0].item(),
                           measured_scalar=merged_cpu.flatten()[0].item(),
                           naive_average_scalar=naive.flatten()[0].item(),
                           naive_average_max_error=(naive - expected).abs().max().item())
                assert row['naive_average_max_error'] > 1.0
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.profile and repetition == 1 and case == 'random' and length == max(args.lengths):
                for label, fn in [('TP2_DCP1', tp_only), ('TP2_DCP2', dcp_path)]:
                    for _ in range(4): fn()
                    torch.npu.synchronize(); dist.barrier(group=group.device_group)
                    path = args.output / 'profiles' / label / f'rank{rank}'
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path), worker_name=f'wiki_cp_{label}_rank{rank}'),
                        record_shapes=True, profile_memory=False, with_stack=False) as prof:
                        for iteration in range(3):
                            with torch.profiler.record_function(f'wiki/{label}/length{length}/iteration{iteration}'):
                                fn()
                                torch.npu.synchronize()
                            prof.step()
                    dist.barrier(group=group.device_group)
    result = {'correctness': 'passed', 'rank': rank, 'config': vars(args) | {'output': str(args.output)},
              'scope': 'CANN FIA + production DCP group query gather and registered output/LSE merge; no full Runner/paged cache',
              'clock_domain': 'independent per-rank Ascend profiler exports; no cross-rank timestamp alignment asserted',
              'rows': rows}
    (args.output / f'result-rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    dist.barrier(group=group.device_group)
    group.destroy()
    destroy_distributed_environment()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
