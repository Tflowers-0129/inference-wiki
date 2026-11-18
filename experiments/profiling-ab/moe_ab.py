#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Weight preparation and operators adapted from ../a5-moe.py and:
# https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/tests/e2e/nightly/single_node/ops/test_mega_moe.py
# See ../LICENSE-APACHE-2.0.txt. The URL identifies provenance, not installed code.
"""Two-rank A5 MoE eager/replay A/B on identical weights, inputs and routing.

No checkpoint or model runner is used. Both MC2 and MegaMoE eager paths first
validate outputs and integer expert counts. The selected implementation is then
compared against its own NPUGraph replay. Profiling uses that SAME execution
object, never an eager substitute. Capture failure is fatal and recorded.

Run with torchrun --nproc-per-node=2 and an external process-tree timeout.
--reservation-id records an external reservation identifier; it does not check
ownership, acquire a lease, inspect devices or stop any existing process.
Output and raw traces are private experiment archives requiring publication review.
--smoke checks the path only; its few timing samples do not establish performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def module_identity(name):
    """Collect installed identity without hardcoding a commit or dumping env."""
    module = importlib.import_module(name)
    result = {"module": name, "version": str(getattr(module, "__version__", "unknown")),
              "git_commit": "unavailable"}
    try:
        distribution = importlib.metadata.distribution(name.replace("_", "-"))
        result["distribution_version"] = distribution.version
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            # Record only an installed commit, never repository URLs or local paths.
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                result["installed_vcs_commit"] = commit
    except importlib.metadata.PackageNotFoundError:
        result["distribution_version"] = "unavailable"
    origin = getattr(module, "__file__", None)
    if origin and Path(origin).is_file():
        file = Path(origin).resolve()
        result["module_file_sha256"] = sha256(file.read_bytes())
        for directory in list(file.parents)[:6]:
            if not (directory / ".git").exists():
                continue
            try:
                result["git_commit"] = subprocess.check_output(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
                dirty = subprocess.check_output(
                    ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5)
                result["tracked_tree_dirty"] = bool(dirty.strip())
            except (OSError, subprocess.SubprocessError):
                result["git_identity"] = "unavailable"
            break
    for field in ("__commit__", "__git_version__", "git_version"):
        value = getattr(module, field, None)
        if isinstance(value, str):
            result[field] = value
    return result


def tensor_digest(tensor):
    import torch
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "sha256": sha256(raw)}


def module_digest(module):
    return {name: tensor_digest(value) for name, value in module.named_parameters()}


class Execution:
    """One callable for correctness, unprofiled timing and profiled execution."""
    def __init__(self, mode, fn=None, inputs=None, graph=None, output=None):
        self.mode, self.fn, self.inputs = mode, fn, inputs
        self.graph, self.output = graph, output

    def run(self):
        if self.mode == "replay":
            self.graph.replay()
        else:
            self.output = self.fn(*self.inputs)
        return self.output


def describe_samples(values):
    ordered = sorted(values)
    def percentile(p):
        x = (len(ordered) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)
    return {"count": len(values), "median_us": statistics.median(values),
            "mean_us": statistics.mean(values), "min_us": min(values),
            "p10_us": percentile(.1), "p90_us": percentile(.9), "max_us": max(values)}


def build_kernels(a, torch, torch_npu, dist, rank, world):
    """Keep the historical production weight packing and operator calls."""
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
    from vllm_ascend.ops.fused_moe.moe_utils import load_cann_mega_moe_ops, select_mega_moe_activation_kwargs
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
    from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4
    from vllm_ascend.quantization.quant_type import QuantType
    from vllm_ascend.utils import bootstrap_custom_op_env
    group = dist.new_group(backend="hccl", timeout=timedelta(seconds=a.timeout_seconds))
    mega_group = dist.new_group(backend="hccl", timeout=timedelta(seconds=a.timeout_seconds))
    name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    mega_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    torch.manual_seed(a.seed + rank)
    torch.npu.manual_seed_all(a.seed + rank)
    # The standalone reference needs SiTU even when the platform disables the
    # generic custom-op capability on A5. Import after selecting the device.
    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    allocate, mega = load_cann_mega_moe_ops()
    act_kwargs = select_mega_moe_activation_kwargs(
        mega,
        activation=a.activation,
        activation_clamp=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    def make_weights(out_dim, in_dim):
        weights, scales = [], []
        for _ in range(a.experts // world):
            source = torch.randn((out_dim, in_dim), device="npu", dtype=torch.bfloat16) / in_dim**0.5
            w, s = torch_npu.npu_dynamic_mx_quant(source, dst_type=torch_npu.float4_e2m1fn_x2)
            weights.append(w.view(torch.uint8))
            scales.append(s.view(torch.uint8).reshape(out_dim, in_dim // 64, 2))
        return torch.stack(weights), torch.stack(scales).flatten(-2)

    def prepare_weights():
        # Exercise the production weight producer for both layouts. Only backend
        # selection is isolated from the global model configuration in this test.
        layer = torch.nn.Module()
        for name, dims in (("w13", (2 * a.intermediate, a.hidden)), ("w2", (a.hidden, a.intermediate))):
            w, scale = make_weights(*dims)
            layer.register_parameter(name + "_weight", torch.nn.Parameter(w, requires_grad=False))
            layer.register_parameter(name + "_weight_scale", torch.nn.Parameter(scale, requires_grad=False))
        baseline = torch.nn.Module()
        for name, value in layer.named_parameters():
            baseline.register_parameter(name, torch.nn.Parameter(value.clone(), requires_grad=False))
        logical_identity = module_digest(layer)
        if module_digest(baseline) != logical_identity:
            raise RuntimeError("Cloned logical weight contents differ before layout conversion")
        method = object.__new__(w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod)
        with patch.object(w4a8_mxfp4, "get_current_vllm_config", return_value=None):
            with patch.object(w4a8_mxfp4, "use_cann_megamoe", return_value=False):
                method.process_weights_after_loading(baseline)
            with patch.object(w4a8_mxfp4, "use_cann_megamoe", return_value=True):
                method.process_weights_after_loading(layer)
        with patch.object(w4a8_mxfp4, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True)):
            payload = method.get_fused_mc2_weights(layer)
        return baseline, layer, payload, logical_identity

    baseline_layer, mega_layer, weights, logical_weight_identity = prepare_weights()
    mega_layer.mega_moe_activation_kwargs = act_kwargs
    w1, s1 = baseline_layer.w13_weight, baseline_layer.w13_weight_scale
    w2, s2 = baseline_layer.w2_weight, baseline_layer.w2_weight_scale
    buffer = allocate(
        mega_group,
        a.experts,
        a.tokens,
        a.topk,
        a.hidden,
        2 * a.intermediate,
        max_recv_token_num=min(65536, a.tokens * world * min(a.topk, a.experts // world)),
        dispatch_quant_mode=4,
        dispatch_quant_out_dtype=24,
    )
    # Instantiate only the integration endpoint: no model runner or checkpoint is
    # required. The unit tests separately cover FusedMC2CommImpl.__init__ binding.
    comm = object.__new__(FusedMC2CommImpl)
    comm.mega_moe = mega
    comm.mega_moe_symm_buffer = buffer
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = a.tokens
    comm.token_dispatcher.global_bs = 0

    def fused(x, ids, probs):
        inp = build_fused_experts_input(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=probs,
            quant_type=QuantType.W4A8MXFP,
            layer=mega_layer,
            mxfp_act_quant_type=torch.float8_e4m3fn,
            mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
            mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
            mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            mxfp_use_bf16=True,
            dynamic_eplb=False,
            activation=a.activation,
        )
        return comm._apply_cann_mega_moe(inp, weights, is_decode_only_node=False)

    def decomposed_chunk(x, ids, probs):
        expanded, scale, assist, counts, ep_counts, tp_counts, expanded_scales = (
            torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=ids,
                expert_shard_type=0,
                shared_expert_rank_num=0,
                moe_expert_num=a.experts,
                global_bs=0,
                expert_token_nums_type=1,
                scales=None,
                quant_mode=4,
                group_ep=name,
                ep_world_size=world,
                ep_rank_id=rank,
                comm_alg="",
                tp_world_size=1,
                tp_rank_id=0,
                y_dtype=torch.float8_e4m3fn,
                expert_scales=probs,
            )[:7]
        )
        if scale.ndim == 2:
            scale = scale.reshape(scale.shape[0], -1, 2)
        mm1 = torch_npu.npu_grouped_matmul(
            x=[expanded],
            weight=[w1],
            antiquant_scale=[s1],
            per_token_scale=[scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=counts,
            group_list_type=1,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        if a.activation == "situ":
            activated, act_scale = torch.ops._C_ascend.situ_mx_quant(
                x=mm1, beta=4.0, linear_beta=25.0, activate_left=True, dst_type=36
            )
        else:
            activated, act_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                mm1, topk_weight=None, group_index=counts, dst_type=torch.float8_e4m3fn, quant_mode=2, clamp_value=0.0
            )
        if act_scale.ndim == 2:
            act_scale = act_scale.reshape(act_scale.shape[0], -1, 2)
        mm2 = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[w2],
            antiquant_scale=[s2],
            per_token_scale=[act_scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=counts,
            group_list_type=1,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        y = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=mm2,
            expert_ids=ids,
            expert_scales=probs,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=a.experts,
            global_bs=0,
            ep_send_counts=ep_counts,
            group_ep=name,
            ep_world_size=world,
            ep_rank_id=rank,
            expand_scales=expanded_scales,
            comm_quant_mode=0,
            comm_alg="",
            assist_info_for_combine=assist,
        )
        return y, counts

    def layout_identity(module):
        # Prepared A5 internal-format parameters do not support generic clone/
        # contiguous. Observe their layout without materializing or altering it.
        return {name: {"shape": list(value.shape), "stride": list(value.stride()),
                       "dtype": str(value.dtype)} for name, value in module.named_parameters()}
    identities = {"logical_weights_before_layout_conversion": logical_weight_identity,
                  "cloned_logical_weights_exact": True,
                  "mc2_prepared_layouts": layout_identity(baseline_layer),
                  "mega_prepared_layouts": layout_identity(mega_layer),
                  "logical_origin": "same random weights cloned and hashed before production layout conversion",
                  "prepared_bytes_hashed": False}
    return {"mc2": decomposed_chunk, "mega": fused}, identities

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', choices=['mc2', 'mega', 'both'], default='mc2')
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--experts', type=int, default=8)
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--intermediate', type=int, default=1024)
    p.add_argument('--topk', type=int, default=2)
    p.add_argument('--activation', choices=['silu', 'situ'], default='silu')
    p.add_argument('--input-scale', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=30, help='Samples per ABBA block')
    p.add_argument('--rounds', type=int, default=2, help='Number of eager/replay/replay/eager block cycles')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-iterations', type=int, default=3)
    p.add_argument('--profile-level', choices=['default', 'Level0', 'Level1', 'Level2'], default='Level1')
    p.add_argument('--graph-rtol', type=float, default=0.0)
    p.add_argument('--graph-atol', type=float, default=0.0)
    p.add_argument('--timeout-seconds', type=int, default=120,
                   help='Process-group timeout; an outer process-tree timeout is still required')
    p.add_argument('--job-id', required=True, help='Same unique ID on both ranks; do not reuse an output directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reservation-id', required=True,
                   help='External reservation identifier only; this script does not validate or acquire a lease')
    p.add_argument('--smoke', action='store_true', help='Use 3 warmups, 2 samples/block, 1 ABBA cycle and 2 profiled iterations')
    a = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', a.job_id):
        p.error('--job-id must use 1–80 ASCII letters, digits, dot, dash or underscore')
    if not a.reservation_id.strip():
        p.error('--reservation-id must be nonempty; it is not proof of exclusivity')
    if not 1 <= a.tokens <= 256:
        p.error('--tokens must be 1..256 so the MC2 reference remains unchunked')
    if a.experts < 2 or a.experts % 2 or not 1 <= a.topk <= a.experts:
        p.error('Use an even positive expert count and 1 <= topk <= experts')
    if any(n <= 0 or n % 64 for n in [a.hidden, a.intermediate]):
        p.error('hidden and intermediate must be positive multiples of 64 for this MXFP fixture')
    if min(a.warmup, a.repeats, a.rounds, a.profile_iterations, a.timeout_seconds) < 1:
        p.error('Warmup, sample counts and timeout must be positive')
    if not math.isfinite(a.input_scale) or a.input_scale <= 0:
        p.error('--input-scale must be finite and positive')
    if any(not math.isfinite(x) or x < 0 for x in [a.graph_rtol, a.graph_atol]):
        p.error('Graph tolerances must be finite and nonnegative')
    if a.smoke:
        a.warmup, a.repeats, a.rounds, a.profile_iterations = 3, 2, 1, 2
    return a


def software_identity(torch, torch_npu):
    names = ['torch', 'torch_npu', 'vllm', 'vllm_ascend',
             'vllm_ascend.ops.fused_moe.moe_comm_method',
             'vllm_ascend.ops.fused_moe.moe_utils',
             'vllm_ascend.ops.fused_moe.token_dispatcher',
             'vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4']
    result = {'python': platform.python_version(), 'modules': [module_identity(n) for n in names],
              'device_name': torch.npu.get_device_name(),
              'torch_git_version': getattr(torch.version, 'git_version', None),
              'selected_environment': {key: os.environ.get(key, 'unset') for key in [
                  'HCCL_OP_EXPANSION_MODE', 'HCCL_EXEC_TIMEOUT', 'ASCEND_LAUNCH_BLOCKING',
                  'TASK_QUEUE_ENABLE', 'MULTI_STREAM_MEMORY_REUSE']},
              'driver_version': 'not queried; attach resource-controller environment manifest',
              'cann_version': 'unavailable'}
    version = getattr(torch_npu, 'version', None)
    result['torch_npu_build'] = {key: str(getattr(version, key)) for key in
                               ['git_version', 'cann_version'] if hasattr(version, key)}
    for key in ['ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME']:
        root = os.environ.get(key)
        if not root:
            continue
        for name in ['version.cfg', 'version.info', 'version.ini']:
            candidate = Path(root) / name
            if candidate.is_file():
                data = candidate.read_bytes()
                result['cann_version_file'] = {'name': name, 'sha256': sha256(data),
                    'version_lines': [line.strip() for line in data.decode(errors='replace').splitlines()
                                      if re.match(r'(?i)^\s*(version|version_dir|cann_version)\s*=', line)]}
                result['cann_version'] = 'see dynamically read version file'
                return result
    return result


def main(argv=None):
    a = parse_args(argv)
    if sys.flags.optimize:
        raise SystemExit('Run without -O/PYTHONOPTIMIZE so correctness assertions remain enabled')
    rank, world = int(os.environ.get('RANK', '-1')), int(os.environ.get('WORLD_SIZE', '0'))
    if world != 2 or rank not in [0, 1]:
        raise SystemExit('Launch with torchrun --nproc-per-node=2; exactly two ranks are required')
    job_dir = a.output / a.job_id
    rank_dir = job_dir / f'rank{rank}'
    # Refuse reuse even when another run failed. Each run needs a fresh job ID.
    rank_dir.mkdir(parents=True, exist_ok=False)
    report_path = rank_dir / 'result.json'
    config = {key: value for key, value in vars(a).items() if key not in ['output', 'reservation_id']}
    report = {'schema_version': 'wiki-moe-graph-ab/1', 'job_id': a.job_id, 'rank': rank,
              'world_size': world, 'started_at': utc_now(), 'status': 'running', 'stage': 'imports',
              'script_sha256': sha256(Path(__file__).read_bytes()), 'config': config,
              'reservation': {'provided': True, 'id_sha256': sha256(a.reservation_id.encode()),
                              'validated_by_script': False},
              'scope': 'two-rank A5 W4A8MXFP fixed-input operator chain; same implementation eager vs NPUGraph replay; no model runner or service',
              'measurement': {'unit': 'microseconds', 'profiler_during_timing': False,
                  'event_boundary': 'NPU events around execution submission; may include host and peer gaps',
                  'host_boundary': 'perf_counter around execution submission only, excludes event synchronization',
                  'rank_max': 'maximum matched local event duration; not a clock-aligned multi-device makespan',
                  'clock_alignment': 'not calibrated; profile rank panels need independent origins',
                  'smoke_only': a.smoke},
              'correctness': {}, 'capture': {}, 'samples': [], 'profiles': [], 'summary': {}}
    write_json(report_path, report)
    dist = None
    succeeded = False
    try:
        import torch
        import torch.distributed as dist
        import torch_npu
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        timeout = timedelta(seconds=a.timeout_seconds)
        report['stage'] = 'process_group_init'; write_json(report_path, report)
        dist.init_process_group('hccl', timeout=timeout)
        cpu_group = dist.new_group(backend='gloo', timeout=timeout)
        shared = [None] * world
        signature = {'config': config, 'script_sha256': report['script_sha256']}
        dist.all_gather_object(shared, signature, group=cpu_group)
        if shared[0] != shared[1]:
            raise RuntimeError('Both ranks must use identical config and script; refusing unmatched collective order')
        report['stage'] = 'environment_identity'; write_json(report_path, report)
        report['environment'] = software_identity(torch, torch_npu)
        report['stage'] = 'weights_and_buffers'; write_json(report_path, report)
        # no_grad avoids autograd and keeps captured buffers mutable normal tensors.
        with torch.no_grad():
            kernels, weight_identity = build_kernels(a, torch, torch_npu, dist, rank, world)
            report['weights'] = weight_identity
            x = torch.randn((a.tokens, a.hidden), device='npu', dtype=torch.bfloat16) * a.input_scale
            logits = torch.rand((a.tokens, a.experts), device='npu')
            ids = logits.topk(a.topk, dim=-1).indices.to(torch.int32)
            probs = torch.softmax(torch.randn((a.tokens, a.topk), device='npu'), dim=-1)
            inputs = (x, ids, probs)
            input_ptrs = [t.data_ptr() for t in inputs]
            report['inputs'] = {name: tensor_digest(t) for name, t in zip(['x', 'expert_ids', 'weights'], inputs)}
            expected_counts = torch.bincount(ids.flatten().to(torch.int64), minlength=a.experts)
            dist.all_reduce(expected_counts)
            local_experts = a.experts // world
            expected_local = expected_counts[rank * local_experts:(rank + 1) * local_experts].cpu()
            report['correctness']['expected_expert_counts'] = expected_local.tolist()
            report['correctness']['global_tasks'] = int(expected_counts.sum().item())
            assert report['correctness']['global_tasks'] == a.tokens * world * a.topk
            report['stage'] = 'eager_correctness'; write_json(report_path, report)
            snapshots = {}
            for impl in ['mc2', 'mega']:
                out, counts = kernels[impl](*inputs)
                torch.npu.synchronize()
                assert torch.equal(counts.cpu().to(torch.int64), expected_local), f'{impl}: count oracle failed'
                assert torch.isfinite(out).all().item(), f'{impl}: nonfinite output'
                snapshots[impl] = (out.detach().clone(), counts.detach().clone())
            ref, value = [snapshots[k][0].cpu().double() for k in ['mc2', 'mega']]
            norm = ref.square().mean().item()
            if norm == 0:
                raise RuntimeError('Zero numerical reference: this input cannot support a relative-error check')
            relative_rmse = ((value - ref).square().mean().item() / norm) ** .5
            cosine = torch.nn.functional.cosine_similarity(value.flatten(), ref.flatten(), dim=0).item()
            report['correctness']['eager_cross_implementation'] = {
                'relative_rmse': relative_rmse, 'cosine': cosine,
                'max_abs_error': (value - ref).abs().max().item(),
                'thresholds': {'relative_rmse_lt': .03, 'cosine_gt': .999}, 'expert_counts_exact': True}
            assert relative_rmse < .03 and cosine > .999, 'MC2/MegaMoE numerical agreement failed'
            implementations = ['mc2', 'mega'] if a.implementation == 'both' else [a.implementation]
            graph_keepalive = []
            for impl in implementations:
                eager = Execution('eager', kernels[impl], inputs)
                report['stage'] = f'{impl}/pre_capture_warmup'; write_json(report_path, report)
                for _ in range(a.warmup):
                    eager.run()
                torch.npu.synchronize()
                dist.barrier(group=cpu_group)
                report['stage'] = f'{impl}/capture'
                report['capture'][impl] = {'status': 'attempting', 'eager_fallback': False}
                write_json(report_path, report)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    captured = kernels[impl](*inputs)
                replay = Execution('replay', graph=graph, output=captured)
                graph_keepalive.append(replay)
                executors = {'eager': eager, 'replay': replay}
                report['stage'] = f'{impl}/capture_validation'; write_json(report_path, report)
                replay.run()
                torch.npu.synchronize()
                torch.testing.assert_close(captured[0], snapshots[impl][0], rtol=a.graph_rtol, atol=a.graph_atol)
                assert torch.equal(captured[1].cpu().to(torch.int64), expected_local)
                assert [t.data_ptr() for t in inputs] == input_ptrs
                capture_ptrs = [t.data_ptr() for t in captured[:2]]
                report['capture'][impl] = {'status': 'passed', 'eager_fallback': False,
                    'output_matches_own_eager': True, 'expert_counts_exact': True,
                    'rtol': a.graph_rtol, 'atol': a.graph_atol,
                    'input_addresses_stable': True, 'output_addresses_stable': True}
                write_json(report_path, report)

                for cycle in range(a.rounds):
                    for block, mode in enumerate(['eager', 'replay', 'replay', 'eager']):
                        executor = executors[mode]
                        report['stage'] = f'{impl}/{mode}/timing/cycle{cycle}/block{block}'
                        write_json(report_path, report)
                        for _ in range(a.warmup):
                            executor.run()
                        torch.npu.synchronize()
                        dist.barrier(group=cpu_group)
                        block_samples = []
                        # Reuse events; the first synchronization also initializes them outside measurement.
                        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        start.record(); end.record(); end.synchronize()
                        for iteration in range(a.repeats):
                            start.record()
                            begin = time.perf_counter_ns()
                            executor.run()
                            submitted = time.perf_counter_ns()
                            end.record()
                            end.synchronize()
                            elapsed = start.elapsed_time(end) * 1000
                            if not math.isfinite(elapsed) or elapsed <= 0:
                                raise RuntimeError('Nonpositive or nonfinite NPU event measurement')
                            block_samples.append({'job_id': a.job_id, 'implementation': impl, 'mode': mode,
                                'cycle': cycle, 'block': block, 'iteration': iteration,
                                'rank': rank, 'npu_event_us': elapsed,
                                'host_submit_us': (submitted - begin) / 1000})
                        # Host aggregation is outside every timing window.
                        peers = [None] * world
                        dist.all_gather_object(peers, block_samples, group=cpu_group)
                        for index, sample in enumerate(block_samples):
                            for peer in peers:
                                assert all(peer[index][key] == sample[key] for key in
                                           ['job_id', 'implementation', 'mode', 'cycle', 'block', 'iteration'])
                            sample['matched_rank_max_us'] = max(peer[index]['npu_event_us'] for peer in peers)
                        report['samples'].extend(block_samples)
                        write_json(report_path, report)

                if a.profile:
                    report['stage'] = f'{impl}/profiler_configuration'; write_json(report_path, report)
                    extra = {}
                    if a.profile_level != 'default':
                        level = getattr(torch_npu.profiler.ProfilerLevel, a.profile_level)
                        extra['experimental_config'] = torch_npu.profiler._ExperimentalConfig(profiler_level=level)
                    for block, mode in enumerate(['eager', 'replay']):
                        executor = executors[mode]
                        target = rank_dir / 'profiles' / impl / mode
                        report['stage'] = f'{impl}/{mode}/profiling'; write_json(report_path, report)
                        for _ in range(a.warmup):
                            executor.run()
                        torch.npu.synchronize()
                        dist.barrier(group=cpu_group)
                        with torch_npu.profiler.profile(
                            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                        torch_npu.profiler.ProfilerActivity.NPU],
                            schedule=torch_npu.profiler.schedule(wait=0, warmup=1,
                                                               active=a.profile_iterations, repeat=1),
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
                            record_shapes=True, profile_memory=False, with_stack=False, **extra,
                        ) as prof:
                            # Profiler initialization is complete on both ranks before paired iterations.
                            dist.barrier(group=cpu_group)
                            for iteration in range(a.profile_iterations + 1):
                                label = f'wiki/{a.job_id}/{impl}/{mode}/rank{rank}/iteration{iteration}'
                                with torch.profiler.record_function(label):
                                    executor.run()
                                # Same serial-completion policy as timing; this is not measured performance.
                                torch.npu.synchronize()
                                prof.step()
                        torch.npu.synchronize()
                        dist.barrier(group=cpu_group)
                        artifacts = [{'path': str(p.relative_to(job_dir)), 'bytes': p.stat().st_size,
                                      'sha256': file_digest(p)}
                                     for p in sorted(target.rglob('*.json')) if p.is_file()]
                        if not artifacts:
                            raise RuntimeError(f'{mode}: profiler produced no JSON artifact')
                        report['profiles'].append({'implementation': impl, 'mode': mode,
                            'rank': rank, 'directory': str(target.relative_to(job_dir)),
                            'warmup_iteration': 0, 'active_iterations': list(range(1, a.profile_iterations + 1)),
                            'execution_object': 'same object used in unprofiled timing',
                            'profile_level': a.profile_level, 'artifacts': artifacts,
                            'trace_semantics': 'unreviewed; replay scopes alone do not prove device task attribution'})
                        write_json(report_path, report)
                report['stage'] = f'{impl}/final_correctness'; write_json(report_path, report)
                for mode, executor in executors.items():
                    output = executor.run()
                    torch.npu.synchronize()
                    torch.testing.assert_close(output[0], snapshots[impl][0], rtol=a.graph_rtol, atol=a.graph_atol)
                    assert torch.equal(output[1].cpu().to(torch.int64), expected_local)
                assert [t.data_ptr() for t in inputs] == input_ptrs
                assert [t.data_ptr() for t in replay.output[:2]] == capture_ptrs
                assert report['inputs'] == {name: tensor_digest(t) for name, t in zip(['x', 'expert_ids', 'weights'], inputs)}
                report['correctness'][impl + '_after_timing_and_profile'] = 'passed'
                for mode in ['eager', 'replay']:
                    samples = [s for s in report['samples'] if s['implementation'] == impl and s['mode'] == mode]
                    report['summary'][f'{impl}/{mode}'] = {
                        'npu_event': describe_samples([s['npu_event_us'] for s in samples]),
                        'host_submit': describe_samples([s['host_submit_us'] for s in samples]),
                        'matched_rank_max': describe_samples([s['matched_rank_max_us'] for s in samples])}
                # Keep graph, its outputs and both weight layouts alive until all replay users finish.
                torch.npu.synchronize()
                dist.barrier(group=cpu_group)
            report['stage'] = 'complete'
            report['status'] = 'passed'
            report['finished_at'] = utc_now()
            write_json(report_path, report)
            print(json.dumps({'job_id': a.job_id, 'rank': rank, 'status': 'passed',
                              'smoke_only': a.smoke, 'samples': len(report['samples'])}), flush=True)
            succeeded = True
    except BaseException as exc:
        report['status'] = 'failed'
        report['failure_stage'] = report['stage']
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        report['finished_at'] = utc_now()
        for capture in report['capture'].values():
            if capture['status'] == 'attempting':
                capture['status'] = 'failed_or_unverified'
                capture['note'] = 'Capture or first replay failed; no eager substitution or performance conclusion'
        write_json(report_path, report)
        # A nonzero worker exit lets torchrun stop peers. Do not add a collective in the error path.
        raise
    finally:
        # Destroying a failed communicator can itself block. The launcher must enforce
        # an outer timeout and drain both workers before releasing the reservation.
        if succeeded and dist is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
