#!/usr/bin/env python3
"""Real CPU/Gloo TP2 SP collectives with a deterministic RMSNorm oracle.

Compares AllReduce -> full RMSNorm with ReduceScatter -> local RMSNorm ->
AllGather. The input is synthetic row-parallel partial output. This does not
execute a vLLM model, Ascend FlashComm, a GPU/NPU kernel, or attention/KV sharding.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform


def values(owner, n, h, repetition):
    return [[((row * 7 + col * 3 + owner * 5 + repetition * 2) % 17 - 8) / 8
             for col in range(h)] for row in range(n)]


def oracle(n, h, repetition, eps):
    ranks = [values(owner, n, h, repetition) for owner in range(2)]
    output = []
    for row in range(n):
        summed = [math.fsum(ranks[owner][row][col] for owner in range(2)) for col in range(h)]
        inv_rms = 1 / math.sqrt(math.fsum(x * x for x in summed) / h + eps)
        output.append([x * inv_rms * ((col + 1) / h) for col, x in enumerate(summed)])
    return output


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        p.error("Launch with torchrun --nproc-per-node=2")
    import torch
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    result = {"schema_version": "wiki-sp-gloo/1", "status": "running", "rank": rank,
              "world_size": 2, "started_at": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "backend": "gloo"},
              "scope": "real CPU collectives and RMSNorm on synthetic row-parallel partial outputs; no model/NPU/FlashComm performance or KV result",
              "rmsnorm": {"hidden_size": 8, "epsilon": 1e-6, "accumulation_dtype": "float32", "weight": "gamma[c]=(c+1)/8"},
              "cases": []}
    target = args.output / f"rank{rank}.json"
    write_json(target, result)
    try:
        for dtype in [torch.float32, torch.bfloat16]:
            for n in [4, 5]:
                for repetition in range(2):
                    padded = math.ceil(n / 2) * 2
                    local_rows, h = padded // 2, 8
                    partial = torch.zeros((padded, h), dtype=dtype)
                    partial[:n] = torch.tensor(values(rank, n, h, repetition), dtype=dtype)
                    gamma = torch.arange(1, h + 1, dtype=torch.float32) / h
                    def norm(x):
                        fp32 = x.float()
                        return (fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + 1e-6) * gamma).to(dtype)
                    full = partial.clone()
                    dist.all_reduce(full)
                    baseline = norm(full)
                    shard = torch.empty((local_rows, h), dtype=dtype)
                    dist.reduce_scatter_tensor(shard, partial.contiguous())
                    local_norm = norm(shard)
                    assembled = torch.empty((padded, h), dtype=dtype)
                    dist.all_gather_into_tensor(assembled, local_norm.contiguous())
                    expected = torch.tensor(oracle(n, h, repetition, 1e-6), dtype=torch.float64)
                    torch.testing.assert_close(assembled, baseline, rtol=0, atol=0)
                    tolerance = 1e-6 if dtype == torch.float32 else 0.004
                    torch.testing.assert_close(assembled[:n].double(), expected, rtol=tolerance, atol=tolerance)
                    assert torch.count_nonzero(assembled[n:]).item() == 0
                    start, end = rank * local_rows, (rank + 1) * local_rows
                    result["cases"].append({"dtype": str(dtype), "valid_rows": n, "padded_rows": padded,
                        "repetition": repetition, "reduce_scatter_input_shape": [padded, h],
                        "local_norm_shape": [local_rows, h], "all_gather_output_shape": [padded, h],
                        "local_padded_row_interval": [start, end], "local_valid_rows": max(0, min(n, end) - start),
                        "baseline_norm_elements_per_rank": padded * h, "sp_norm_elements_per_rank": local_rows * h,
                        "sp_vs_allreduce_exact": True, "padding_zero": True, "oracle_rtol": tolerance,
                        "oracle_atol": tolerance, "oracle_max_abs_error": (assembled[:n].double() - expected).abs().max().item(),
                        "output_checksum_fp64": assembled[:n].double().sum().item()})
                    write_json(target, result)
        result["status"] = "passed"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(target, result)
        dist.barrier()
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
        write_json(target, result)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
