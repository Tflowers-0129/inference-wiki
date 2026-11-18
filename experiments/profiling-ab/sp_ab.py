#!/usr/bin/env python3
"""Compare two TP2 SP micro-paths, with correctness before timing or profiling.

A: AllReduce -> RMSNorm on every row.
B: ReduceScatter -> RMSNorm on local rows -> AllGather.

NPU uses HCCL and torch_npu.npu_rms_norm. CPU is a separately labelled
Gloo/TorchNorm smoke test, never evidence of NPU or full-model performance.
This script does not enable vLLM FlashComm, run a model, or measure KV memory.
The reservation ID records an external allocation; it does not reserve devices.

Example (after an external two-device allocation and environment setup):
  torchrun --standalone --nproc-per-node=2 sp_ab.py --device npu \
    --reservation-id ALLOCATION_ID --output ./captures/sp-ab --profile
CPU smoke (in an environment with CPU PyTorch/Gloo):
  torchrun --standalone --nproc-per-node=2 sp_ab.py --device cpu \
    --output ./captures/sp-cpu --warmup 1 --samples 2
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


MODES = ("A", "B")
PATHS = {"A": "AllReduce -> full RMSNorm", "B": "ReduceScatter -> local RMSNorm -> AllGather"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def partial_value(owner, row, channel, repetition):
    # Multiples of 1/8 are exactly representable in BF16 for this value range.
    return ((row * 7 + channel * 3 + owner * 5 + repetition * 2) % 17 - 8) / 8


def reference_fp64(valid_rows, hidden, repetition, gamma, epsilon):
    """Independent Python FP64 sum/norm, using the actual quantized gamma."""
    output = []
    for row in range(valid_rows):
        summed = [math.fsum(partial_value(owner, row, c, repetition) for owner in range(2))
                  for c in range(hidden)]
        inverse = 1.0 / math.sqrt(math.fsum(v * v for v in summed) / hidden + epsilon)
        output.append([v * inverse * gamma[c] for c, v in enumerate(summed)])
    return output


def distribution_identity(name, module_name=None):
    identity = {}
    try:
        distribution = importlib.metadata.distribution(name)
        identity["version"] = distribution.version
        direct = distribution.read_text("direct_url.json")
        if direct:
            # Do not publish URLs or local installation paths.
            vcs = json.loads(direct).get("vcs_info", {})
            identity["installed_vcs_commit"] = vcs.get("commit_id")
    except importlib.metadata.PackageNotFoundError:
        identity["version"] = None
    if module_name:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None and spec.origin:
                directory = Path(spec.origin).resolve().parent
                probe = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                       capture_output=True, text=True, timeout=5)
                if probe.returncode == 0:
                    identity["source_checkout_head"] = probe.stdout.strip()
        except (ImportError, OSError, subprocess.TimeoutExpired):
            pass
    return identity


def environment_identity(torch, device, local_rank, torch_npu):
    packages = {name: distribution_identity(name, module) for name, module in [
        ("torch", None), ("torch-npu", None), ("triton-ascend", None),
        ("vllm", "vllm"), ("vllm-ascend", "vllm_ascend")
    ]}
    result = {"python": platform.python_version(), "system": platform.system(),
              "machine": platform.machine(), "torch": torch.__version__,
              "device": device, "collective_backend": "hccl" if device == "npu" else "gloo",
              "packages": packages}
    if device == "npu":
        result.update(torch_npu=getattr(torch_npu, "__version__", None),
                      device_name=torch.npu.get_device_name(local_rank),
                      HCCL_OP_EXPANSION_MODE=os.environ.get("HCCL_OP_EXPANSION_MODE", "unset"))
        # Record only version entries, never a host path, endpoint or serial number.
        candidates = [Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg")]
        if os.environ.get("ASCEND_HOME_PATH"):
            candidates.insert(0, Path(os.environ["ASCEND_HOME_PATH"]) / "version.cfg")
        result["cann_version_entries"] = []
        for candidate in candidates:
            if candidate.is_file():
                lines = candidate.read_text(errors="replace").splitlines()
                entries = [line.strip() for line in lines if "=" in line
                           and "version" in line.split("=", 1)[0].lower()]
                if entries:
                    result["cann_version_entries"] = entries
                    break
        result["cann_version_verified"] = bool(result["cann_version_entries"])
    return result


def sample_stats(values):
    ordered = sorted(values)
    return {"count": len(values), "min": ordered[0], "median": statistics.median(values),
            "p95_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1], "max": ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "npu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", default="", help="unique trace ID shared by ranks; rank 0 generates one if omitted")
    parser.add_argument("--reservation-id", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50, help="samples per mode and row count; alternating AB/BA pairs")
    parser.add_argument("--profile", action="store_true", help="separate post-timing traces, one warmup and two active iterations per mode")
    parser.add_argument("--profile-level", choices=["Level1", "default"], default="Level1",
                        help="NPU profiler detail; unsupported requested level fails explicitly")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        parser.error("Launch with torchrun --nproc-per-node=2")
    if args.device == "npu" and not args.reservation_id.strip():
        parser.error("NPU requires --reservation-id recording an externally verified allocation")
    if args.warmup < 1 or args.samples < 2 or args.timeout_seconds < 1:
        parser.error("Require warmup >= 1, samples >= 2 and a positive timeout")
    if args.job_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.job_id):
        parser.error("--job-id must use 1–80 ASCII letters, digits, dot, dash or underscore")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    report_path = args.output / f"rank{rank}.json"
    if report_path.exists():
        parser.error(f"Result already exists: {report_path}; use a fresh output directory")
    report = {"schema_version": "wiki-sp-ab/1", "started_at": utc_now(), "status": "initializing",
              "rank": rank, "world_size": 2, "device": args.device, "job_id": args.job_id or None,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reservation": {"id": args.reservation_id, "establishes_exclusivity": False},
              "scope": ("NPU HCCL + fused RMSNorm micro-path" if args.device == "npu"
                        else "CPU/Gloo + TorchNorm smoke only; not an NPU measurement"),
              "exclusions": ["no vLLM model", "no FlashComm integration", "no graph capture",
                             "no KV capacity measurement", "no service throughput claim"],
              "configuration": {"dtype": "bfloat16", "hidden_size": 4096, "valid_rows": [128, 129],
                                "epsilon": 1e-6, "gamma": "BF16((c+1)/4096)",
                                "warmup_pairs": args.warmup, "samples_per_mode": args.samples,
                                "order": "even pair AB; odd pair BA", "paths": PATHS,
                                "profile_requested": args.profile, "profile_level_requested": args.profile_level},
              "timing_boundary": {"preparation": "input reset, device synchronization and control barrier excluded",
                                  "host_ms": "perf_counter through completed local path; includes launch and synchronization",
                                  "npu_event_ms": "local current-stream event interval including collective dependencies; not pure kernel time",
                                  "cross_rank": "independent clocks; do not subtract timestamps from different ranks"},
              "correctness": [], "samples": [], "profiles": []}
    save_json(report_path, report)
    dist = None
    cpu_group = None
    succeeded = False
    try:
        import torch
        import torch.distributed as dist
        torch.set_num_threads(1)
        torch_npu = None
        device = torch.device("cpu")
        if args.device == "npu":
            import torch_npu
            torch.npu.set_device(local_rank)
            device = torch.device("npu", local_rank)
        dist.init_process_group("hccl" if args.device == "npu" else "gloo",
                                timeout=timedelta(seconds=args.timeout_seconds))
        cpu_group = (dist.new_group(backend="gloo", timeout=timedelta(seconds=args.timeout_seconds))
                     if args.device == "npu" else dist.group.WORLD)
        signatures = [None, None]
        signature = {"device": args.device, "requested_job_id": args.job_id,
                     "configuration": report["configuration"],
                     "script_sha256": report["script_sha256"]}
        dist.all_gather_object(signatures, signature, group=cpu_group)
        if signatures[0] != signatures[1]:
            raise RuntimeError("Both ranks must use identical configuration and script")
        job_id = [args.job_id or ("sp_ab_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
                  if rank == 0 else None]
        dist.broadcast_object_list(job_id, src=0, group=cpu_group)
        report["job_id"] = job_id[0]
        report["environment"] = environment_identity(torch, args.device, local_rank, torch_npu)
        report["normalization"] = "torch_npu.npu_rms_norm" if args.device == "npu" else "Torch FP32 RMSNorm, cast to BF16"
        report["status"] = "correctness"
        save_json(report_path, report)

        def synchronize():
            if args.device == "npu":
                torch.npu.synchronize()

        def align():
            synchronize()
            dist.barrier(group=cpu_group)

        gamma_cpu = (torch.arange(1, 4097, dtype=torch.float32) / 4096).to(torch.bfloat16)
        gamma = gamma_cpu.to(device)
        oracle_gamma = gamma_cpu.double().tolist()
        cases = {}
        for n in [128, 129]:
            padded, hidden = (n + 1) // 2 * 2, 4096
            parts = []
            for repetition in range(2):
                source = torch.zeros((padded, hidden), dtype=torch.bfloat16)
                source[:n] = torch.tensor([[partial_value(rank, row, col, repetition)
                                           for col in range(hidden)] for row in range(n)], dtype=torch.bfloat16)
                parts.append(source.to(device))
            cases[n] = {"padded": padded, "parts": parts,
                        "full": torch.empty((padded, hidden), dtype=torch.bfloat16, device=device),
                        "shard": torch.empty((padded // 2, hidden), dtype=torch.bfloat16, device=device),
                        "assembled": torch.empty((padded, hidden), dtype=torch.bfloat16, device=device)}

        def norm(x):
            if args.device == "npu":
                return torch_npu.npu_rms_norm(x, gamma, 1e-6)[0]
            value = x.float()
            return (value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
                    * gamma.float()).to(torch.bfloat16)

        def prepare(case, mode, repetition):
            if mode == "A":
                # AllReduce modifies its input; restore it outside the measured region.
                case["full"].copy_(case["parts"][repetition])
            align()

        def forward(case, mode, repetition, scopes=False):
            def mark(label):
                return torch.profiler.record_function(label) if scopes else nullcontext()
            if mode == "A":
                with mark("sp/A/all_reduce"):
                    dist.all_reduce(case["full"])
                with mark("sp/A/full_rmsnorm"):
                    return norm(case["full"])
            with mark("sp/B/reduce_scatter"):
                dist.reduce_scatter_tensor(case["shard"], case["parts"][repetition])
            with mark("sp/B/local_rmsnorm"):
                local = norm(case["shard"])
            with mark("sp/B/all_gather"):
                dist.all_gather_into_tensor(case["assembled"], local.contiguous())
            return case["assembled"]

        # Validate every input variant on both paths before any timed work.
        with torch.inference_mode():
            for n, case in cases.items():
                for repetition in range(2):
                    actual = {}
                    for mode in MODES:
                        prepare(case, mode, repetition)
                        actual[mode] = forward(case, mode, repetition).float().cpu().double()
                        synchronize()
                    expected = torch.tensor(reference_fp64(n, 4096, repetition, oracle_gamma, 1e-6), dtype=torch.float64)
                    for mode in MODES:
                        torch.testing.assert_close(actual[mode][:n], expected, atol=0.004, rtol=0.004)
                        if torch.count_nonzero(actual[mode][n:]).item() != 0:
                            raise AssertionError(f"Nonzero padding in {mode}, N={n}")
                    torch.testing.assert_close(actual["A"], actual["B"], atol=0.004, rtol=0.004)
                    start, end = rank * (case["padded"] // 2), (rank + 1) * (case["padded"] // 2)
                    report["correctness"].append({"valid_rows": n, "padded_rows": case["padded"],
                        "input_variant": repetition, "full_norm_shape": [case["padded"], 4096],
                        "local_norm_shape": [case["padded"] // 2, 4096],
                        "local_padded_row_interval": [start, end], "local_valid_rows": max(0, min(n, end) - start),
                        "padding_zero": True, "atol": 0.004, "rtol": 0.004,
                        "oracle": "Python FP64 math.fsum/sqrt with BF16 input/gamma values",
                        "oracle_max_abs_error": {mode: (actual[mode][:n] - expected).abs().max().item() for mode in MODES},
                        "path_max_abs_error": (actual["A"] - actual["B"]).abs().max().item(),
                        "path_exact": bool(torch.equal(actual["A"], actual["B"])),
                        "output_sum_fp64": {mode: actual[mode][:n].sum().item() for mode in MODES}})
                    save_json(report_path, report)
            align()
            report["status"] = "timing_without_profiler"
            save_json(report_path, report)
            for n, case in cases.items():
                for iteration in range(args.warmup):
                    for mode in (MODES if iteration % 2 == 0 else MODES[::-1]):
                        prepare(case, mode, iteration % 2)
                        forward(case, mode, iteration % 2)
                        synchronize()
                begin_event = end_event = None
                if args.device == "npu":
                    begin_event = torch.npu.Event(enable_timing=True)
                    end_event = torch.npu.Event(enable_timing=True)
                    # Initialize the reusable events before any measured sample.
                    begin_event.record()
                    end_event.record()
                    synchronize()
                # Profiler/record_function is not enabled in this measurement loop.
                for iteration in range(args.samples):
                    order = MODES if iteration % 2 == 0 else MODES[::-1]
                    repetition = iteration % 2
                    for mode in order:
                        prepare(case, mode, repetition)
                        start_ns = time.perf_counter_ns()
                        if begin_event is not None:
                            begin_event.record()
                        output = forward(case, mode, repetition)
                        if end_event is not None:
                            end_event.record()
                        synchronize()
                        stop_ns = time.perf_counter_ns()
                        report["samples"].append({"valid_rows": n, "padded_rows": case["padded"],
                            "iteration": iteration, "order": "".join(order), "mode": mode,
                            "input_variant": repetition, "host_ms": (stop_ns - start_ns) / 1e6,
                            "npu_event_ms": begin_event.elapsed_time(end_event) if begin_event is not None else None})
                        del output
                    save_json(report_path, report)
                align()

            # Independent trace runs: same case, mode, semantic iteration and input
            # variant on both ranks; traces never supply the timing sample table.
            if args.profile:
                report["status"] = "profiling"
                save_json(report_path, report)
                profiler = torch_npu.profiler if args.device == "npu" else torch.profiler
                activities = ([profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
                              if args.device == "npu" else [profiler.ProfilerActivity.CPU])
                for n, case in cases.items():
                    for mode in MODES:
                        relative = Path("profiles") / f"N{n}" / mode / f"rank{rank}"
                        target = args.output / relative
                        align()
                        extra = {}
                        if args.device == "npu" and args.profile_level != "default":
                            level = getattr(torch_npu.profiler.ProfilerLevel, args.profile_level)
                            extra["experimental_config"] = torch_npu.profiler._ExperimentalConfig(profiler_level=level)
                        with profiler.profile(activities=activities,
                            schedule=profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
                            on_trace_ready=profiler.tensorboard_trace_handler(str(target)),
                            record_shapes=True, profile_memory=False, with_stack=False, **extra) as prof:
                            for iteration in range(3):
                                repetition = iteration % 2
                                prepare(case, mode, repetition)
                                marker = f"wiki/{report['job_id']}/sp_ab/N{n}/{mode}/iteration{iteration}/input{repetition}/rank{rank}"
                                with torch.profiler.record_function(marker):
                                    forward(case, mode, repetition, scopes=True)
                                    synchronize()
                                prof.step()
                        report["profiles"].append({"valid_rows": n, "mode": mode,
                            "directory": str(relative), "warmup_iterations": [0], "active_iterations": [1, 2],
                            "input_variants": {"1": 1, "2": 0}, "record_shapes": True,
                            "profile_level_requested": args.profile_level if args.device == "npu" else "CPU",
                            "clock_domain": "local rank only; semantic markers permit matching, not timestamp subtraction"})
                        save_json(report_path, report)
                        align()

        report["statistics"] = []
        for n in cases:
            for mode in MODES:
                rows = [s for s in report["samples"] if s["valid_rows"] == n and s["mode"] == mode]
                report["statistics"].append({"valid_rows": n, "mode": mode,
                    "host_ms": sample_stats([s["host_ms"] for s in rows]),
                    "npu_event_ms": sample_stats([s["npu_event_ms"] for s in rows]) if args.device == "npu" else None})
        report["status"] = "passed"
        report["finished_at"] = utc_now()
        save_json(report_path, report)
        align()
        print(json.dumps({"rank": rank, "status": "passed", "device": args.device, "job_id": report["job_id"],
                          "correctness_cases": len(report["correctness"]), "samples": len(report["samples"]),
                          "profile_runs": len(report["profiles"])}), flush=True)
        succeeded = True
    except BaseException as error:
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        save_json(report_path, report)
        raise
    finally:
        # Do not risk blocking communicator destruction before a failed worker
        # can exit. The launcher must terminate/drain its peer under an outer timeout.
        if succeeded and dist is not None and dist.is_initialized():
            if args.device == "npu" and cpu_group is not None:
                dist.destroy_process_group(cpu_group)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
