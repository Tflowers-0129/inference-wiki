#!/usr/bin/env python3
"""Replay pinned DP dispatch source, optionally with real two-rank HCCL graphs.

The manager's capture catalogue and batches are controlled fixtures. The NPU
payload is an AllGather -> one scalar expert per rank -> AllReduce teaching
kernel, not vLLM Attention, MC2, MegaMoE, or a complete model. No performance
claim is produced. --self-test checks only the independent integer oracle.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import enum
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import ModuleType, SimpleNamespace

CAPTURE_SIZES = [1, 2, 4, 8]
# These are inputs to the probe, not observations of an engine scheduler.
CASES = [
    {"name": "uneven_decode", "scheduled": [3, 5], "uniform": [1, 1], "expected_mode": "FULL", "expected_rows": [8, 8]},
    {"name": "empty_partner_dummy", "scheduled": [0, 5], "uniform": [1, 1], "expected_mode": "FULL", "expected_rows": [8, 8]},
    {"name": "mixed_prefill", "scheduled": [3, 5], "uniform": [1, None], "expected_mode": "NONE", "expected_rows": [3, 5]},
    {"name": "outside_capture_catalogue", "scheduled": [3, 9], "uniform": [1, 1], "expected_mode": "NONE", "expected_rows": [3, 9]},
    {"name": "all_idle", "scheduled": [0, 0], "uniform": [1, 1], "expected_mode": "NONE", "expected_rows": [0, 0]},
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def fixture_input(case, rank, repetition, width=4):
    count = case["scheduled"][rank]
    return [
        [100 * repetition + 10 * rank + row + col + 1 for col in range(width)]
        for row in range(count)
    ]


def reference_output(case, rank, repetition, width=4):
    rows = fixture_input(case, rank, repetition, width)
    return [[value * ((row + rank + repetition) % 2 + 1) for value in values]
            for row, values in enumerate(rows)]


def self_test():
    # An independent destination-centric loop must agree with the direct oracle.
    tested = 0
    for case in CASES:
        for repetition in range(2):
            for owner in range(2):
                rows = fixture_input(case, owner, repetition)
                assembled = [[0] * 4 for _ in rows]
                for expert in range(2):
                    for row, values in enumerate(rows):
                        if (row + owner + repetition) % 2 == expert:
                            for col, value in enumerate(values):
                                assembled[row][col] += value * (expert + 1)
                assert assembled == reference_output(case, owner, repetition)
                tested += 1
    assert reference_output(CASES[0], 0, 0) == [[1, 2, 3, 4], [4, 6, 8, 10], [3, 4, 5, 6]]
    assert reference_output(CASES[1], 0, 1) == []
    return {"status": "passed", "scope": "local Python integer oracle only", "checks": tested + 2,
            "production_source_executed": False, "device_used": False,
            "graph_capture_executed": False}


def source_nodes(path, names, owner=None):
    data = path.read_bytes()
    tree = ast.parse(data.decode(), filename=path.name)
    body = tree.body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    nodes = [next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
             for name in names]
    identity = {"file": path.name, "sha256": digest(data), "nodes": [
        {"name": n.name, "line": n.lineno, "end_line": n.end_lineno,
         "ast_sha256": digest(ast.dump(n, include_attributes=False).encode())} for n in nodes]}
    return nodes, identity


def load_protocol(source_dir, torch, dist, cpu_group):
    # Exact AST nodes execute with real CPU tensors and the real Gloo group.
    # Module imports, model construction and GPU/Ascend patching are not run.
    manifest = json.loads((source_dir / "sources.json").read_text())
    for item in manifest["sources"]:
        path = source_dir / item["file"]
        if digest(path.read_bytes()) != item["sha256"]:
            raise RuntimeError(f"Source changed: {path.name}; audit and update sources.json first")
    selections = [
        ("compilation.py", ["CUDAGraphMode"], None),
        ("cudagraph_utils.py", ["BatchExecutionDescriptor", "_is_compatible"], None),
        ("cudagraph_utils.py", ["_resolve_effective_loras", "dispatch"], "CudaGraphManager"),
        ("dp_utils.py", ["DPSyncState", "sync_cudagraph_and_dp_padding", "dispatch_cg_and_sync_dp"], None),
    ]
    body, identities = [], []
    for filename, names, owner in selections:
        nodes, identity = source_nodes(source_dir / filename, names, owner)
        identities.append(identity)
        if owner:
            nodes = [ast.ClassDef(name=owner, bases=[], keywords=[], body=nodes, decorator_list=[])]
        body.extend(nodes)
    module_ast = ast.parse("from __future__ import annotations\n")
    module_ast.body.extend(body)
    ast.fix_missing_locations(module_ast)
    module = ModuleType("wiki_pinned_dp_protocol")
    sys.modules[module.__name__] = module
    module.__dict__.update({"dataclass": dataclass, "replace": replace, "enum": enum,
                           "torch": torch, "dist": dist,
                           "get_dp_group": lambda: SimpleNamespace(cpu_group=cpu_group),
                           "should_skip_dp_coordination": lambda: False})
    exec(compile(module_ast, "<pinned DP protocol AST>", "exec"), module.__dict__)
    manager = module.CudaGraphManager()
    manager._graphs_captured = True
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    descs = [module.BatchExecutionDescriptor(cg_mode=module.CUDAGraphMode.FULL,
              num_tokens=n, num_reqs=n, uniform_token_count=1) for n in CAPTURE_SIZES]
    # Candidate index is a fixture; its creation/capture is not under test.
    manager._candidates = {(n, 0): [d for d in descs if d.num_tokens >= n]
                           for n in range(1, max(CAPTURE_SIZES) + 1)}
    return module, manager, {"manifest": manifest, "extracted_nodes": identities,
                             "method_bodies_modified": False,
                             "controlled_fields": ["capture catalogue", "get_dp_group", "should_skip_dp_coordination=False"],
                             "excluded": ["DBO", "LoRA", "speculation", "varlen decode", "model runner", "attention"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("captures/parallel-graph"))
    parser.add_argument("--reservation-id", default="")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    if args.source_dir is None:
        parser.error("--source-dir with audited source files and sources.json is required")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        parser.error("Launch with torchrun --nproc-per-node=2")
    if args.device == "npu" and not args.reservation_id:
        parser.error("Supply --reservation-id for an externally verified exclusive device reservation")
    import torch
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    if args.device == "npu":
        import torch_npu
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("hccl", timeout=timedelta(seconds=90))
        cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=90))
        cpu_group = dist.group.WORLD
    module, manager, source_identity = load_protocol(args.source_dir, torch, dist, cpu_group)
    result = {"schema_version": "wiki-parallel-graph/1", "started_at": datetime.now(timezone.utc).isoformat(),
              "rank": rank, "world_size": 2, "device": args.device, "status": "running",
              "script_sha256": digest(Path(__file__).read_bytes()), "sources": source_identity,
              "environment": {"python": platform.python_version(), "torch": torch.__version__},
              "fixture": {"topology": "two DP participants, one scalar expert each", "capture_sizes": CAPTURE_SIZES,
                          "width": 4, "dtype": "float32", "repetitions": 2},
              "capture_events": [], "cases": [],
              "scope": "production DP source replay with fixture capture catalogue; optional teaching expert kernel with actual HCCL/NPUGraph; no vLLM model or service timing"}
    if args.device == "npu":
        result["environment"].update({"torch_npu": torch_npu.__version__, "device_name": torch.npu.get_device_name(),
            "HCCL_OP_EXPANSION_MODE": os.environ.get("HCCL_OP_EXPANSION_MODE", "unset")})
    report_path = args.output / f"rank{rank}.json"
    write_json(report_path, result)
    graphs = {}

    def make_payload(capacity):
        x = torch.zeros((capacity, 4), dtype=torch.float32, device="npu")
        destinations = torch.full((capacity,), -1, dtype=torch.int32, device="npu")
        gathered = torch.empty((2 * capacity, 4), dtype=torch.float32, device="npu")
        gathered_destinations = torch.empty((2 * capacity,), dtype=torch.int32, device="npu")
        output = torch.empty_like(gathered)

        def forward():
            dist.all_gather_into_tensor(gathered, x)
            dist.all_gather_into_tensor(gathered_destinations, destinations)
            output.copy_(gathered * (gathered_destinations == rank).unsqueeze(1) * (rank + 1))
            dist.all_reduce(output)
        return (x, destinations, gathered, gathered_destinations, output), forward

    try:
        if args.device == "npu":
            for capacity in CAPTURE_SIZES:
                tensors, forward = make_payload(capacity)
                for _ in range(3):
                    forward()
                torch.npu.synchronize()
                dist.barrier(group=cpu_group)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    forward()
                graph.replay()
                torch.npu.synchronize()
                assert torch.count_nonzero(tensors[-1]).item() == 0
                graphs[capacity] = (graph, tensors, [t.data_ptr() for t in tensors])
                result["capture_events"].append({"capacity": capacity, "capture_returned": True,
                                                 "zero_input_replay_passed": True})
                write_json(report_path, result)
                dist.barrier(group=cpu_group)

        for case in CASES:
            live = any(case["scheduled"])
            # Empty workers use a one-row dummy only while a peer has real work.
            protocol_tokens = max(1, case["scheduled"][rank]) if live else 0
            uniform = case["uniform"][rank]
            requests = protocol_tokens if uniform == 1 else int(protocol_tokens > 0)
            desc, sync = module.dispatch_cg_and_sync_dp(manager, requests, protocol_tokens, uniform, 2, rank)
            assert desc.cg_mode.name == case["expected_mode"]
            assert desc.num_tokens == case["expected_rows"][rank]
            row = {"name": case["name"], "scheduled_tokens": case["scheduled"],
                   "local_protocol_tokens": protocol_tokens, "mode": desc.cg_mode.name,
                   "descriptor_tokens": desc.num_tokens,
                   "tokens_across_dp": sync.num_tokens_across_dp.tolist() if sync else None,
                   "uniform_token_count": sync.uniform_token_count if sync else None,
                   "protocol_assertions": "passed", "device_checks": []}
            if args.device == "npu" and live:
                capacity = desc.num_tokens if desc.cg_mode.name == "FULL" else max(case["scheduled"])
                if desc.cg_mode.name == "FULL":
                    graph, tensors, pointers = graphs[capacity]
                    forward = graph.replay
                else:
                    tensors, forward = make_payload(capacity)
                    pointers = [t.data_ptr() for t in tensors]
                for repetition in range(2):
                    inputs = fixture_input(case, rank, repetition)
                    destinations = [(i + rank + repetition) % 2 for i in range(len(inputs))]
                    tensors[0].zero_()
                    tensors[1].fill_(-1)
                    if inputs:
                        tensors[0][:len(inputs)].copy_(torch.tensor(inputs, dtype=torch.float32))
                        tensors[1][:len(inputs)].copy_(torch.tensor(destinations, dtype=torch.int32))
                    torch.npu.synchronize()
                    dist.barrier(group=cpu_group)
                    if args.profile:
                        target = str(args.output / "profiles" / f"rank{rank}" / case["name"] / str(repetition))
                        with torch_npu.profiler.profile(
                            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(target),
                            record_shapes=True, with_stack=False,
                        ):
                            with torch.profiler.record_function(f"wiki/{case['name']}/rep{repetition}/{desc.cg_mode.name}"):
                                forward()
                                torch.npu.synchronize()
                    else:
                        forward()
                        torch.npu.synchronize()
                    actual = tensors[-1][rank * capacity:(rank + 1) * capacity].cpu().tolist()
                    expected = reference_output(case, rank, repetition)
                    assert actual[:len(inputs)] == expected
                    assert all(value == 0 for r in actual[len(inputs):] for value in r)
                    assert [t.data_ptr() for t in tensors] == pointers
                    row["device_checks"].append({"repetition": repetition, "valid_rows": len(inputs),
                        "payload_capacity_per_rank": capacity, "exact_output": True, "padding_zero": True,
                        "buffer_addresses_unchanged": True, "graph_replay": desc.cg_mode.name == "FULL",
                        "output_checksum": sum(sum(r) for r in actual)})
            result["cases"].append(row)
            write_json(report_path, result)
        result["status"] = "passed"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, result)
        dist.barrier(group=cpu_group)
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
        write_json(report_path, result)
        raise
    finally:
        # Collective timeout is bounded; the outer launcher must still drain
        # both worker processes if a peer exits during a graph collective.
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
