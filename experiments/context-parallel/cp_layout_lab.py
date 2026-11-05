#!/usr/bin/env python3
"""Replay one installed PCP partition method without importing its module.

This CPU-only source replay compiles the unchanged AST node for
AscendPCPManager._iter_rank_chunks. Its controlled receiver exposes only
pcp_world_size and is_decode_sharded=True. It does not run a model runner,
attention, tensor collectives, torch, or NPU kernels. The TP memory table is
separate teaching arithmetic, not an execution of QKVParallelLinear.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_partition_method(source_path):
    """Extract the exact function AST; do not import the source module."""
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source, filename=str(source_path))
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendPCPManager"
    )
    node = next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_iter_rank_chunks"
    )
    if node.decorator_list:
        raise RuntimeError("Unexpected decorators; re-audit the target method")
    node_source = ast.get_source_segment(source, node)
    if node_source is None:
        raise RuntimeError("Could not recover the exact source node")
    # The future import postpones the np/Iterator annotations. Only this method
    # is compiled; module imports and class initialization never execute.
    module = ast.parse("from __future__ import annotations\n")
    module.body.append(node)
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    method = namespace["_iter_rank_chunks"]
    return method, {
        "source_file": str(source_path),
        "source_file_sha256": sha256(source_bytes),
        "qualified_name": "AscendPCPManager._iter_rank_chunks",
        "node_source_sha256": sha256(node_source.encode("utf-8")),
        "node_ast_sha256": sha256(
            ast.dump(node, include_attributes=False).encode("utf-8")
        ),
        "node_start_line": node.lineno,
        "node_end_line": node.end_lineno,
        "method_body_modified": False,
        "module_imported": False,
        "controlled_receiver": {
            "pcp_world_size": "set per case",
            "is_decode_sharded": True,
        },
    }


def replay_case(method, name, world_size, query_lengths, is_prefilling):
    receiver = SimpleNamespace(
        pcp_world_size=world_size, is_decode_sharded=True
    )
    ranks = []
    observed = {request: [] for request in range(len(query_lengths))}
    for rank in range(world_size):
        chunks = list(method(receiver, rank, query_lengths, is_prefilling))
        requests = []
        for request in range(len(query_lengths)):
            positions = [
                position
                for req, start, length in chunks if req == request
                for position in range(start, start + length)
            ]
            assert len(positions) == len(set(positions)), (name, rank, request)
            assert all(0 <= p < query_lengths[request] for p in positions)
            observed[request].extend((p, rank) for p in positions)
            if positions:
                requests.append({
                    "request_index": request,
                    "query_offsets": positions,
                    "is_prefilling": is_prefilling[request],
                    "fresh_causal_query_key_pairs": (
                        sum(p + 1 for p in positions)
                        if is_prefilling[request] else None
                    ),
                })
        ranks.append({
            "rank": rank,
            "chunks": [list(chunk) for chunk in chunks],
            "requests": requests,
            "query_count": sum(item[2] for item in chunks),
        })
    # Every scheduled query is assigned once, including short/nondivisible
    # prefills and zero-token rows. Do not presume all ranks are nonempty.
    for request, length in enumerate(query_lengths):
        assert sorted(p for p, _rank in observed[request]) == list(range(length)), (
            name, request, observed[request]
        )
        if not is_prefilling[request] and length:
            assert len({rank for _p, rank in observed[request]}) == 1
    assert sum(rank["query_count"] for rank in ranks) == sum(query_lengths)
    return {
        "name": name,
        "pcp_world_size": world_size,
        "dcp_size_condition": 1,
        "query_lengths": query_lengths,
        "is_prefilling": is_prefilling,
        "ranks": ranks,
        "coverage_and_unique_ownership": "passed",
    }


def positions_for(result, rank, request=0):
    matches = [
        item["query_offsets"] for item in result["ranks"][rank]["requests"]
        if item["request_index"] == request
    ]
    return matches[0] if matches else []


def teaching_tp_table():
    query_heads, kv_heads, head_dim = 16, 2, 128
    batch, sequence, dtype_bytes = 1, 16384, 2
    rows = []
    for tp in [1, 2, 4]:
        local_kv_heads = max(1, kv_heads // tp)
        local_bytes = batch * sequence * local_kv_heads * head_dim * dtype_bytes * 2
        rows.append({
            "tp_size": tp,
            "query_heads_per_rank": query_heads // tp,
            "kv_heads_per_rank": local_kv_heads,
            "kv_head_replicas": max(1, tp // kv_heads),
            "per_rank_valid_kv_bytes_one_layer": local_bytes,
            "per_rank_valid_kv_mib_one_layer": local_bytes / (1024 ** 2),
            "group_valid_kv_bytes_one_layer": local_bytes * tp,
        })
    assert [row["per_rank_valid_kv_mib_one_layer"] for row in rows] == [16, 8, 8]
    return {
        "scope": "Independent teaching arithmetic; does not execute a TP class",
        "conditions": {
            "query_heads": query_heads, "kv_heads": kv_heads,
            "head_dim": head_dim, "batch": batch, "sequence_length": sequence,
            "dtype": "BF16", "dtype_bytes": dtype_bytes, "layers": 1,
            "ordinary_separate_k_and_v": True,
            "excludes": ["block padding", "scales", "pool reserve", "weights"],
        },
        "formula": "B * S * local_kv_heads * head_dim * dtype_bytes * 2",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    method, source_identity = load_partition_method(args.source)
    specifications = [
        ("L8_P2", 2, [8], [True]),
        ("L16_P4", 4, [16], [True]),
        ("L7_P2_nondivisible", 2, [7], [True]),
        ("L11_P3_nondivisible", 3, [11], [True]),
        ("L1_P4_short", 4, [1], [True]),
        ("L3_P4_short", 4, [3], [True]),
        ("L0_P4_zero", 4, [0], [True]),
        ("mixed_decode_prefill_zero", 2, [1, 8, 1, 0, 3, 1],
         [False, True, False, False, False, False]),
    ]
    repetitions = []
    for repetition in range(2):
        cases = [replay_case(method, *spec) for spec in specifications]
        assert positions_for(cases[0], 0) == [0, 1, 6, 7]
        assert positions_for(cases[0], 1) == [2, 3, 4, 5]
        assert [
            rank["requests"][0]["fresh_causal_query_key_pairs"]
            for rank in cases[1]["ranks"]
        ] == [34, 34, 34, 34]
        for request, owner in {0: 0, 2: 1, 4: 0, 5: 1}.items():
            assert positions_for(cases[-1], owner, request) == list(
                range(cases[-1]["query_lengths"][request])
            )
        repetitions.append({"repetition": repetition, "cases": cases})
    result = {
        "schema": "wiki-cp-layout-source-replay/1",
        "correctness": "passed",
        "scope": "CPU source-method replay; not full Runner or NPU execution",
        "source": source_identity,
        "repetitions": repetitions,
        "case_count_per_repetition": len(specifications),
        "causal_pair_count_boundary": (
            "Prefill counts assume a fresh standard causal sequence without "
            "a cached prefix. Pair counts are arithmetic, not device durations."
        ),
        "teaching_tp_memory": teaching_tp_table(),
        "limits": [
            "The controlled receiver fixes the decode-sharded PCP-only branch.",
            "No assertion is made about PCP+DCP replicated-short-prefill policy.",
            "No imports, metadata construction, cache writes, or collectives of the source module execute.",
            "Coverage success does not establish full-model PCP correctness or speedup.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "correctness": result["correctness"], "scope": result["scope"],
        "repetitions": len(repetitions), "cases_per_repetition": len(specifications),
        "source_file_sha256": source_identity["source_file_sha256"],
        "node_source_sha256": source_identity["node_source_sha256"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
