#!/usr/bin/env python3
"""Export observed PD Host call boundaries to a Chrome/Perfetto trace.

Example:
  python export_host_trace.py --root PRIVATE/pull \
    --summary PRIVATE/pull/lifecycle-summary.json \
    --clock-domain same-host-monotonic --request-key q0123456789abcdef \
    --output PRIVATE/pull-host.json

The clock option asserts that all input logs were captured on the same host.
This exporter does not establish that fact. It uses one shared monotonic origin
for the whole input, including when a request is selected. It never imports or
aligns CANN/NPU timestamps. A slice spans an observed begin and its matching end;
it includes observer overhead. An end without an observed begin is an instant,
even if the log contains a body-duration counter. Request aliases come only
from the supplied analyzer summary's explicit edges. No transfer, wait, device
execution, or physical-memory reuse interval is synthesized.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from analyze_lifecycle import primary_ids, valid_id

_SCHEMA = "wiki-pd-lifecycle/1"
_SAFE_KEYS = frozenset("""
request_id req_id remote_request_id request_ids finished_req_ids failed_req_ids
input_request_id output_request_id request_aliases request req request_metadata
req_meta request_data requests send_requests metadata connector_output
kv_connector_output result transfer_context transfer_entries transfer_flags
requests_to_send reqs_in_batch finished_sending finished_recving failed_recving
num_prompt_tokens num_computed_tokens num_external_tokens local_num_prompt_tokens
remote_num_prompt_tokens prompt_len local_computed_tokens local_transed_tokens
remote_cache_tokens chunk_finish do_virtual all_task_done shard_idx awaiting_kvs
status before_status do_remote_prefill do_remote_decode remote_tp_size
remote_pcp_size remote_dcp_size num_scheduled_tokens total_num_scheduled_tokens
local_block_ids remote_block_ids local_full_block_ids allocated_block_ids block_ids
layer_idx layer_name trans_count group_idx trans_flag delay_free_blocks
remote_pp_rank remote_pcp_rank remote_tp_rank report_completion
transfer_bytes transfer_ranges done_source_count pending_pull_tasks spec_index
layer_pair done_requests failed_requests _pending_reuse_layers delay_free
has_transfer_params matched_tokens load_async return_code all_tasks_done proceed
transfer_error exception_type
""".split())
_STATIC_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.\[\]-]{0,199}\Z")
_STATUS = re.compile(r"[A-Z][A-Z0-9_]{0,99}\Z")
_EVENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,199}\Z")
_OMIT = object()


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def safe(value, key=None, depth=0):
    """Retain allowlisted counters, block IDs, static names and hashed IDs."""
    if depth > 10:
        return _OMIT
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if valid_id(value):
            return value
        if key in ("layer_name", "exception_type") and _STATIC_NAME.fullmatch(value):
            return value
        if key in ("status", "before_status") and _STATUS.fullmatch(value):
            return value
        return _OMIT
    if isinstance(value, list):
        output = []
        for item in value[:1024]:
            cleaned = safe(item, key, depth + 1)
            if cleaned is not _OMIT:
                output.append(cleaned)
        return output
    if isinstance(value, dict):
        output = {}
        for name, item in value.items():
            if not isinstance(name, str) or (name not in _SAFE_KEYS and not valid_id(name)):
                continue
            cleaned = safe(item, name, depth + 1)
            if cleaned is not _OMIT:
                output[name] = cleaned
        return output
    return _OMIT


class Aliases:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.root(self.parent[key])
        return self.parent[key]

    def link(self, left, right):
        a, b = self.root(left), self.root(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def load_rows(root):
    sources, rows, issues = [], [], []
    paths = sorted((root / "lifecycle").rglob("*.jsonl"))
    if not paths:
        raise ValueError("No lifecycle JSONL found under root/lifecycle")
    for number, path in enumerate(paths):
        blob = path.read_bytes()
        source = f"lifecycle-{number:03d}.jsonl"
        sources.append({"source": source, "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
        for line_number, line in enumerate(blob.decode("utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                if (not isinstance(row, dict) or row.get("schema") != _SCHEMA
                        or not integer(row.get("monotonic_ns"))
                        or not integer(row.get("pid")) or not integer(row.get("tid"))
                        or not isinstance(row.get("event"), str)
                        or not _EVENT.fullmatch(row["event"])):
                    raise ValueError("row_schema")
            except (ValueError, TypeError, KeyError):
                issues.append({"source": source, "line": line_number, "issue": "invalid_or_incomplete_row"})
                continue
            # Installation records have no observed method-call interval.
            if row.get("phase") not in ("begin", "end", "error"):
                continue
            rows.append({"data": row, "ref": {"source": source, "line": line_number,
                                               "seq": row.get("seq") if integer(row.get("seq")) else None}})
    if not rows:
        raise ValueError("No runtime begin/end/error observations found")
    return sources, rows, issues


def load_aliases(summary_path, sources):
    aliases, edges = Aliases(), []
    if summary_path is None:
        return aliases, edges, None
    blob = summary_path.read_bytes()
    summary = json.loads(blob)
    if summary.get("schema") != "wiki-pd-lifecycle-summary/1":
        raise ValueError("Unsupported lifecycle summary schema")
    if summary.get("clock_domain") != "same-host-monotonic":
        raise ValueError("Summary must explicitly use same-host-monotonic")
    if summary.get("sources") != sources:
        raise ValueError("Summary source manifest differs from the current JSONL input")
    for edge in summary.get("alias_edges", []):
        left, right = edge.get("left"), edge.get("right")
        if not (valid_id(left) and valid_id(right)):
            raise ValueError("Invalid explicit alias edge")
        aliases.link(left, right)
        # Keep the analyzer's actual edge evidence; do not infer edges from order.
        clean_edge = {"left": left, "right": right}
        if isinstance(edge.get("event"), str) and _EVENT.fullmatch(edge["event"]):
            clean_edge["event"] = edge["event"]
        if edge.get("phase") in ("begin", "end", "error"):
            clean_edge["phase"] = edge["phase"]
        if integer(edge.get("seq")):
            clean_edge["seq"] = edge["seq"]
        if edge.get("role") in ("P", "D", "U"):
            clean_edge["role"] = edge["role"]
        edges.append(clean_edge)
    return aliases, edges, hashlib.sha256(blob).hexdigest()


def observed_calls(rows):
    pending = defaultdict(list)
    calls = []
    for item in sorted(rows, key=lambda x: (x["data"]["monotonic_ns"], x["ref"]["source"], x["ref"]["line"])):
        row = item["data"]
        key = (row["pid"], row["tid"], row["event"])
        if row["phase"] == "begin":
            pending[key].append(item)
        else:
            begin = pending[key].pop() if pending[key] else None
            calls.append({"begin": begin, "end": item})
    for items in pending.values():
        calls.extend({"begin": item, "end": None} for item in items)
    return calls


def identity(row):
    role = row.get("role") if row.get("role") in ("P", "D") else "U"
    rank = str(row.get("tp_rank", "U"))
    if rank not in ("S", "U") and not rank.isdecimal():
        rank = "U"
    return role, rank


def label_identity(roles):
    labels = []
    for role, rank in sorted(roles):
        labels.append(role + ("/Scheduler" if rank == "S" else "/rank" + rank))
    return " + ".join(labels)


def export(root, summary_path=None, request_keys=(), window=None):
    sources, rows, issues = load_rows(root)
    aliases, edges, summary_sha = load_aliases(summary_path, sources)
    # A shared input origin is retained even for a selected request/window.
    origin = min(item["data"]["monotonic_ns"] for item in rows)
    selected = {aliases.root(key) for key in request_keys}
    trace, references = [], []
    process_roles, thread_roles, thread_events = defaultdict(set), defaultdict(set), defaultdict(set)
    all_request_windows = {}
    for call in observed_calls(rows):
        begin, end = call["begin"], call["end"]
        first, last = begin or end, end or begin
        first_row, last_row = first["data"], last["data"]
        original_ids = set()
        for item in (begin, end):
            if item is not None:
                original_ids |= primary_ids(item["data"])
        normalized = {aliases.root(key) for key in original_ids}
        if selected and not selected.intersection(normalized):
            continue
        start_us = (first_row["monotonic_ns"] - origin) / 1000
        end_us = (last_row["monotonic_ns"] - origin) / 1000
        if window is not None and (end_us < window[0] or start_us > window[1]):
            continue
        merged = {**(begin["data"] if begin else {}), **(end["data"] if end else {})}
        role, rank = identity(last_row)
        # Initializers may populate rank fields between entry and exit. Both
        # recorded identities remain visible in arguments rather than fabricated.
        pid, tid = first_row["pid"], first_row["tid"]
        for item in (begin, end):
            if item is not None:
                process_roles[pid].add(identity(item["data"]))
                thread_roles[(pid, tid)].add(identity(item["data"]))
        thread_events[(pid, tid)].add(merged["event"])
        args = safe(merged)
        if not isinstance(args, dict):
            args = {}
        args.update(observed_role=role, observed_tp_rank=rank,
                    original_hashed_request_ids=sorted(original_ids),
                    normalized_request_keys=sorted(normalized),
                    observer_phase=last_row["phase"],
                    begin_identity={"role": identity(begin["data"])[0], "tp_rank": identity(begin["data"])[1]} if begin else None,
                    end_identity={"role": identity(end["data"])[0], "tp_rank": identity(end["data"])[1]} if end else None,
                    begin_ref=begin["ref"] if begin else None,
                    end_ref=end["ref"] if end else None)
        if integer(merged.get("duration_ns")):
            args["reported_instrumented_body_duration_ns"] = merged["duration_ns"]
        paired = begin is not None and end is not None
        args["observation"] = ("begin/end Host envelope including observer overhead" if paired
                               else "end-only Host observation; no interval reconstructed" if end
                               else "begin-only Host observation; no completion observed")
        event = {"name": merged["event"], "cat": "observed_host_call" if paired else "observed_host_instant",
                 "ph": "X" if paired else "i", "ts": start_us, "pid": pid, "tid": tid, "args": args}
        if paired:
            if end_us < start_us:
                raise ValueError("Negative observed begin/end envelope")
            event["dur"] = (last_row["monotonic_ns"] - first_row["monotonic_ns"]) / 1000
        else:
            event["s"] = "t"
        event_index = len(trace)
        trace.append(event)
        references.append({"event_index": event_index, "event": merged["event"], "ph": event["ph"],
                           "pid": pid, "tid": tid, "role": role, "tp_rank": rank,
                           "begin_us": start_us if begin else None, "end_us": end_us if end else None,
                           "envelope_duration_us": event.get("dur"),
                           "original_hashed_request_ids": sorted(original_ids),
                           "normalized_request_keys": sorted(normalized),
                           "begin_ref": begin["ref"] if begin else None,
                           "end_ref": end["ref"] if end else None})
        for key in normalized:
            span = all_request_windows.setdefault(key, [start_us, end_us])
            span[0], span[1] = min(span[0], start_us), max(span[1], end_us)
    if not trace:
        raise ValueError("No observations match the explicit selection")
    metadata = []
    for order, pid in enumerate(sorted(process_roles, key=lambda p: (min({"P": 0, "D": 1, "U": 2}[role] for role, _ in process_roles[p]), label_identity(process_roles[p]), p))):
        name = "Observed Host / " + label_identity(process_roles[pid]) + f" / pid {pid}"
        metadata.extend([
            {"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": name}},
            {"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0, "args": {"sort_index": order}},
        ])
        for thread_order, (process, tid) in enumerate(sorted(key for key in thread_roles if key[0] == pid)):
            has_te = any(name.startswith("TransferEngine.") for name in thread_events[(process, tid)])
            kind = "TE calls" if has_te else "observed calls"
            name = label_identity(thread_roles[(process, tid)]) + f" / {kind} / tid {tid}"
            metadata.extend([
                {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}},
                {"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": thread_order}},
            ])
    limits = [
        "Observed Python Host call envelopes and end-only instants; not CANN-native hardware communication lanes.",
        "One shared Host monotonic origin for every process/rank; no per-request or per-rank zeroing.",
        "No NPU timestamps were imported or aligned; the clock-domain assertion must be independently checked by the operator.",
        "Envelope durations include observer timestamping, logging and Python overhead; they are diagnostic and not a latency benchmark.",
        "An end-only polling result remains an instant even when a body-duration counter exists.",
        "Request identity joins use only the supplied analyzer's explicit alias edges; no ordinal or duration-based joins.",
        "A TE return, request bookkeeping and actual allocator reuse/device computation are separate observations.",
    ]
    common = {"schema": "wiki-pd-observed-host-trace/1", "clock_domain": "same-host-monotonic",
              "clock_scope": "Host only", "timestamp_unit": "us", "shared_origin_monotonic_ns": origin,
              "origin_selection": "minimum monotonic_ns among all runtime rows in the complete input, before filtering",
              "summary_sha256": summary_sha, "sources": sources, "alias_edges": edges,
              "selection": {"request_keys": sorted(selected), "window_us": list(window) if window else None,
                            "intervals_are_clipped": False},
              "observations": dict(Counter(event["ph"] for event in trace)),
              "window_us": [min(e["begin_us"] if e["begin_us"] is not None else e["end_us"] for e in references),
                            max(e["end_us"] if e["end_us"] is not None else e["begin_us"] for e in references)],
              "request_windows_us": {key: value for key, value in sorted(all_request_windows.items())},
              "issues": issues, "evidence_limits": limits}
    return ({"traceEvents": metadata + trace, "displayTimeUnit": "ms", "otherData": common},
            {**common, "event_index_basis": "non-metadata trace event order before metadata prefix", "events": references})


def parse_window(value):
    try:
        start, end = (float(item) for item in value.split(":"))
        if not (0 <= start <= end < float("inf")):
            raise ValueError()
        return start, end
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected finite nonnegative START_US:END_US") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Analyzer JSON with matching source hashes and explicit alias edges.")
    parser.add_argument("--clock-domain", choices=("same-host-monotonic",), required=True)
    parser.add_argument("--request-key", action="append", default=[], help="Observed hashed request ID or an explicitly linked alias; repeatable.")
    parser.add_argument("--window-us", type=parse_window, help="Filter on the shared input axis without clipping observed intervals.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="Default: output basename with .refs.json suffix.")
    args = parser.parse_args()
    if any(not valid_id(key) for key in args.request_key):
        parser.error("--request-key must be an observed q + 16-hex-digit identifier")
    try:
        trace, refs = export(args.root, args.summary, args.request_key, args.window_us)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    output_refs = args.references or args.output.with_suffix(".refs.json")
    for path, value in ((args.output, trace), (output_refs, refs)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"observations": refs["observations"], "window_us": refs["window_us"],
                      "request_keys": refs["selection"]["request_keys"], "issues": len(refs["issues"]),
                      "output": str(args.output), "references": str(output_refs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
