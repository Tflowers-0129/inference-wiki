#!/usr/bin/env python3
"""Summarize observer JSONL without guessing request aliases or clock alignment.

Example:
  python analyze_lifecycle.py --root PRIVATE/push --clock-domain same-host-monotonic --output summary.json

--root contains result.json and lifecycle/**/*.jsonl. Absolute clocks, endpoints,
inputs, exception messages and raw request names are not copied to the summary.
An explicit request-ID edge is required before comparing different identities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

HASH_ID = re.compile(r"q[0-9a-f]{16}\Z")
REQUEST_FIELDS = ("request", "req", "request_metadata", "req_meta", "request_data")
COMPLETION_FIELDS = ("finished_sending", "finished_recving", "failed_recving", "request_ids")


def valid_id(value):
    return isinstance(value, str) and HASH_ID.fullmatch(value) is not None


def ids_in(value):
    if valid_id(value):
        return {value}
    if isinstance(value, (list, tuple)):
        return {item for item in value if valid_id(item)}
    return set()


class Aliases:
    def __init__(self):
        self.parent = {}
        self.evidence = []

    def root(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.root(self.parent[key])
        return self.parent[key]

    def link(self, left, right, evidence):
        if not (valid_id(left) and valid_id(right)):
            return
        a, b = self.root(left), self.root(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)
        if left != right:
            self.evidence.append({"left": left, "right": right, **evidence})


def primary_ids(row):
    """Only arguments/results belonging to this invocation, not ambient sets."""
    output = set()
    for field in ("request_id", "req_id", "remote_request_id"):
        output |= ids_in(row.get(field))
    for field in REQUEST_FIELDS:
        value = row.get(field)
        if isinstance(value, dict):
            for key in ("request_id", "req_id", "remote_request_id"):
                output |= ids_in(value.get(key))
    for field in ("requests", "send_requests"):
        value = row.get(field)
        if isinstance(value, dict):
            output.update(key for key in value if valid_id(key))
    for field in ("metadata", "connector_output", "kv_connector_output"):
        value = row.get(field)
        if isinstance(value, dict):
            requests = value.get("requests", {})
            if isinstance(requests, dict):
                output.update(key for key in requests if valid_id(key))
            for key in COMPLETION_FIELDS:
                output |= ids_in(value.get(key))
    context = row.get("transfer_context")
    if isinstance(context, dict):
        output |= ids_in(context.get("request_ids"))
    for field in ("request_ids", "finished_req_ids", "failed_req_ids"):
        output |= ids_in(row.get(field))
    result = row.get("result")
    if isinstance(result, dict):
        for key in COMPLETION_FIELDS:
            output |= ids_in(result.get(key))
        requests = result.get("requests", {})
        if isinstance(requests, dict):
            output.update(key for key in requests if valid_id(key))
        scheduled = result.get("num_scheduled_tokens")
        if isinstance(scheduled, dict):
            output.update(key for key in scheduled if valid_id(key))
    for entry in row.get("transfer_entries", []):
        if isinstance(entry, dict):
            output |= ids_in(entry.get("request_id"))
    # Alias observers name the input/output roles explicitly.
    for key in ("input_request_id", "output_request_id"):
        output |= ids_in(row.get(key))
    return output


def learn_aliases(row, aliases):
    evidence = {"event": row.get("event"), "phase": row.get("phase"), "seq": row.get("seq"), "role": row.get("role")}
    for value in [row] + [row[k] for k in REQUEST_FIELDS if isinstance(row.get(k), dict)]:
        local = value.get("request_id") or value.get("req_id")
        aliases.link(local, value.get("remote_request_id"), evidence)
    for parent in (row, row.get("metadata", {}), row.get("result", {})):
        if not isinstance(parent, dict):
            continue
        for name in ("requests", "send_requests"):
            values = parent.get(name, {})
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, dict):
                        aliases.link(key, value.get("remote_request_id"), evidence)
    mappings = row.get("request_aliases")
    if isinstance(mappings, dict):
        for left, right in mappings.items():
            aliases.link(left, right, evidence)
    aliases.link(row.get("input_request_id"), row.get("output_request_id"), evidence)
    for key in primary_ids(row):
        aliases.root(key)


def load(root):
    rows, issues, sources = [], [], []
    paths = sorted((root / "lifecycle").rglob("*.jsonl"))
    if not paths:
        raise ValueError("No lifecycle JSONL found under root/lifecycle")
    for number, path in enumerate(paths):
        blob = path.read_bytes()
        sources.append({"source": f"lifecycle-{number:03d}.jsonl", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
        for line_number, line in enumerate(blob.decode("utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                if row.get("schema") != "wiki-pd-lifecycle/1":
                    raise ValueError("schema")
                if not isinstance(row.get("monotonic_ns"), int):
                    raise ValueError("monotonic_ns")
                rows.append(row)
            except (ValueError, TypeError, KeyError):
                issues.append({"source": number, "line": line_number, "issue": "invalid_or_incomplete_row"})
    result_path = root / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    return rows, result, issues, sources


def spans(rows):
    stacks = defaultdict(list)
    output, issues = [], []
    # Per-file seq is not cross-process order. Only use monotonic time here.
    for row in sorted(rows, key=lambda r: (r["monotonic_ns"], r.get("seq", 0))):
        key = (row.get("pid"), row.get("tid"), row.get("event"))
        phase = row.get("phase")
        if phase == "begin":
            stacks[key].append(row)
        elif phase in ("end", "error"):
            begin = stacks[key].pop() if stacks[key] else None
            merged = {**(begin or {}), **row}
            # End snapshots omit TE byte counts and the transfer's argument context.
            output.append({"begin": begin, "end": row, "data": merged})
    for (pid, tid, event), pending in stacks.items():
        if pending:
            issues.append({"event": event, "count": len(pending), "issue": "begin_without_end"})
    return output, issues


def compact(span, origin):
    data, begin, end = span["data"], span["begin"], span["end"]
    output = {
        "event": data.get("event"), "phase": end.get("phase"),
        "role": data.get("role"), "tp_rank": str(data.get("tp_rank", "U")),
        "begin_us": (begin["monotonic_ns"] - origin) / 1000 if begin else None,
        "end_us": (end["monotonic_ns"] - origin) / 1000,
        "duration_us": data.get("duration_ns", 0) / 1000,
    }
    for name in ("layer_idx", "layer_name", "num_external_tokens", "num_computed_tokens",
                 "block_ids", "allocated_block_ids", "local_block_ids", "remote_block_ids",
                 "transfer_bytes", "transfer_ranges", "trans_count", "trans_flag",
                 "done_source_count", "pending_pull_tasks", "all_task_done"):
        if name in data:
            output[name] = data[name]
    if "before_status" in data:
        output["before_status"] = data["before_status"]
    for name in REQUEST_FIELDS:
        if isinstance(data.get(name), dict) and data[name]:
            output[name] = data[name]
    if data.get("result"):
        output["result"] = data["result"]
    context = data.get("transfer_context")
    if isinstance(context, dict):
        output["transfer_context"] = context
    if isinstance(data.get("send_requests"), dict):
        output["send_requests"] = data["send_requests"]
    return output


def compare_release_to_write(events, clock_domain):
    frees = [e for e in events if e["event"] == "Scheduler._free_blocks" and e["role"] == "P" and e["phase"] == "end"]
    writes = [e for e in events if e["event"] == "TransferEngine.batch_transfer_sync_write" and e["role"] == "P"]
    successful = [e for e in writes if e["phase"] == "end" and isinstance(e.get("result", {}).get("return_code"), int) and e["result"]["return_code"] >= 0]
    result = {
        "source_free_observed": bool(frees), "write_observed": bool(writes),
        "observed_write_call_count": len(writes),
        "observed_successful_write_call_count": len(successful),
        "verdict": "inconclusive",
        "scope": "observed host-side call boundaries; not proof that all expected writes were captured or that allocator reuse occurred",
    }
    if not frees or not writes:
        result["missing"] = [name for name, present in (("P Scheduler._free_blocks", frees), ("P TE write", writes)) if not present]
        return result
    if clock_domain != "same-host-monotonic":
        result["missing"] = ["confirmed shared host monotonic clock"]
        return result
    first_free = min(e["end_us"] for e in frees)
    first_free_begin = min((e["begin_us"] for e in frees if e["begin_us"] is not None), default=None)
    last_write = max(e["end_us"] for e in writes)
    last_success = max((e["end_us"] for e in successful), default=None)
    result.update(first_source_free_begin_us=first_free_begin, first_source_free_end_us=first_free, last_observed_write_end_us=last_write,
                  last_observed_successful_write_end_us=last_success,
                  source_free_minus_last_observed_write_us=first_free-last_write)
    if first_free < last_write:
        result["verdict"] = "source_free_call_finished_before_last_observed_write_return"
    elif first_free_begin is not None and first_free_begin >= last_write:
        result["verdict"] = "source_free_call_began_after_all_observed_write_returns"
    else:
        result["verdict"] = "overlapping_host_call_intervals_no_order"
    return result


def analyze(root, clock_domain="unknown"):
    rows, report, issues, sources = load(root)
    aliases = Aliases()
    for row in rows:
        learn_aliases(row, aliases)
    all_spans, span_issues = spans(rows)
    issues.extend(span_issues)
    by_request = defaultdict(list)
    unassigned_transfers = []
    transfer_totals = defaultdict(lambda: {"calls": 0, "bytes": 0, "ranges": 0, "return_codes": Counter(), "missing_byte_count_calls": 0, "unassigned_calls": 0})
    for span in all_spans:
        data = span["data"]
        keys = {aliases.root(key) for key in primary_ids(data)}
        for key in keys:
            by_request[key].append(span)
        if str(data.get("event", "")).startswith("TransferEngine.batch_transfer_sync_"):
            key = (data.get("role", "U"), str(data.get("tp_rank", "U")), data["event"].rsplit("_", 1)[-1])
            total = transfer_totals[key]
            total["calls"] += 1
            if isinstance(data.get("transfer_bytes"), int):
                total["bytes"] += data["transfer_bytes"]
                total["ranges"] += data.get("transfer_ranges", 0)
            else:
                total["missing_byte_count_calls"] += 1
            code = data.get("result", {}).get("return_code")
            total["return_codes"][str(code) if isinstance(code, int) else "missing"] += 1
            if not keys:
                total["unassigned_calls"] += 1
                unassigned_transfers.append({"role": key[0], "tp_rank": key[1], "direction": key[2]})
    groups = defaultdict(list)
    for key in aliases.parent:
        groups[aliases.root(key)].append(key)
    requests = []
    for key, items in sorted(by_request.items(), key=lambda pair: min(s["end"]["monotonic_ns"] for s in pair[1])):
        origin = min((s["begin"] or s["end"])["monotonic_ns"] for s in items)
        events = sorted((compact(s, origin) for s in items), key=lambda e: e["end_us"])
        roles = sorted({e["role"] for e in events})
        allocation = [e for e in events if e["event"].endswith(".update_state_after_alloc")]
        delayed = [e for e in events if "delay_free" in e.get("result", {})]
        transfers = [e for e in events if e["event"].startswith("TransferEngine.batch_transfer_sync_")]
        done = [e for e in events if any(part in e["event"] for part in ("send_done", "update_done_task", "get_finished", "get_and_clear_finished", "update_done_task_count"))]
        remote_unblock = [e for e in events if e["role"] == "D" and e["event"] == "Scheduler._update_waiting_for_remote_kv"]
        transitions = [e for e in events if e["role"] == "D" and e["event"] == "Scheduler._handle_blocked_waiting_request"
                       and e.get("before_status") == "WAITING_FOR_REMOTE_KVS"
                       and e.get("request", {}).get("status") in ("WAITING", "PREEMPTED")]
        free = [e for e in events if e["event"] in ("Scheduler._free_blocks", "Scheduler._free_request_blocks")]
        scheduled = [e for e in events if e["event"] == "Scheduler.schedule" and e.get("result", {}).get("num_scheduled_tokens")]
        missing = []
        if "P" not in roles or "D" not in roles:
            missing.append("no explicit alias joining both P and D in this request group")
        if not allocation:
            missing.append("allocation observation")
        if not transfers:
            missing.append("request-attributed TE call")
        if not remote_unblock:
            missing.append("D remote-KV completion handling")
        requests.append({
            "request_key": key, "explicit_aliases": sorted(groups[key]), "roles": roles,
            "ranks": {role: sorted({e["tp_rank"] for e in events if e["role"] == role}) for role in roles},
            "origin": "first associated observed call; unrelated request origins must not be overlaid",
            "allocation": allocation, "delay_free_results": delayed, "transfers": transfers,
            "done_observations": done, "remote_kv_completion_handling": remote_unblock,
            "remote_wait_status_transitions": transitions,
            "scheduler_scheduled_work": scheduled, "free_calls": free,
            "source_free_vs_write": compare_release_to_write(events, clock_domain),
            "missing_observations": missing,
        })
    safe_report = None
    if isinstance(report, dict):
        safe_report = {key: report[key] for key in ("mode", "topology", "profile_enabled", "status") if key in report}
        safe_report["requests"] = []
        for request in report.get("requests", []):
            item = {key: request[key] for key in ("prompt_length", "matches_colocated_greedy", "wall_ns") if key in request}
            if isinstance(request.get("usage"), dict):
                item["usage"] = {key: request["usage"][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                                 if isinstance(request["usage"].get(key), int)}
            safe_report["requests"].append(item)
        safe_report["request_pairing"] = "HTTP input labels are omitted; no ordinal join to internal EngineCore IDs is assumed"
    hook_counts = Counter((r.get("method"), r.get("phase")) for r in rows if r.get("event") == "observer.hook")
    return {
        "schema": "wiki-pd-lifecycle-summary/1", "clock_domain": clock_domain,
        "clock_scope": "Host monotonic observations only. These values are not calibrated NPU-device timestamps.",
        "evidence_limits": [
            "Hooked entry/exit adds Python and I/O overhead; durations are diagnostic, not a performance benchmark.",
            "rankS denotes Scheduler and rankU denotes unknown. Neither is silently relabeled rank0.",
            "A release call is allocator/request bookkeeping; it does not prove immediate physical HBM reuse.",
            "TE sync return >=0 is a host completion observation; absent calls/scopes are not proof of absent transfers.",
            "D _update_waiting_for_remote_kv handles completed receipt; its method boundary alone is not a logged status transition or device decode start.",
            "Requests are joined only by observed local/remote ID or explicit mapping edges; batched calls may belong to several requests.",
        ],
        "rows": len(rows), "sources": sources, "input_report": safe_report,
        "hooks": [{"method": method, "phase": phase, "process_count": count} for (method, phase), count in sorted(hook_counts.items())],
        "issues": issues,
        "transfer_totals": [
            {"role": role, "tp_rank": rank, "direction": direction, **{k: dict(v) if isinstance(v, Counter) else v for k, v in value.items()}}
            for (role, rank, direction), value in sorted(transfer_totals.items())
        ],
        "unassigned_transfer_calls": unassigned_transfers,
        "alias_edges": list({json.dumps(edge, sort_keys=True): edge for edge in aliases.evidence}.values()),
        "requests": requests,
        "trace_processor_queries": {
            "custom_scopes": "SELECT name, COUNT(*) AS calls, ROUND(SUM(dur)/1e6,3) AS total_ms FROM slice WHERE name GLOB 'wiki/*' GROUP BY name ORDER BY name;",
            "scope_tracks": "SELECT s.name, s.ts, s.dur, s.track_id, t.name AS track_name FROM slice s LEFT JOIN track t ON t.id=s.track_id WHERE s.name GLOB 'wiki/*' ORDER BY s.ts;",
            "transfer_scopes": "SELECT name, ts, dur, track_id FROM slice WHERE name GLOB 'wiki/*TransferEngine.batch_transfer_sync*' ORDER BY ts;",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clock-domain", choices=("same-host-monotonic", "unknown"), default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.root, args.clock_domain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": summary["rows"], "request_groups": len(summary["requests"]),
                      "issues": len(summary["issues"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
