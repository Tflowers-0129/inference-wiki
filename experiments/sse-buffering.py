#!/usr/bin/env python3
"""Measure an actual loopback SSE path and a buffering proxy; no model or NPU.

The seven labels are an explicit output-count oracle, not tokenizer results.
Timestamps are observed with one host's monotonic_ns clock. A send interval
brackets write+flush and does not mean that the peer has acknowledged it.
"""

import argparse
from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import socket
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit


def stamp():
    return time.monotonic_ns()


class Records:
    def __init__(self):
        self.lock = threading.Lock()
        self.runs = {}

    def create(self, run_id, count, group_sizes):
        with self.lock:
            self.runs[run_id] = {
                "labels": [f"T{i}" for i in range(count)],
                "group_sizes": group_sizes,
                "upstream": {"generated": [], "sent": []},
                "proxy": {"received": [], "sent": []},
                "upstream_done": threading.Event(),
                "proxy_done": threading.Event(),
            }

    def get(self, run_id):
        with self.lock:
            return self.runs[run_id]


def encoded_event(kind, labels=None):
    if kind == "done":
        return "[DONE]"
    if kind == "role":
        payload = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
    elif kind == "usage":
        payload = {"choices": [], "usage": {"completion_tokens": len(labels)}}
    else:
        payload = {
            "choices": [{"delta": {"content": "".join(labels)}}],
            "mock_token_labels": labels,
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode_event(data):
    if data == "[DONE]":
        return "done", [], None
    payload = json.loads(data)
    labels = payload.get("mock_token_labels", [])
    if labels:
        text = payload["choices"][0]["delta"]["content"]
        if text != "".join(labels):
            raise AssertionError("Content does not match known output labels")
        return "content", labels, payload
    return ("usage" if "usage" in payload else "role"), [], payload


def read_sse(response):
    """Yield complete SSE data events, not TCP read boundaries."""
    data_lines = []
    while True:
        line = response.readline()
        if not line:
            if data_lines:
                raise RuntimeError("EOF inside an SSE event")
            return
        line = line.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines), stamp()
                data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def open_stream(self):
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True

    def emit(self, record, kind, labels=None):
        data = encoded_event(kind, labels)
        begin = stamp()
        self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
        self.wfile.flush()
        record["sent"].append({
            "kind": kind, "labels": labels if kind == "content" else [],
            "send_begin_ns": begin, "send_end_ns": stamp(),
        })


class Upstream(Handler):
    def do_GET(self):
        run_id = parse_qs(urlsplit(self.path).query)["run_id"][0]
        run = self.server.records.get(run_id)
        record = run["upstream"]
        record["request_received_ns"] = stamp()
        pending = queue.Queue()

        def produce():
            start = stamp()
            record["producer_start_ns"] = start
            for index, label in enumerate(run["labels"]):
                deadline = start + (index + 1) * self.server.period_ns
                delay = (deadline - stamp()) / 1_000_000_000
                if delay > 0:
                    time.sleep(delay)
                record["generated"].append({"label": label, "generated_ns": stamp()})
                pending.put(label)
            pending.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        try:
            self.open_stream()
            self.emit(record, "role")
            producer.start()
            while True:
                label = pending.get(timeout=10)
                if label is None:
                    break
                self.emit(record, "content", [label])
            producer.join(timeout=2)
            self.emit(record, "usage", run["labels"])
            self.emit(record, "done")
        except Exception as exc:
            record["error"] = repr(exc)
        finally:
            record["handler_end_ns"] = stamp()
            run["upstream_done"].set()


class Proxy(Handler):
    def do_GET(self):
        run_id = parse_qs(urlsplit(self.path).query)["run_id"][0]
        run = self.server.records.get(run_id)
        record = run["proxy"]
        record["request_received_ns"] = stamp()
        connection = http.client.HTTPConnection("127.0.0.1", self.server.upstream_port, timeout=10)
        pending, group_index = [], 0
        try:
            self.open_stream()
            record["upstream_request_start_ns"] = stamp()
            connection.request("GET", "/stream?" + urlencode({"run_id": run_id}))
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"Upstream status {response.status}")
            for data, received_ns in read_sse(response):
                kind, labels, _payload = decode_event(data)
                record["received"].append({"kind": kind, "labels": labels, "receive_ns": received_ns})
                if kind == "content":
                    pending.extend(labels)
                    wanted = run["group_sizes"][group_index]
                    if len(pending) == wanted:
                        self.emit(record, "content", pending[:])
                        pending.clear()
                        group_index += 1
                    elif len(pending) > wanted:
                        raise AssertionError("Proxy group overfilled")
                else:
                    if kind in {"usage", "done"} and pending:
                        raise AssertionError("Control event arrived with an incomplete group")
                    self.emit(record, kind, run["labels"] if kind == "usage" else None)
                if kind == "done":
                    break
            if group_index != len(run["group_sizes"]):
                raise AssertionError("Proxy did not emit every group")
        except Exception as exc:
            record["error"] = repr(exc)
        finally:
            connection.close()
            record["handler_end_ns"] = stamp()
            run["proxy_done"].set()


def client(port, run_id):
    record = {"request_start_ns": stamp(), "content_events": [], "control_events": []}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", "/stream?" + urlencode({"run_id": run_id}))
        response = connection.getresponse()
        record["headers_received_ns"] = stamp()
        if response.status != 200:
            raise RuntimeError(f"Client status {response.status}")
        for data, received_ns in read_sse(response):
            kind, labels, _payload = decode_event(data)
            event = {"kind": kind, "labels": labels, "receive_ns": received_ns}
            record["content_events" if kind == "content" else "control_events"].append(event)
            if kind == "done":
                break
        record["receive_end_ns"] = stamp()
        return record
    finally:
        connection.close()


def intervals_ms(values):
    return [(b - a) / 1_000_000 for a, b in zip(values, values[1:])]


def measure_case(records, upstream, proxy, repetition, name, count, groups):
    run_id = f"rep{repetition}-{name}"
    records.create(run_id, count, groups)
    run = records.get(run_id)
    use_proxy = name != "direct" and name != "one_output"
    received = client(proxy.server_port if use_proxy else upstream.server_port, run_id)
    if not run["upstream_done"].wait(2):
        raise RuntimeError("Upstream completion was not recorded")
    if use_proxy and not run["proxy_done"].wait(2):
        raise RuntimeError("Proxy completion was not recorded")
    for owner in ("upstream", "proxy"):
        if "error" in run[owner]:
            raise RuntimeError(f"{owner}: {run[owner]['error']}")
    events = received["content_events"]
    labels = [label for event in events for label in event["labels"]]
    generated = [event["label"] for event in run["upstream"]["generated"]]
    actual_groups = [len(event["labels"]) for event in events]
    controls = [event["kind"] for event in received["control_events"]]
    assert generated == labels == run["labels"], "Output labels changed or were dropped"
    assert actual_groups == groups, (actual_groups, groups)
    assert controls == ["role", "usage", "done"], controls
    arrival = [event["receive_ns"] for event in events]
    return {
        "run_id": run_id, "scenario": name, "repetition": repetition,
        "path": "client -> proxy -> upstream" if use_proxy else "client -> upstream",
        "output_count_oracle": count, "labels": labels,
        "observed_event_sizes": actual_groups, "correctness": "passed",
        "upstream": run["upstream"], "proxy": run["proxy"] if use_proxy else None,
        "client": received,
        "metrics": {
            "generation_intervals_ms": intervals_ms([x["generated_ns"] for x in run["upstream"]["generated"]]),
            "client_content_event_itl_ms": intervals_ms(arrival),
            "client_ttft_ms": (arrival[0] - received["request_start_ns"]) / 1_000_000,
            "client_e2el_to_last_content_ms": (arrival[-1] - received["request_start_ns"]) / 1_000_000,
            "client_elapsed_including_control_ms": (received["receive_end_ns"] - received["request_start_ns"]) / 1_000_000,
            "client_tpot_to_last_content_ms_per_output": (
                (arrival[-1] - arrival[0]) / (count - 1) / 1_000_000 if count > 1 else None
            ),
            "tpot_definition": "(last content receive - first content receive)/(known output count - 1)",
            "tpot_undefined_reason": "O <= 1" if count <= 1 else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--period-ms", type=float, default=30)
    args = parser.parse_args()
    if args.repetitions < 2 or not 0 < args.period_ms <= 1000:
        parser.error("Use at least two repetitions and a positive period <= 1000 ms")
    records = Records()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    for server in (upstream, proxy):
        server.daemon_threads = True
        server.records = records
    upstream.period_ns = round(args.period_ms * 1_000_000)
    proxy.upstream_port = upstream.server_port
    threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
               for s in (upstream, proxy)]
    for thread in threads:
        thread.start()
    try:
        cases = []
        for rep in range(args.repetitions):
            # A/B then B/A ordering separates the two main paths' execution order.
            scenario_order = ["direct", "buffer_1_3_3"] if rep % 2 == 0 else ["buffer_1_3_3", "direct"]
            for name in scenario_order + ["single_block", "one_output"]:
                count, groups = {
                    "direct": (7, [1] * 7), "buffer_1_3_3": (7, [1, 3, 3]),
                    "single_block": (7, [7]), "one_output": (1, [1]),
                }[name]
                cases.append(measure_case(records, upstream, proxy, rep, name, count, groups))
        result = {
            "scope": "Actual loopback HTTP/SSE protocol and metric validation; no model or NPU performance claim",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "clock_domain": "One Linux/macOS host process, time.monotonic_ns; all observed times share that clock",
            "nominal_generation_period_ms": args.period_ms,
            "time_caveat": "Actual generation and arrival intervals are measured; no equal-spacing assertion",
            "send_caveat": "send_begin/end bracket write+flush, not peer acknowledgement",
            "count_caveat": "Known mock labels supply output counts; no chunk-by-chunk tokenization",
            "excluded_from_content_events": ["role", "usage", "[DONE]"],
            "correctness": "passed", "repetitions": args.repetitions, "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps({"correctness": "passed", "cases": len(cases), "output": str(args.output),
                          "scope": result["scope"]}, ensure_ascii=False))
    finally:
        for server in (proxy, upstream):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
