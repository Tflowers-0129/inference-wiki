#!/usr/bin/env python3
"""Observe the fixed vLLM-Ascend PD paths without changing their results.

Run ``python instrument_pd.py --mode pull --role P --output-dir OUT --
serve MODEL ...``. Spawned Python workers inherit a generated sitecustomize.
Lifecycle JSONL contains hashed request identifiers, counters and block IDs;
addresses, endpoints, payloads, model inputs and exception messages are omitted.
Profiler scopes use the normal torch record_function API. Background-thread
scope capture depends on the profiler implementation and must be checked.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

_INSTALLED = False
_LOCK = threading.RLock()
_LOCAL = threading.local()
_SEQUENCE = 0
_FD = None
_FD_PID = None
_LIMIT = 1024

_BASE = "vllm_ascend.distributed.kv_transfer.kv_p2p."
_TARGETS = {
    _BASE + "mooncake_connector": {
        "MooncakeConnectorScheduler": (
            "__init__", "on_new_request", "get_num_new_matched_tokens",
            "update_state_after_alloc", "build_connector_meta", "request_finished",
        ),
        "MooncakeConnectorWorker": (
            "__init__", "register_kv_caches", "start_load_kv", "get_finished",
            "get_transfer_results", "get_block_ids_with_load_errors",
        ),
        "MooncakeConnector": ("wait_for_layer_load", "save_kv_layer", "wait_for_save"),
        "KVCacheRecvingThread": (
            "__init__", "add_request", "_submit_request", "_handle_request",
            "_transfer_kv_cache_all_groups", "_mark_request_task_submitted",
            "_mark_request_task_done", "_mark_failed_recv_request",
            "_send_done_recv_signal", "_send_done_signal_to_free_remote_port",
            "_reformat_pending_kv_caches", "get_and_clear_finished_requests",
            "get_and_clear_failed_recv_requests",
        ),
        "KVCacheSendingThread": (
            "__init__", "add_not_transfer_request", "add_delayed_request",
            "get_and_clear_finished_requests",
        ),
        "KVCacheTaskTracker": (
            "add_req_to_process", "add_not_transfer_request", "add_delayed_request",
            "update_done_task_count", "get_and_clear_finished_requests",
        ),
    },
    _BASE + "mooncake.base_worker": {
        "MooncakeBaseConnectorWorker": ("__init__", "register_kv_caches"),
    },
    _BASE + "mooncake.pull_worker": {
        "MooncakePullConnectorWorker": (
            "register_kv_caches", "start_load_kv", "get_transfer_results",
            "get_block_ids_with_load_errors",
        ),
        "MooncakePullRecvingThread": (
            "__init__", "add_requests", "_handle_requests",
            "_execute_tp_transfer_bucket", "_mark_request_failed",
            "get_and_clear_finished_requests",
        ),
    },
    _BASE + "mooncake.pull_scheduler": {
        "MooncakePullConnectorScheduler": (
            "__init__", "get_num_new_matched_tokens", "update_state_after_alloc",
            "build_connector_meta", "request_finished", "update_connector_output",
        ),
        "MooncakeSchedulerSendingThread": (
            "add_delayed_request", "_handle_finished_request",
            "_mark_finished_locked", "get_and_clear_finished_requests",
            "_retrieve_expired_requests_locked",
        ),
        "MooncakeSchedulerRecvingThread": ("add_request", "_send_done_recving"),
    },
    _BASE + "mooncake_layerwise_connector": {
        "MooncakeLayerwiseConnectorScheduler": (
            "__init__", "get_num_new_matched_tokens", "update_state_after_alloc",
            "build_connector_meta", "_access_metaserver", "request_finished",
            "request_finished_all_groups",
        ),
        "MooncakeLayerwiseConnectorWorker": (
            "__init__", "register_kv_caches", "prepare_layerwise_step",
            "start_load_kv", "on_kv_cache_written", "save_kv_layer",
            "wait_for_layer_load", "wait_for_layer_reuse",
            "_mark_layer_reuse_pending", "_complete_layer_reuse",
            "send_done_send_signal", "get_finished", "get_block_ids_with_load_errors",
        ),
        "MooncakeLayerwiseConnector": ("wait_for_save",),
        "KVCacheSendingLayerThread": (
            "__init__", "_handle_request", "_transfer_kv_cache",
        ),
        "KVCacheRecvingLayerThread": (
            "__init__", "update_done_task", "update_failed_task",
            "get_and_clear_done_requests", "get_and_clear_failed_requests",
        ),
    },
    "vllm.v1.core.sched.scheduler": {
        "Scheduler": (
            "__init__", "add_request", "schedule", "finish_requests",
            "_connector_finished", "_free_request", "_free_blocks",
            "_free_request_blocks", "_update_waiting_for_remote_kv",
            "_handle_blocked_waiting_request",
            "_update_from_kv_xfer_finished", "_handle_failed_recving",
        ),
    },
    "mooncake.engine": {
        "TransferEngine": (
            "batch_transfer_sync_read", "batch_transfer_sync_write",
        ),
    },
}
_FUNCTION_TARGETS = {
    _BASE + "mooncake_layerwise_connector": ("get_external_request_id",),
}
_POLL_METHODS = {
    "get_finished", "get_transfer_results", "get_and_clear_finished_requests",
    "get_and_clear_done_requests", "get_and_clear_failed_requests",
    "get_block_ids_with_load_errors", "_retrieve_expired_requests_locked",
    "_handle_blocked_waiting_request",
}


def _request_key(value):
    """Stable across this job's processes; no original request ID is retained."""
    salt = os.environ.get("WIKI_PD_ID_SALT", "wiki-pd-local")
    return "q" + hashlib.sha256((salt + "\0" + str(value)).encode()).hexdigest()[:16]


def _ids(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [_request_key(value) for value in list(values)[:_LIMIT]]


def _ints(values):
    """Accept only Python host integers; never copy or synchronize a tensor."""
    if isinstance(values, bool):
        return values
    if isinstance(values, int):
        return values
    if isinstance(values, (tuple, list)):
        return [_ints(v) for v in values[:_LIMIT] if isinstance(v, (int, tuple, list))]
    return None


def _get(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _request(value):
    result = {}
    def field(key):
        return value.get(key) if isinstance(value, dict) else _get(value, key)
    for key in ("request_id", "req_id", "remote_request_id"):
        item = field(key)
        if isinstance(item, str):
            result[key] = _request_key(item)
    for key in (
        "num_prompt_tokens", "num_computed_tokens", "num_external_tokens",
        "local_num_prompt_tokens", "remote_num_prompt_tokens", "prompt_len",
        "local_computed_tokens", "local_transed_tokens", "remote_cache_tokens",
        "chunk_finish", "do_virtual", "all_task_done", "shard_idx", "awaiting_kvs",
    ):
        item = field(key)
        if isinstance(item, (int, bool)):
            result[key] = item
    status = field("status")
    if status is not None:
        # Enum names carry state, not arbitrary Request string contents.
        result["status"] = _get(status, "name", type(status).__name__)
    for key in ("local_block_ids", "remote_block_ids", "local_full_block_ids"):
        item = field(key)
        if isinstance(item, (tuple, list)):
            result[key] = _ints(item)
    params = field("kv_transfer_params")
    if isinstance(params, dict):
        result["transfer_flags"] = {
            key: params[key] for key in (
                "do_remote_prefill", "do_remote_decode", "do_virtual",
                "remote_num_prompt_tokens", "remote_tp_size", "remote_pcp_size",
                "remote_dcp_size",
            ) if isinstance(params.get(key), (int, bool))
        }
        if isinstance(params.get("remote_request_id"), str):
            result["remote_request_id"] = _request_key(params["remote_request_id"])
    return result


def _metadata(value):
    result = {}
    requests = _get(value, "requests")
    if isinstance(requests, dict):
        result["requests"] = {_request_key(k): _request(v) for k, v in list(requests.items())[:_LIMIT]}
    requests_to_send = _get(value, "requests_to_send")
    if isinstance(requests_to_send, dict):
        # Values are delay timestamps, intentionally omitted.
        result["requests_to_send"] = _ids(requests_to_send)
    for name in ("finished_sending", "finished_recving", "failed_recving", "reqs_in_batch"):
        item = _get(value, name)
        if isinstance(item, (set, list, tuple)):
            result[name] = _ids(item)
    return result


def _identity(obj=None):
    role = os.environ.get("WIKI_PD_ROLE", "U")
    config = _get(obj, "vllm_config")
    kv = _get(obj, "kv_transfer_config") or _get(config, "kv_transfer_config")
    if kv is not None:
        if _get(kv, "is_kv_producer"):
            role = "P"
        elif _get(kv, "is_kv_consumer"):
            role = "D"
    previous = getattr(_LOCAL, "identity", None)
    rank = _get(obj, "tp_rank")
    if isinstance(rank, int):
        rank_name = str(rank)
    elif obj is not None and "Scheduler" in type(obj).__name__:
        rank_name = "S"
    elif previous:
        rank_name = previous[1]
    else:
        dist = sys.modules.get("torch.distributed")
        try:
            rank_name = str(dist.get_rank()) if dist and dist.is_initialized() else "U"
        except Exception:
            rank_name = "U"
    return role, rank_name


def _emit(event, phase, fields=None, identity=None):
    global _FD, _FD_PID, _SEQUENCE
    try:
        role, rank = identity or _identity()
        with _LOCK:
            pid = os.getpid()
            if _FD is None or _FD_PID != pid:
                if _FD is not None:
                    os.close(_FD)
                directory = Path(os.environ["WIKI_PD_LOG_DIR"])
                directory.mkdir(parents=True, exist_ok=True)
                _FD = os.open(str(directory / f"lifecycle-{role}-pid{pid}.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                _FD_PID = pid
            _SEQUENCE += 1
            item = {
                "schema": "wiki-pd-lifecycle/1", "seq": _SEQUENCE,
                "time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                "pid": pid, "tid": threading.get_native_id(),
                "mode": os.environ.get("WIKI_PD_MODE", "unknown"),
                "role": role, "tp_rank": rank, "event": event, "phase": phase,
            }
            if fields:
                item.update(fields)
            os.write(_FD, (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    except Exception:
        # Observer failures never replace production results or exceptions.
        pass


def _snapshot(obj, bound):
    data = {}
    for name in ("request", "req", "request_metadata", "req_meta", "request_data"):
        if name in bound:
            data[name] = _request(bound[name])
    for name in ("request_id", "req_id", "remote_request_id"):
        if isinstance(bound.get(name), str):
            data[name] = _request_key(bound[name])
    for name in ("request_ids", "finished_req_ids", "failed_req_ids"):
        if isinstance(bound.get(name), (set, tuple, list)):
            data[name] = _ids(bound[name])
    for name in (
        "num_computed_tokens", "num_external_tokens", "layer_idx", "trans_count",
        "group_idx", "trans_flag", "delay_free_blocks", "remote_pp_rank",
        "remote_pcp_rank", "remote_tp_rank", "remote_tp_size", "remote_dcp_size",
        "all_task_done", "awaiting_kvs", "report_completion", "shard_idx",
    ):
        value = bound.get(name)
        if isinstance(value, (int, bool)):
            data[name] = value
    layer_name = bound.get("layer_name")
    if isinstance(layer_name, str) and len(layer_name) < 200:
        # Model layer names, unlike request/endpoint strings, are static names.
        data["layer_name"] = layer_name
    for name in ("block_ids", "local_block_ids", "remote_block_ids"):
        if isinstance(bound.get(name), (list, tuple)):
            data[name] = _ints(bound[name])
    blocks = bound.get("blocks")
    block_ids = _get(blocks, "blocks")
    if isinstance(block_ids, (tuple, list)):
        data["allocated_block_ids"] = [
            [_get(block, "block_id") for block in group[:_LIMIT]
             if isinstance(_get(block, "block_id"), int)]
            for group in block_ids[:_LIMIT]
        ]
    for name in ("metadata", "connector_output", "kv_connector_output"):
        if name in bound:
            data[name] = _metadata(bound[name])
    requests = bound.get("requests")
    if isinstance(requests, dict):
        data["requests"] = {_request_key(k): _request(v) for k, v in list(requests.items())[:_LIMIT]}
    buckets = bound.get("transfer_entries_by_spec")
    if isinstance(buckets, dict):
        entries = []
        for spec_index, by_layer in buckets.items():
            for layer_pair, records in by_layer.items():
                for request_id, local_blocks, remote_blocks in records:
                    entries.append({
                        "request_id": _request_key(request_id),
                        "spec_index": spec_index, "layer_pair": _ints(layer_pair),
                        "local_block_ids": _ints(local_blocks),
                        "remote_block_ids": _ints(remote_blocks),
                    })
        data["transfer_entries"] = entries[:_LIMIT]
    task = bound.get("send_task")
    if task is not None:
        data["layer_idx"] = _get(task, "layer_idx")
        data["layer_name"] = _get(task, "layer_name")
        data["send_requests"] = {
            _request_key(k): _request(v)
            for k, v in list((_get(task, "send_request") or {}).items())[:_LIMIT]
        }
    # Queue sizes/counters do not consume queues or clear completion state.
    for name in ("done_requests", "failed_requests", "_pending_reuse_layers"):
        value = _get(obj, name)
        if isinstance(value, set):
            data[name] = _ints(sorted(value)) if name == "_pending_reuse_layers" else _ids(sorted(value))
    tracker = _get(obj, "task_tracker")
    req_id = bound.get("req_id")
    if isinstance(tracker, dict) and isinstance(req_id, str):
        entry = tracker.get(req_id)
        data["done_source_count"] = len(entry) if isinstance(entry, set) else 0
    request_id = bound.get("request_id")
    counts = _get(obj, "request_task_counts")
    if isinstance(counts, dict) and isinstance(request_id, str):
        data["pending_pull_tasks"] = counts.get(request_id, 0)
    request_map = _get(obj, "request_map")
    if isinstance(request_map, dict) and request_map:
        data["request_aliases"] = {
            _request_key(external): _request_key(internal)
            for external, internal in list(request_map.items())[:_LIMIT]
            if isinstance(external, str) and isinstance(internal, str)
        }
    return data


def _result(method, value):
    if method in {"request_finished", "request_finished_all_groups", "_connector_finished"}:
        if isinstance(value, tuple) and value:
            return {"delay_free": bool(value[0]), "has_transfer_params": value[1] is not None}
    if method == "get_num_new_matched_tokens" and isinstance(value, tuple):
        return {"matched_tokens": _ints(value[0]), "load_async": bool(value[1])}
    if method == "get_finished" and isinstance(value, tuple):
        return {"finished_sending": _ids(value[0]), "finished_recving": _ids(value[1])}
    if isinstance(value, set):
        if "block" in method:
            return {"block_ids": _ints(sorted(value))}
        return {"request_ids": _ids(sorted(value))}
    if method in {"build_connector_meta", "get_transfer_results", "get_and_clear_finished_requests"}:
        return _metadata(value)
    if method == "schedule":
        result = {}
        counts = _get(value, "num_scheduled_tokens")
        if isinstance(counts, dict):
            result["num_scheduled_tokens"] = {_request_key(k): v for k, v in counts.items() if isinstance(v, int)}
        result["total_num_scheduled_tokens"] = _get(value, "total_num_scheduled_tokens")
        return result
    if method.startswith("batch_transfer_sync_") and isinstance(value, int):
        return {"return_code": value}
    if method == "_mark_request_task_done" and isinstance(value, bool):
        return {"all_tasks_done": value}
    if method == "_handle_blocked_waiting_request" and isinstance(value, bool):
        return {"proceed": value}
    if method == "_transfer_kv_cache":
        return {"transfer_error": value is not None}
    return {}


def _has_content(value):
    return any(bool(v) for v in value.values())


@contextlib.contextmanager
def _scope(name):
    manager = None
    try:
        torch = sys.modules.get("torch")
        record = _get(_get(torch, "profiler"), "record_function")
        if record:
            manager = record(name)
            manager.__enter__()
    except Exception:
        manager = None
    try:
        yield
    finally:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass


def _wrap(cls, name):
    original = cls.__dict__.get(name)
    if original is None or not callable(original) or getattr(original, "_wiki_pd_observer", False):
        return
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None
    label = cls.__name__ + "." + name

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        old_identity = getattr(_LOCAL, "identity", None)
        old_context = getattr(_LOCAL, "transfer_context", None)
        identity = _identity(self)
        _LOCAL.identity = identity
        try:
            try:
                bound = dict(signature.bind_partial(self, *args, **kwargs).arguments) if signature else {}
                before = _snapshot(self, bound)
                request_keys = set()
                for field in ("requests", "send_requests"):
                    request_keys.update(before.get(field, {}))
                for field in ("request", "req", "request_metadata", "req_meta", "request_data"):
                    request_keys.update(v for k, v in before.get(field, {}).items() if k in ("request_id", "req_id"))
                request_keys.update(row["request_id"] for row in before.get("transfer_entries", []))
                if isinstance(before.get("request_id"), str):
                    request_keys.add(before["request_id"])
                if isinstance(before.get("req_id"), str):
                    request_keys.add(before["req_id"])
                context = {k: before[k] for k in ("layer_idx", "remote_pp_rank", "remote_pcp_rank", "remote_tp_rank") if k in before}
                if request_keys:
                    context["request_ids"] = sorted(request_keys)
                if context:
                    _LOCAL.transfer_context = context
                if name.startswith("batch_transfer_sync_"):
                    if old_context:
                        before["transfer_context"] = old_context
                    lengths = args[3] if len(args) > 3 else kwargs.get("lengths", ())
                    if isinstance(lengths, (list, tuple)) and all(isinstance(x, int) for x in lengths):
                        before.update(transfer_ranges=len(lengths), transfer_bytes=sum(lengths))
                if name not in _POLL_METHODS:
                    _emit(label, "begin", before, identity)
            except Exception:
                bound, before = {}, {}
            scope = f"wiki/{os.environ.get('WIKI_PD_MODE', 'unknown')}/{identity[0]}/rank{identity[1]}/{label}"
            if isinstance(before.get("layer_idx"), int):
                scope += f"/layer{before['layer_idx']}"
            start = time.monotonic_ns()
            try:
                with _scope(scope):
                    value = original(self, *args, **kwargs)
            except BaseException as exc:
                _emit(label, "error", {"exception_type": type(exc).__name__, "duration_ns": time.monotonic_ns() - start}, identity)
                raise
            try:
                # __init__ has populated vllm_config and tp_rank by this point.
                identity = _identity(self)
                after = _snapshot(self, bound)
                if name == "_handle_blocked_waiting_request":
                    after["before_status"] = before.get("request", {}).get("status")
                result = _result(name, value)
                if name not in _POLL_METHODS or _has_content(result):
                    _emit(label, "end", {**after, "result": result, "duration_ns": time.monotonic_ns() - start}, identity)
            except Exception:
                pass
            return value
        finally:
            _LOCAL.identity = old_identity
            _LOCAL.transfer_context = old_context

    observed._wiki_pd_observer = True
    try:
        setattr(cls, name, observed)
        _emit("observer.hook", "installed", {"method": label})
    except (TypeError, AttributeError):
        # Some extension builds expose an immutable TransferEngine type.
        _emit("observer.hook", "unavailable", {"method": label, "reason": "immutable_type"})


def _patch(module):
    for class_name, methods in _TARGETS.get(module.__name__, {}).items():
        cls = getattr(module, class_name, None)
        if cls is not None:
            for method in methods:
                _wrap(cls, method)
    for name in _FUNCTION_TARGETS.get(module.__name__, ()):
        original = getattr(module, name, None)
        if original is None or getattr(original, "_wiki_pd_observer", False):
            continue
        def make_observer(function, label):
            @functools.wraps(function)
            def observed(*args, **kwargs):
                identity = _identity()
                value = args[0] if args else kwargs.get("request_id")
                fields = {"input_request_id": _request_key(value)} if isinstance(value, str) else {}
                start = time.monotonic_ns()
                scope = f"wiki/{os.environ.get('WIKI_PD_MODE', 'unknown')}/{identity[0]}/rank{identity[1]}/{label}"
                with _scope(scope):
                    result = function(*args, **kwargs)
                if isinstance(result, str):
                    fields["output_request_id"] = _request_key(result)
                fields["duration_ns"] = time.monotonic_ns() - start
                _emit(label, "end", fields, identity)
                return result
            observed._wiki_pd_observer = True
            return observed
        setattr(module, name, make_observer(original, name))
        _emit("observer.hook", "installed", {"method": name})


class _ObserverLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        method = getattr(self.loader, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch(module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS and fullname not in _FUNCTION_TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader and hasattr(spec.loader, "exec_module"):
            spec.loader = _ObserverLoader(spec.loader)
        return spec


def install():
    global _INSTALLED
    if _INSTALLED or "WIKI_PD_LOG_DIR" not in os.environ:
        return
    _INSTALLED = True
    sys.meta_path.insert(0, _ObserverFinder())
    for name in set(_TARGETS) | set(_FUNCTION_TARGETS):
        if name in sys.modules:
            _patch(sys.modules[name])
    _emit("observer.process", "installed", {"observer_semantics": "return_and_exception_preserving"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pull", "push"), required=True)
    parser.add_argument("--role", choices=("P", "D"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-salt", default="wiki-pd-local", help="Use the same non-secret value for both engines.")
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cli = args.vllm_args
    if cli[:1] == ["--"]:
        cli = cli[1:]
    if cli[:1] == ["vllm"]:
        cli = cli[1:]
    if not cli:
        parser.error("Pass normal vllm CLI arguments after --, starting with serve.")
    output = args.output_dir.resolve()
    bootstrap = output / "observer-bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    (bootstrap / "sitecustomize.py").write_text(
        "import os, runpy\n"
        "if os.environ.get('WIKI_PD_OBSERVER_FILE'):\n"
        "    runpy.run_path(os.environ['WIKI_PD_OBSERVER_FILE'], run_name='wiki_pd_observer')['install']()\n",
        encoding="utf-8",
    )
    os.environ.update(
        WIKI_PD_MODE=args.mode, WIKI_PD_ROLE=args.role,
        WIKI_PD_LOG_DIR=str(output / "lifecycle"),
        WIKI_PD_OBSERVER_FILE=str(Path(__file__).resolve()),
        WIKI_PD_ID_SALT=args.id_salt,
    )
    os.environ["PYTHONPATH"] = str(bootstrap) + os.pathsep + os.environ.get("PYTHONPATH", "")
    install()
    sys.argv = ["vllm", *cli]
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
