"""Run real single-host P/D engines through the installed Ascend proxies.

Use only devices covered by a current exclusive lease. Raw server logs, traces,
and HTTP responses belong in a private experiment root, outside the Wiki.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def http(base, path, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:3000]}") from exc


def prepare_model(root, kv_heads=2):
    checkpoint = root / ("tiny-qwen2-gqa" if kv_heads == 2 else f"tiny-qwen2-kv{kv_heads}")
    if (checkpoint / "tokenizer.json").exists():
        return checkpoint
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(20261105)
    config = Qwen2Config(vocab_size=256, hidden_size=512,
                        intermediate_size=1024, num_hidden_layers=2,
                        num_attention_heads=8, num_key_value_heads=kv_heads,
                        max_position_embeddings=4096, bos_token_id=1,
                        eos_token_id=2, tie_word_embeddings=False,
                        use_sliding_window=False)
    model = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
    model.save_pretrained(checkpoint, safe_serialization=True)
    vocab = {f"T{i:03}": i for i in range(256)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="T000"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                  unk_token="T000", bos_token="T001",
                                  eos_token="T002", pad_token="T000")
    fast.save_pretrained(checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["pull", "push"], required=True)
    parser.add_argument("--devices", default="1,2,3,6")
    parser.add_argument("--p-tp", type=int, default=2)
    parser.add_argument("--d-tp", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dcp", type=int, default=1)
    parser.add_argument("--device-selection", choices=["split-visible", "shared-visible"],
                        default="split-visible")
    parser.add_argument("--ascend-source", type=Path,
                        default=Path("/vllm-workspace/vllm-ascend"))
    parser.add_argument("--base-port", type=int, default=18210)
    parser.add_argument("--kv-base-port", type=int, default=26200)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    leased = {int(x) for x in os.environ.get("MINDIE_LEASED_DEVICES", "").split(",") if x}
    assert len(devices) == args.p_tp + args.d_tp and len(set(devices)) == len(devices)
    assert set(devices) <= leased, "Acquire and pass an exclusive device lease first"
    args.root.mkdir(parents=True, exist_ok=True)
    checkpoint = prepare_model(args.root, args.kv_heads)
    output = args.root / args.mode
    output.mkdir(exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    processes = []
    handles = []
    endpoints = {"P": f"http://127.0.0.1:{args.base_port}",
                 "D": f"http://127.0.0.1:{args.base_port+10}",
                 "proxy": f"http://127.0.0.1:{args.base_port+20}"}
    report = {"mode": args.mode, "topology": {"P_tp": args.p_tp, "D_tp": args.d_tp},
              "kv_heads": args.kv_heads, "dcp": args.dcp,
              "device_selection": args.device_selection,
              "profile_enabled": args.profile, "requests": [], "status": "starting"}

    def save():
        tmp = output / "result.tmp"
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(output / "result.json")

    def start(label, command, env):
        handle = (output / f"{label}.log").open("wb")
        handles.append(handle)
        proc = subprocess.Popen(command, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((label, proc))
        print("START", label, proc.pid, flush=True)
        return proc

    def ready(label, path):
        for _ in range(240):
            for name, proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited {proc.returncode}; inspect private log")
            try:
                http(endpoints[label], path, timeout=2)
                print("READY", label, flush=True)
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"{label} startup timeout")

    env_base = os.environ.copy()
    env_base.update({"VLLM_USE_V2_MODEL_RUNNER": "1",
                     "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                     "HCCL_CONNECT_TIMEOUT": "120", "HCCL_EXEC_TIMEOUT": "120",
                     "VLLM_HOST_IP": "127.0.0.1", "HCCL_IF_IP": "127.0.0.1",
                     "GLOO_SOCKET_IFNAME": "lo", "TP_SOCKET_IFNAME": "lo",
                     "ASCEND_LOCAL_COMM_RES": '{"version":"1.3"}',
                     "VLLM_LOGGING_LEVEL": "INFO"})
    connector = "MooncakeConnectorV1" if args.mode == "pull" else "MooncakeLayerwiseConnector"
    try:
        for index, role in enumerate(["P", "D"]):
            env = env_base.copy()
            role_devices = devices[:args.p_tp] if role == "P" else devices[args.p_tp:]
            role_tp = args.p_tp if role == "P" else args.d_tp
            visible_devices = devices if args.device_selection == "shared-visible" else role_devices
            env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(x) for x in visible_devices)
            transfer = {"kv_connector": connector,
                        "kv_role": "kv_producer" if role == "P" else "kv_consumer",
                        "engine_id": f"wiki-{args.mode}-{role}",
                        "kv_port": args.kv_base_port + index * 100,
                        "kv_connector_extra_config": {
                            "prefill": {"dp_size": 1, "tp_size": args.p_tp},
                            "decode": {"dp_size": 1, "tp_size": args.d_tp}}}
            command = [sys.executable, str(script_dir / "instrument_pd.py"),
                       "--mode", args.mode, "--role", role,
                       "--output-dir", str(output), "--",
                       "serve", str(checkpoint), "--served-model-name", "wiki-tiny-pd",
                       "--host", "127.0.0.1", "--port", str(args.base_port + index*10),
                       "--tensor-parallel-size", str(role_tp), "--distributed-executor-backend", "mp",
                       "--decode-context-parallel-size", str(args.dcp),
                       "--block-size", "128", "--cp-kv-cache-interleave-size", "128",
                       "--dtype", "bfloat16", "--seed", "20261105",
                       "--enforce-eager", "--max-model-len", "2048",
                       "--max-num-batched-tokens", "1024", "--max-num-seqs", "4",
                       "--kv-cache-memory-bytes", str(32*1024*1024),
                       "--no-enable-prefix-caching", "--kv-transfer-config", json.dumps(transfer)]
            if args.device_selection == "shared-visible":
                selected_indices = list(range(args.p_tp)) if role == "P" else list(range(args.p_tp, len(devices)))
                command += ["--device-ids", ",".join(str(x) for x in selected_indices)]
            if args.profile:
                command += ["--profiler-config", json.dumps({"profiler": "torch",
                            "torch_profiler_dir": str(output / "profiles" / role),
                            "torch_profiler_with_stack": False})]
            start(role, command, env)
        ready("P", "/health")
        ready("D", "/health")
        proxy_name = "load_balance_proxy_server_example.py" if args.mode == "pull" else "load_balance_proxy_layerwise_server_example.py"
        proxy_path = args.ascend_source / "examples/disaggregated_prefill_v1" / proxy_name
        proxy_command = [sys.executable, str(proxy_path), "--host", "127.0.0.1",
                         "--port", str(args.base_port+20), "--prefiller-hosts", "127.0.0.1",
                         "--prefiller-ports", str(args.base_port), "--decoder-hosts", "127.0.0.1",
                         "--decoder-ports", str(args.base_port+10)]
        start("proxy", proxy_command, env_base)
        ready("proxy", "/healthcheck")
        def request(length, request_id):
            return {"model": "wiki-tiny-pd", "prompt": [(i*17+length)%253+3 for i in range(length)],
                    "temperature": 0, "max_tokens": 4, "ignore_eos": True,
                    "return_token_ids": True, "logprobs": 5, "stream": False,
                    "request_id": request_id}
        for length in [129, 513]:
            baseline = http(endpoints["P"], "/v1/completions", request(length, f"{args.mode}-baseline-{length}"))
            warm = http(endpoints["proxy"], "/v1/completions", request(length, f"{args.mode}-warm-{length}"))
            (output / f"baseline-{length}.json").write_text(json.dumps(baseline, indent=2))
            (output / f"warm-{length}.json").write_text(json.dumps(warm, indent=2))
            assert baseline["choices"][0]["token_ids"] == warm["choices"][0]["token_ids"], (length, baseline, warm)
        if args.profile:
            for role in ["P", "D"]:
                http(endpoints[role], "/start_profile", {})
        for repetition in range(2):
            for length in [129, 513]:
                rid = f"{args.mode}-rep{repetition}-n{length}"
                begin = time.monotonic_ns()
                result = http(endpoints["proxy"], "/v1/completions", request(length, rid))
                elapsed = time.monotonic_ns() - begin
                baseline = json.loads((output / f"baseline-{length}.json").read_text())
                tokens = result["choices"][0]["token_ids"]
                match = tokens == baseline["choices"][0]["token_ids"]
                reference_lp = baseline["choices"][0]["logprobs"]["token_logprobs"]
                actual_lp = result["choices"][0]["logprobs"]["token_logprobs"]
                assert len(reference_lp) == len(actual_lp) == 4
                logprob_max_abs = max(abs(a-b) for a,b in zip(reference_lp, actual_lp))
                report["requests"].append({"id": rid, "prompt_length": length,
                    "token_ids": tokens, "matches_colocated_greedy": match,
                    "selected_logprob_max_abs": logprob_max_abs,
                    "wall_ns": elapsed, "usage": result.get("usage")})
                (output / f"{rid}.response.json").write_text(json.dumps(result, indent=2))
                save()
                assert match, rid
                assert logprob_max_abs < 0.05, (rid, logprob_max_abs)
        time.sleep(2)
        if args.profile:
            for role in ["P", "D"]:
                http(endpoints[role], "/stop_profile", {})
        report["status"] = "passed"
        report["scope"] = "random tiny Qwen2 checkpoint; request handoff correctness, no pretrained quality or throughput claim"
        save()
        print("WIKI_PD_RESULT", json.dumps(report), flush=True)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)[:3000]
        save()
        raise
    finally:
        for label, proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for label, proc in reversed(processes):
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            print("STOP", label, proc.returncode, flush=True)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
