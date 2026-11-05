"""Run the installed engine on a small, deterministic random Qwen2 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--pcp', type=int, default=1)
    p.add_argument('--dcp', type=int, default=1)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    root = args.root
    checkpoint = root / 'tiny-qwen2-random'
    label = f'TP{args.tp}_PCP{args.pcp}_DCP{args.dcp}'
    target = root / label
    target.mkdir(parents=True, exist_ok=True)
    if not (checkpoint / 'model.safetensors').exists():
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(20261105)
        config = Qwen2Config(vocab_size=256, hidden_size=512, intermediate_size=1024,
                            num_hidden_layers=2, num_attention_heads=8,
                            num_key_value_heads=1, max_position_embeddings=8192,
                            bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
                            use_sliding_window=False)
        model = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
        model.save_pretrained(checkpoint, safe_serialization=True)
        del model
    from vllm import LLM, SamplingParams
    kwargs = dict(model=str(checkpoint), skip_tokenizer_init=True, dtype='bfloat16',
                  tensor_parallel_size=args.tp, prefill_context_parallel_size=args.pcp,
                  decode_context_parallel_size=args.dcp,
                  distributed_executor_backend='mp', enforce_eager=True,
                  max_model_len=2048, max_num_batched_tokens=512, max_num_seqs=4,
                  kv_cache_memory_bytes=32 * 1024 * 1024, enable_prefix_caching=False,
                  enable_chunked_prefill=True, seed=20261105)
    if args.profile:
        kwargs['profiler_config'] = {'profiler': 'torch',
                                    'torch_profiler_dir': str(target / 'profiles'),
                                    'torch_profiler_with_stack': False}
    llm = LLM(**kwargs)
    prompts = [{'prompt_token_ids': [(i * 17 + n) % 253 + 3 for i in range(n)]}
               for n in [64, 129, 256, 513]]
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True,
                            detokenize=False, logprobs=5)
    def generate():
        return llm.generate(prompts, params, use_tqdm=False)
    warm = generate()
    if args.profile:
        llm.start_profile()
    results = []
    for rep in range(2):
        begin = time.perf_counter()
        outputs = generate()
        seconds = time.perf_counter() - begin
        rows = []
        for output in outputs:
            seq = output.outputs[0]
            chosen_logprobs = [step[token].logprob for step, token in zip(seq.logprobs, seq.token_ids)]
            rows.append({'prompt_length': len(output.prompt_token_ids),
                         'token_ids': list(seq.token_ids), 'chosen_logprobs': chosen_logprobs})
        assert [o.outputs[0].token_ids for o in outputs] == [o.outputs[0].token_ids for o in warm]
        results.append({'repetition': rep, 'wall_seconds': seconds, 'outputs': rows})
    if args.profile:
        llm.stop_profile()
    config = llm.llm_engine.vllm_config
    report = {'scope': 'installed ModelRunnerV2, two-layer random Qwen2; correctness only, not pretrained quality or service throughput',
              'topology': {'tp': args.tp, 'pcp': args.pcp, 'dcp': args.dcp},
              'checkpoint_sha256': hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest(),
              'model_config': json.loads((checkpoint / 'config.json').read_text()),
              'local_kv_heads': config.model_config.get_num_kv_heads(config.parallel_config),
              'cache_config': str(config.cache_config), 'results': results,
              'correctness': 'deterministic within configuration; compare across configurations separately'}
    (target / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('WIKI_TINY_CP_RESULT', json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
