"""Compare the installed Triton kernel with its Torch expression on Ascend."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.ops.triton.spec_decode.utils import prepare_inputs_padded_kernel
from vllm_ascend.worker.spec_decode_utils import PREPARE_INPUTS_BLOCK_SIZE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
ROOT = parser.parse_args().output
ROOT.mkdir(parents=True, exist_ok=True)
torch.npu.set_device(0)

def reference(cu, valid, query):
    drafts = torch.cat([cu[:1], cu[1:] - cu[:-1]])
    rejected = torch.where(drafts > 0, drafts + 1 - valid, torch.zeros_like(drafts))
    return (query[1:] - 1 - rejected).to(torch.int32), rejected

def fused(cu, valid, query):
    n = valid.numel()
    out = torch.empty(n, device='npu', dtype=torch.int32)
    rejected = torch.empty_like(out)
    prepare_inputs_padded_kernel[(1,)](cu, valid, query, out, rejected, n,
                                      BLOCK_SIZE=PREPARE_INPUTS_BLOCK_SIZE)
    return out, rejected

cases = [([3], [a + 1]) for a in range(4)]
cases += [([3, 1, 0], [3, 1, 1]), ([0, 0, 0], [1, 1, 1])]
gen = torch.Generator().manual_seed(20261104)
k = torch.randint(0, 6, (127,), generator=gen).tolist()
v = [int(torch.randint(1, ki + 2, (1,), generator=gen)) for ki in k]
cases.append((k, v))
results = []
for repetition in range(2):
    for ks, vs in cases:
        cu = torch.tensor(ks, dtype=torch.int32).cumsum(0).to(torch.int32).npu()
        valid = torch.tensor(vs, dtype=torch.int32).npu()
        query = torch.tensor([0] + [sum(ks[:i + 1]) + i + 1 for i in range(len(ks))], dtype=torch.int32).npu()
        expected_rejected = [ki + 1 - vi if ki > 0 else 0 for ki, vi in zip(ks, vs)]
        expected_indices = [sum(ks[:i + 1]) + i - expected_rejected[i] for i in range(len(ks))]
        for fn in [reference, fused]:
            got = fn(cu, valid, query)
            assert got[0].cpu().tolist() == expected_indices
            assert got[1].cpu().tolist() == expected_rejected
        results.append({'repetition': repetition, 'num_requests': len(ks), 'K': ks if len(ks) < 10 else 'seeded 0..5',
                        'valid_count': vs if len(ks) < 10 else 'seeded 1..K+1',
                        'indices': expected_indices if len(ks) < 10 else 'oracle matched',
                        'rejected': expected_rejected if len(ks) < 10 else 'oracle matched'})

# Capture a fixed 127-request batch after JIT and operator warmup.
for _ in range(5): reference(cu, valid, query); fused(cu, valid, query)
torch.npu.synchronize()
activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
with torch_npu.profiler.profile(activities=activities,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(ROOT / 'profile'), worker_name='wiki_padded'),
        record_shapes=True, with_stack=False) as prof:
    for rep, order in enumerate([('torch', 'triton'), ('triton', 'torch')]):
        for mode in order:
            with torch.profiler.record_function(f'padded_{mode}_rep{rep}'):
                result = (reference if mode == 'torch' else fused)(cu, valid, query)
                torch.npu.synchronize()
            prof.step()
summary = {'correctness': 'passed', 'repetitions': 2, 'cases_per_repetition': len(cases),
           'profile_input': {'requests': len(k), 'dtype': 'int32', 'graph': 'eager', 'order': ['Torch', 'Triton', 'Triton', 'Torch']},
           'scope': 'kernel/formula equivalence, not full Runner performance', 'results': results}
(ROOT / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
print('TRACE_FILES', [str(p) for p in ROOT.rglob('trace_view.json')], flush=True)
