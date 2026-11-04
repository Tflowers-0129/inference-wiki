import argparse
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.worker.spec_decode_utils import update_num_computed_tokens_for_batch_change

torch.npu.set_device(0)
rows=[]
for rep in range(2):
 for next_k in [0,1,3]:
  computed=torch.tensor([8,20,0],device='npu',dtype=torch.int32)
  accepted=torch.tensor([1,1,1],device='npu',dtype=torch.int32)
  previous_positions=torch.tensor([1,-1,0],device='npu',dtype=torch.int64)
  valid=torch.tensor([3,1,0],device='npu',dtype=torch.int32)
  previous_k=torch.tensor([3,0,0],device='npu',dtype=torch.int32)
  cpu_derived_values=torch.tensor([21,5,12],device='npu',dtype=torch.int32)
  update_num_computed_tokens_for_batch_change(computed,accepted,previous_positions,valid,previous_k,cpu_derived_values)
  got,acc=computed.cpu().tolist(),accepted.cpu().tolist()
  assert got==[21,5,11] and acc==[1,1,3],(got,acc)
  rows.append(dict(repetition=rep,K_next=next_k,previous_order=['R','S'],next_order=['S','T','R'],computed=got,accepted=acc,R_next_query_length=1+next_k))
result=dict(correctness='passed',scope='installed helper only; K_next is separate metadata, not this helper input',rows=rows)
parser=argparse.ArgumentParser(description='Verify installed batch-change helper')
parser.add_argument('--output',type=Path,required=True)
p=parser.parse_args().output;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
