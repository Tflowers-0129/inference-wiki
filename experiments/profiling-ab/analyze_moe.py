#!/usr/bin/env python3
"""Export MoE A/B measurements and complete active trace windows, without PII.

Graph replay lacks Host-to-Device flow sources in this capture. Keep all exported
model tasks and all three Host scopes, but never assign a graph task to a Host
iteration by order or timestamp proximity. This script performs no device work.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from decimal import Decimal as D
import json
import math
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode = True
from analyze_sp import digest, audit_public, device_event, duration_event, point, inside


def stats(values):
    ordered = sorted(values)
    def percentile(p):
        position = (len(ordered)-1)*p
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    return {'count': len(values), 'median_us': statistics.median(values),
            'mean_us': statistics.mean(values), 'min_us': min(values),
            'p10_us': percentile(.1), 'p90_us': percentile(.9), 'max_us': max(values)}


def trace_window(path, rank, mode, job, active_iterations, launch_log):
    raw = json.loads(path.read_text(), parse_float=D)
    events = raw if isinstance(raw, list) else raw['traceEvents']
    names = {(e['pid'], e['tid']): e['args']['name'] for e in events
             if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
    scopes = []
    for iteration in active_iterations:
        name = f'wiki/{job}/mc2/{mode}/rank{rank}/iteration{iteration}'
        matches = [(i,e) for i,e in enumerate(events) if duration_event(e) and e.get('name') == name]
        assert len(matches) == 1
        index, event = matches[0]
        scopes.append((index,event,iteration))
    origin = min(D(e['ts']) for _,e,_ in scopes)
    tasks = [(i,e) for i,e in enumerate(events) if device_event(e)]
    task_points = defaultdict(list)
    cpu_points = defaultdict(list)
    flows = defaultdict(lambda: defaultdict(list))
    for i,e in enumerate(events):
        if device_event(e):
            task_points[point(e)].append((i,e))
        if duration_event(e) and e.get('cat') == 'cpu_op':
            cpu_points[point(e)].append((i,e))
        if e.get('cat') == 'async_npu' and e.get('name') == 'torch_to_npu' and e.get('ph') in ['s','f']:
            flows[str(e['id'])][e['ph']].append((i,e))
    links, assignment, unresolved = [], {}, []
    for endpoints in flows.values():
        if len(endpoints['s']) != 1 or len(endpoints['f']) != 1:
            unresolved.append('nonunique framework flow'); continue
        si, source = endpoints['s'][0]; fi, target = endpoints['f'][0]
        candidates = task_points.get(point(target), [])
        owners = [(sc,iteration) for _,sc,iteration in scopes if inside(source,sc)]
        if len(candidates) != 1 or len(owners) != 1:
            unresolved.append('unmatched exact task or Host scope'); continue
        task_index, task = candidates[0]; _,iteration = owners[0]
        cpu = cpu_points.get(point(source), [])
        host = min(cpu,key=lambda item:D(item[1]['dur'])) if cpu else None
        links.append({'source_flow_event_index': si, 'target_flow_event_index': fi,
            'host_operator_event_index': host[0] if host else None,
            'host_operator': host[1]['name'] if host else None,
            'device_event_index': task_index, 'host_iteration': iteration,
            'source_relative_us': str(D(source['ts'])-origin),
            'destination_relative_us': str(D(target['ts'])-origin),
            'raw_endpoint_delta_us_not_latency': str(D(target['ts'])-D(source['ts']))})
        assignment[task_index] = iteration
    device_rows = []
    for index,event in tasks:
        args = event.get('args',{}); track = names.get((event['pid'],event['tid']))
        assert track and re.fullmatch(r'(Stream \d+|Group group_name_\d+ Communication)',track)
        device_rows.append({'event_index': index, 'name': event['name'], 'track_name': track,
            'physical_stream_id': args.get('Physic Stream Id'), 'task_type': args.get('Task Type','HCCL parent/AICPU'),
            'relative_start_us': str(D(event['ts'])-origin), 'duration_us': str(D(event['dur'])),
            'host_iteration': assignment.get(index),
            'association': 'exact_framework_flow' if index in assignment else 'full_export_context_no_host_iteration_link',
            'task_fields': {key:args[key] for key in ['Model Id','Task Id','Batch Id','Subtask Id','connection_id'] if key in args}})
    warning = 'Failed to get acl to npu flow events'
    logs = []
    matching_lines = [number for number,line in enumerate(launch_log.read_text(errors='replace').splitlines(),1)
                      if warning in line and f'/rank{rank}/profiles/mc2/{mode}/' in line]
    if matching_lines:
        logs.append({'launch_log_sha256':digest(launch_log), 'message':warning,
                     'line_numbers':matching_lines, 'occurrences':len(matching_lines)})
    dangling = [(i,e) for i,e in enumerate(events) if e.get('cat') == 'HostToDevice' and e.get('ph') in ['s','f']]
    ids = defaultdict(lambda:Counter())
    for _,e in dangling:
        ids[str(e['id'])][e['ph']]+=1
    info_path, = path.parent.parent.glob('profiler_info_*.json')
    info = json.loads(info_path.read_text())
    assert int(info['rank_id']) == rank
    level = info['config']['experimental_config']['_profiler_level']
    assert level == 'Level1'
    assert not unresolved
    if mode == 'eager':
        assert len(tasks) == 15 and len(assignment) == 15 and len(links) == 15
    else:
        assert len(tasks) == 24 and not links and logs
        assert sum(e.get('ph') == 's' for _,e in dangling) == 0
        assert all(row['host_iteration'] is None for row in device_rows)
    return {'mode':mode,'rank':rank,'trace_sha256':digest(path),'trace_bytes':path.stat().st_size,
        'profiler_info_sha256':digest(info_path), 'profiler_level':level,
        'cann_version_from_profiler':info['cann_version'], 'torch_npu_version_from_profiler':info['torch_npu_version'],
        'selection':'entire exported active capture; all device duration tasks retained, no single-iteration crop',
        'time_origin':'first active Host scope start in this rank/configuration export; independent origin',
        'host_scope_semantics':'executor.run() only; local completion synchronization occurs after the scope, before profiler.step()',
        'host_scopes':[{'event_index':i,'iteration':iteration,'relative_start_us':str(D(e['ts'])-origin),
                        'duration_us':str(D(e['dur']))} for i,e,iteration in scopes],
        'device_events':device_rows,'tracks':sorted({e['track_name'] for e in device_rows}),
        'device_first_relative_us':str(min(D(e['ts']) for _,e in tasks)-origin),
        'device_last_relative_us':str(max(D(e['ts'])+D(e['dur']) for _,e in tasks)-origin),
        'task_counts_by_name':dict(Counter(e['name'] for _,e in tasks)),
        'model_ids':sorted({e['args'].get('Model Id') for _,e in tasks if 'Model Id' in e['args']}),
        'flow_links':links,'export_warnings':logs,
        'acl_flow_diagnostics':{'source_count':sum(e['ph']=='s' for _,e in dangling),
            'destination_count':sum(e['ph']=='f' for _,e in dangling),
            'unique_flow_id_count':len(ids),
            'destination_event_indices':[i for i,e in dangling if e['ph']=='f'],
            'repeated_destination_flow_ids':sum(count['f']>1 for count in ids.values()),
            'source_less_destination_groups':sum(count['s']==0 and count['f']>0 for count in ids.values())},
        'host_iteration_mapping': 'exact framework flows for all 15 tasks' if mode=='eager' else
            'not established: reused Model/Task IDs and source-less ACL flow endpoints do not identify individual Host iterations'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Private job directory containing rank0/result.json and rank1/result.json')
    parser.add_argument('--launch-log',type=Path,required=True,help='Private supervisor stdout log containing profiler export diagnostics')
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results')/'moe-graph-ab.json')
    args=parser.parse_args()
    reports=[json.loads((args.input/f'rank{rank}/result.json').read_text()) for rank in [0,1]]
    assert reports[0]['config']==reports[1]['config'] and reports[0]['job_id']==reports[1]['job_id']
    assert reports[0]['script_sha256']==reports[1]['script_sha256']
    assert reports[0]['script_sha256']==digest(Path(__file__).with_name('moe_ab.py'))
    assert reports[0]['environment']==reports[1]['environment']
    cfg={key:value for key,value in reports[0]['config'].items() if key!='job_id'}
    assert cfg['implementation']=='mc2' and cfg['profile'] and not cfg['smoke']
    assert cfg['profile_iterations']==3
    preflight=Path(__file__).with_name('results')/'moe-preflight.json'
    output={'schema_version':'wiki-moe-graph-ab-public/1','measured_on':'2026-11-18',
        'scope':'single two-rank A5 W4A8MXFP MC2 operator chain: identical weights/inputs/routes, eager versus its NPUGraph replay',
        'config':cfg,'script_sha256':reports[0]['script_sha256'],'analysis_script_sha256':digest(Path(__file__)),
        'analysis_helper_sha256':digest(Path(__file__).with_name('analyze_sp.py')),
        'preflight_reference':{'file':'moe-preflight.json','sha256':digest(preflight),
            'launch_condition':'Prepend <vllm_ascend package>/_cann_ops_custom/vendors/custom_transformer to ASCEND_CUSTOM_OPP_PATH before Python/device initialization, preserving existing entries.',
            'evidence':'launch receipts and controlled preflight; loaded-library paths were not directly captured'},
        'health_condition':{'warning':'81AF8009 UB bus error','provenance':'pre-existing warning from external resource audit',
            'limit':'warning impact not isolated; results describe this run and do not predict healthy-machine performance'},
        'measurement':reports[0]['measurement'],
        'limits':[
            'No Scheduler, ModelRunnerV2, Attention or cross-DP graph negotiation was executed; this does not prove full-model FULL_DECODE_ONLY.',
            'Both implementations were compared in eager correctness checks, but only MC2 was captured and timed in this run.',
            'NPU event time includes Host submission/peer gaps; Host submit time excludes completion synchronization.',
            'Fixed small shapes are sensitive to Python/Host submission; ratios cannot be extrapolated to model or service speedup.',
            'Profiling was collected separately from timing and contains three active Host calls per configuration/rank.',
            'Replay export reports missing ACL-to-NPU flows. All 24 exported tasks are retained; none is assigned to a Host iteration.',
            'Model ID 63, repeated Task IDs and three MODEL_EXECUTE records describe device structure, not a proven Host iteration mapping.',
            'Each rank/configuration uses its own Host-origin timeline; neither cross-rank nor Host/Device clocks were independently calibrated.',
            'MC2 communication remains inside the Dispatch/Combine device tasks; no separate peer transfer or fabricated communication stream is drawn.',
            'Raw traces, logs, hostnames, process IDs, paths and allocation details remain private.'],
        'environment':reports[0]['environment'],'ranks':[],'trace_windows':[]}
    expected_count=cfg['rounds']*4*cfg['repeats']
    expected_mode=expected_count//2
    for rank,report in enumerate(reports):
        assert report['status']=='passed' and report['stage']=='complete' and report['rank']==rank
        assert len(report['samples'])==expected_count
        assert report['capture']['mc2']['status']=='passed' and not report['capture']['mc2']['eager_fallback']
        assert report['capture']['mc2']['rtol']==0 and report['capture']['mc2']['atol']==0
        assert report['correctness']['mc2_after_timing_and_profile']=='passed'
        assert report['correctness']['global_tasks']==cfg['tokens']*2*cfg['topk']
        samples=[{k:v for k,v in sample.items() if k!='job_id'} for sample in report['samples']]
        assert len({(s['mode'],s['cycle'],s['block'],s['iteration']) for s in samples})==expected_count
        calculated={}
        for mode in ['eager','replay']:
            rows=[s for s in samples if s['mode']==mode];assert len(rows)==expected_mode
            calculated['mc2/'+mode]={key:stats([s[field] for s in rows]) for key,field in [
                ('npu_event','npu_event_us'),('host_submit','host_submit_us'),('matched_rank_max','matched_rank_max_us')]}
        assert calculated==report['summary']
        output['ranks'].append({'rank':rank,'source_result_sha256':digest(args.input/f'rank{rank}/result.json'),
            'correctness':report['correctness'],'capture':report['capture'],'inputs':report['inputs'],
            'weights':report['weights'],'samples':samples,'summary':calculated})
        for mode in ['eager','replay']:
            path,=(args.input/f'rank{rank}/profiles/mc2'/mode).rglob('trace_view.json')
            receipt=next(p for p in report['profiles'] if p['mode']==mode and p['implementation']=='mc2')
            source=next(a for a in receipt['artifacts'] if a['path'].endswith('/trace_view.json'))
            assert source['sha256']==digest(path) and source['bytes']==path.stat().st_size
            output['trace_windows'].append(trace_window(path,rank,mode,report['job_id'],receipt['active_iterations'],args.launch_log))
    left,right=[{(s['mode'],s['cycle'],s['block'],s['iteration']):s for s in r['samples']} for r in output['ranks']]
    assert left.keys()==right.keys()
    for key in left:
        maximum=max(left[key]['npu_event_us'],right[key]['npu_event_us'])
        assert left[key]['matched_rank_max_us']==right[key]['matched_rank_max_us']==maximum
    assert sum(sum(r['correctness']['expected_expert_counts']) for r in output['ranks'])==128
    output['correctness_summary']={'global_token_expert_assignments':128,
        'rank_assignments':[sum(r['correctness']['expected_expert_counts']) for r in output['ranks']],
        'graph_matches_own_eager_exactly':True,'inputs_and_buffer_addresses_stable':True,
        'final_checks_after_timing_and_profile':'passed on both ranks'}
    output['environment']['cann_version_from_all_profiler_reports']=sorted({w['cann_version_from_profiler'] for w in output['trace_windows']})
    output['findings']={'timing':'Both local NPU-event and Host-submit medians are lower in replay for this fixed-input MC2 micro-experiment.',
        'trace':'Eager exports five tasks per linked Host call on Stream 61. Replay exports the same five kernel names on Stream 26 plus model control on Stream 61; individual Host-call correspondence is unproven.'}
    audit_public(output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'samples_per_rank':expected_count,'windows':[(w['mode'],w['rank'],len(w['device_events']),len(w['flow_links'])) for w in output['trace_windows']]}))


if __name__=='__main__':
    main()
