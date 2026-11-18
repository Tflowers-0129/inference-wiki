#!/usr/bin/env python3
"""Sanitize two-rank SP results and extract flow-associated Level1 windows.

Read private captures only; do not execute device work. Public output excludes
hostnames, process/thread IDs, reservation IDs, absolute paths and timestamps.
Unlinked SQE/transport events remain visible as temporal context, never invented
collective dependencies. An external resource audit supplied the health warning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal as D
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stat(values):
    values = sorted(values)
    return {'count': len(values), 'min': values[0], 'median': statistics.median(values),
            'p95_nearest_rank': values[math.ceil(.95 * len(values)) - 1], 'max': values[-1]}


def duration_event(e):
    return e.get('ph') == 'X'


def device_event(e):
    return duration_event(e) and ('Task Type' in e.get('args', {}) or 'rank_size' in e.get('args', {}))


def point(e):
    return e['pid'], e['tid'], D(e['ts'])


def inside(e, scope):
    return ((e['pid'], e['tid']) == (scope['pid'], scope['tid'])
            and D(scope['ts']) <= D(e['ts']) < D(scope['ts']) + D(scope['dur']))


def analyze_trace(path, n, mode, rank, job, iteration):
    raw = json.loads(path.read_text(), parse_float=D)
    events = raw if isinstance(raw, list) else raw['traceEvents']
    names = {(e['pid'], e['tid']): e['args']['name'] for e in events
             if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
    expected_scope = f'wiki/{job}/sp_ab/N{n}/{mode}/iteration{iteration}/input{iteration % 2}/rank{rank}'
    scopes = [(i, e) for i, e in enumerate(events) if duration_event(e) and e.get('name') == expected_scope]
    if len(scopes) != 1:
        raise ValueError('Expected one exact job/case/rank/iteration scope')
    scope_index, scope = scopes[0]
    origin = D(scope['ts'])
    phases = [(i, e) for i, e in enumerate(events) if duration_event(e)
              and e.get('name', '').startswith(f'sp/{mode}/') and inside(e, scope)]
    phase_at = lambda event: next((e['name'] for _, e in phases if inside(event, e)), None)
    tasks = defaultdict(list)
    cpu = defaultdict(list)
    flows = defaultdict(lambda: defaultdict(list))
    for index, event in enumerate(events):
        if device_event(event):
            tasks[point(event)].append((index, event))
        if duration_event(event) and event.get('cat') == 'cpu_op':
            cpu[point(event)].append((index, event))
        if (event.get('cat') == 'async_npu' and event.get('name') == 'torch_to_npu'
                and event.get('ph') in ['s', 'f']):
            flows[str(event['id'])][event['ph']].append((index, event))
    direct, links, unresolved = {}, [], []
    for pairs in flows.values():
        starts = [(i, e) for i, e in pairs.get('s', []) if inside(e, scope)]
        if not starts:
            continue
        if len(starts) != 1 or len(pairs.get('f', [])) != 1:
            unresolved.append({'source_event_indices': [i for i, _ in starts], 'reason': 'nonunique flow endpoints'})
            continue
        si, source = starts[0]
        fi, finish = pairs['f'][0]
        matches = tasks.get(point(finish), [])
        if len(matches) != 1:
            unresolved.append({'source_event_indices': [si], 'target_flow_event_index': fi,
                               'reason': f'{len(matches)} exact target duration matches'})
            continue
        target_index, target = matches[0]
        source_ops = cpu.get(point(source), [])
        source_op = min(source_ops, key=lambda item: D(item[1]['dur'])) if source_ops else None
        phase = phase_at(source)
        direct[target_index] = {'method': 'exact_local_flow', 'host_phase': phase}
        links.append({'source_flow_event_index': si, 'target_flow_event_index': fi,
                      'host_operator_event_index': source_op[0] if source_op else None,
                      'host_operator': source_op[1]['name'] if source_op else None,
                      'host_phase': phase, 'device_event_index': target_index,
                      'source_relative_us': str(D(source['ts']) - origin),
                      'destination_relative_us': str(D(finish['ts']) - origin),
                      'raw_endpoint_delta_us_not_latency': str(D(finish['ts']) - D(source['ts']))})
    if unresolved or not direct:
        raise ValueError(f'Cannot silently select an incomplete scope: {len(unresolved)} unresolved flows')
    first = min(D(events[i]['ts']) for i in direct)
    last = max(D(events[i]['ts']) + D(events[i]['dur']) for i in direct)
    # Context is selected in the SAME device export domain, not by a Host scope interval.
    window_tasks = [(i, e) for i, e in enumerate(events) if device_event(e)
                    and D(e['ts']) < last and D(e['ts']) + D(e['dur']) > first]
    parents = [(i, events[i]) for i in direct if 'rank_size' in events[i].get('args', {})]
    selected = []
    for index, e in window_tasks:
        args = e.get('args', {})
        track = names.get((e['pid'], e['tid']))
        if not track or not re.fullmatch(r'(Stream \d+|Group group_name_\d+ Communication)', track):
            raise ValueError('Unexpected device track name; inspect before public export')
        association = direct.get(index)
        if association is None:
            linked = [(pi, pe) for pi, pe in parents
                      if args.get('connection_id') == pe['args'].get('connection_id')
                      and D(pe['ts']) <= D(e['ts'])
                      and D(e['ts']) + D(e['dur']) <= D(pe['ts']) + D(pe['dur']) + D('.001')]
            if len(linked) == 1:
                pi, _ = linked[0]
                association = {'method': 'same_local_connection_within_parent',
                               'parent_event_index': pi, 'host_phase': direct[pi]['host_phase']}
            else:
                association = {'method': 'device_window_context_only',
                               'note': 'No unique local flow/connection; do not infer a collective dependency'}
        item = {'event_index': index, 'name': e['name'], 'track_name': track,
                'physical_stream_id': args.get('Physic Stream Id'),
                'task_type': args.get('Task Type', 'HCCL parent/AICPU'),
                'relative_start_us': str(D(e['ts']) - origin), 'duration_us': str(D(e['dur'])),
                'association': association}
        # Keep technical IDs needed for trace lookup, but omit pid/tid and absolute timestamps.
        item['task_fields'] = {k: args[k] for k in ['Task Id', 'Model Id', 'Batch Id', 'Subtask Id', 'connection_id'] if k in args}
        if 'rank_size' in args:
            item['collective'] = {k: args.get(k) for k in ['rank_size', 'data_type', 'count', 'alg_type', 'relay', 'retry']}
        selected.append(item)
    info_path, = path.parent.parent.glob('profiler_info_*.json')
    info = json.loads(info_path.read_text())
    actual_level = info['config']['experimental_config']['_profiler_level']
    assert int(info['rank_id']) == rank and actual_level == 'Level1'
    communication_path = path.parent / 'communication.json'
    communication = json.loads(communication_path.read_text(), parse_float=D)
    comm_rows = []
    # Exact raw operation name identifies its communication analysis entry. Timestamp
    # in communication.json is rounded, so do not use it as an exact time join.
    for index, parent in parents:
        matches = []
        for step, row in communication.items():
            for key, value in row.get('collective', {}).items():
                if key.split('@')[0] == parent['name']:
                    matches.append((step, value))
        assert len(matches) == 1
        step, value = matches[0]
        time_info = value['Communication Time Info']
        comm_rows.append({'parent_event_index': index, 'exported_step': step,
                          'operation': parent['name'],
                          'exporter_time_fields': {k: str(v) for k, v in time_info.items() if k != 'Start Timestamp(us)'},
                          'interpretation': 'exporter categories overlap; not peer causality or measured wire bytes'})
    return {'case': f'N{n}-{mode}-rank{rank}', 'valid_rows': n, 'mode': mode, 'rank': rank,
            'iteration': iteration, 'input_variant': iteration % 2,
            'trace_sha256': digest(path), 'trace_bytes': path.stat().st_size,
            'profiler_info_sha256': digest(info_path), 'communication_sha256': digest(communication_path),
            'profiler_level': actual_level, 'cann_version_from_profiler': info['cann_version'],
            'torch_npu_version_from_profiler': info['torch_npu_version'],
            'scope_event_index': scope_index, 'host_scope_duration_us': str(D(scope['dur'])),
            'time_origin': 'this rank and configuration Host scope start in its own export; not a shared clock',
            'device_first_relative_us': str(first - origin), 'device_last_relative_us': str(last - origin),
            'device_span_us': str(last - first),
            'host_phases': [{'event_index': i, 'name': e['name'], 'relative_start_us': str(D(e['ts']) - origin),
                             'duration_us': str(D(e['dur']))} for i, e in phases],
            'tracks': sorted({e['track_name'] for e in selected}), 'events': selected,
            'flow_links': links, 'unresolved_flows': unresolved,
            'event_counts_by_association': dict(Counter(e['association']['method'] for e in selected)),
            'communication_analysis': comm_rows}


def audit_public(value):
    text = json.dumps(value, ensure_ascii=False)
    for forbidden in ['localhost', '/Users/', '/root/', '/home/', '.codex/', 'reservation', '"pid"', '"tid"']:
        assert forbidden not in text, forbidden
    assert not re.search(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])', text), 'possible IP address'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Private sp-profile root containing rank0.json/rank1.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results') / 'sp-ab.json')
    parser.add_argument('--iteration', type=int, default=2)
    args = parser.parse_args()
    reports = [json.loads((args.input / f'rank{rank}.json').read_text()) for rank in [0, 1]]
    assert reports[0]['job_id'] == reports[1]['job_id']
    assert reports[0]['configuration'] == reports[1]['configuration']
    assert reports[0]['script_sha256'] == reports[1]['script_sha256']
    assert reports[0]['environment'] == reports[1]['environment']
    output = {'schema_version': 'wiki-sp-ab-public/1', 'measured_on': '2026-11-18',
              'scope': 'single A5 TP2 HCCL/fused-RMSNorm micro-path experiment; not a vLLM/FlashComm model or service',
              'health_condition': {'warning': '81AF8009 UB bus error',
                  'provenance': 'pre-existing device warning reported by the external resource audit',
                  'limit': 'Results describe this run only; warning impact was not isolated and no healthy-machine performance claim is made'},
              'configuration': reports[0]['configuration'], 'script_sha256': reports[0]['script_sha256'],
              'analysis_script_sha256': digest(Path(__file__)),
              'association': {'cross_rank_clock_calibrated': False, 'host_device_clock_independently_calibrated': False,
                  'matching': 'same job, case, mode, explicit iteration and input variant; collective signatures cross-checked',
                  'device_mapping': 'exact local torch_to_npu flow endpoints; local connection expansion explicitly labelled',
                  'context': 'all exported device tracks intersecting the flow-attributed device window; unlinked context remains unassigned',
                  'clock': 'one origin per rank/configuration, shared by its tracks; identical microsecond scale across panels',
                  'not_inferred': ['cross-rank arrival order', 'which peer caused a wait', 'physical meaning of exported Stream 65535', 'wire bandwidth']},
              'measurement_limits': [
                  'Timing samples were collected without profiler; selected traces are separate instrumented iterations.',
                  'Host timings include local completion synchronization; NPU events include dependencies and host/peer gaps.',
                  'HCCL parents overlap transport and synchronization events; do not add all lane durations.',
                  'Communication analysis categories and raw count do not establish measured wire traffic.',
                  'RMSNorm row count halves, but B restores full rows with AllGather; this is not full-model FlashComm.',
                  'One run and the known device warning limit generalization; no service or healthy-hardware speedup conclusion.'],
              'environment': {'hardware': 'Ascend950DT (A5)', 'python': reports[0]['environment']['python'],
                  'torch': reports[0]['environment']['torch'], 'torch_npu': reports[0]['environment']['torch_npu'],
                  'packages': reports[0]['environment']['packages'],
                  'cann_version_from_rank_probe_verified': reports[0]['environment']['cann_version_verified'],
                  'HCCL_OP_EXPANSION_MODE': reports[0]['environment']['HCCL_OP_EXPANSION_MODE']},
              'ranks': [], 'paired_rank_max_summary': [], 'trace_sources': [], 'selected_windows': []}
    for rank, report in enumerate(reports):
        assert report['status'] == 'passed' and report['rank'] == rank and report['device'] == 'npu'
        assert len(report['samples']) == 200 and len(report['correctness']) == 4 and len(report['profiles']) == 4
        assert all(c['path_exact'] and c['padding_zero'] and c['path_max_abs_error'] == 0 for c in report['correctness'])
        expected_stats = []
        for n in [128, 129]:
            for mode in ['A', 'B']:
                rows = [s for s in report['samples'] if s['valid_rows'] == n and s['mode'] == mode]
                assert len(rows) == 50
                expected_stats.append({'valid_rows': n, 'mode': mode,
                    'host_ms': stat([s['host_ms'] for s in rows]), 'npu_event_ms': stat([s['npu_event_ms'] for s in rows])})
        assert expected_stats == report['statistics']
        output['ranks'].append({'rank': rank, 'source_result_sha256': digest(args.input / f'rank{rank}.json'),
                               'correctness': report['correctness'], 'samples': report['samples'],
                               'statistics': expected_stats})
    output['correctness_summary'] = {'rank_cases': 8, 'both_paths_exact': True,
        'padding_zero': True, 'max_abs_error_against_fp64': max(c['oracle_max_abs_error'][m]
            for r in reports for c in r['correctness'] for m in ['A', 'B']), 'atol': .004, 'rtol': .004}
    for n in [128, 129]:
        for mode in ['A', 'B']:
            paired = []
            for iteration in range(50):
                rows = [next(s for s in r['samples'] if s['valid_rows'] == n and s['mode'] == mode
                             and s['iteration'] == iteration) for r in reports]
                assert rows[0]['input_variant'] == rows[1]['input_variant'] and rows[0]['order'] == rows[1]['order']
                paired.append(max(s['npu_event_ms'] for s in rows))
            output['paired_rank_max_summary'].append({'valid_rows': n, 'mode': mode, 'npu_event_ms': stat(paired),
                'limit': 'maximum matched local duration, not a synchronized makespan'})
    for n in [128, 129]:
        for mode in ['A', 'B']:
            pair = []
            for rank in [0, 1]:
                trace, = (args.input / 'profiles' / f'N{n}' / mode / f'rank{rank}').rglob('trace_view.json')
                window = analyze_trace(trace, n, mode, rank, reports[rank]['job_id'], args.iteration)
                pair.append(window)
                output['trace_sources'].append({k: window[k] for k in ['case', 'trace_sha256', 'trace_bytes',
                    'profiler_info_sha256', 'communication_sha256', 'profiler_level', 'cann_version_from_profiler',
                    'torch_npu_version_from_profiler', 'scope_event_index', 'iteration', 'input_variant', 'event_counts_by_association']})
                if n == 128:
                    output['selected_windows'].append(window)
            signatures = [[(e['name'], e['collective']) for e in w['events'] if 'collective' in e] for w in pair]
            assert signatures[0] == signatures[1], 'Collective signature mismatch across ranks'
    output['environment']['cann_version_from_all_profiler_reports'] = sorted({s['cann_version_from_profiler'] for s in output['trace_sources']})
    output['findings'] = {'N128': 'B has fewer RMSNorm rows but higher unprofiled local NPU-event medians on both ranks in this run.',
        'trace': 'Selected B windows contain ReduceScatter, local RMSNorm and an added AllGather with synchronization; this does not isolate the cause of the entire timing difference.'}
    audit_public(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': args.output.name, 'trace_sources': len(output['trace_sources']),
        'windows': [(w['case'], len(w['events']), w['event_counts_by_association']) for w in output['selected_windows']],
        'correctness': output['correctness_summary']}))


if __name__ == '__main__':
    main()
