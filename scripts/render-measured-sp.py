#!/usr/bin/env python3
"""Render sanitized SP A/B events with real exported tracks and shared scale.

Each rank/configuration retains a separate origin. The graph uses true widths;
small event markers locate events and never represent their durations. Overlap
within one exported stream is drawn in subrows without inventing new streams.
"""
from decimal import Decimal as D
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'experiments/profiling-ab/results/sp-ab.json'
OUTPUT = ROOT / 'dist/assets/sp-measured-rank-streams.svg'
W, H = 1120, 1870
LEFT, RIGHT, END = D(270), D(1080), D(500)
SCALE = (RIGHT - LEFT) / END
INK, MUTED = '#172b4d', '#52627a'
PALETTE = {'compute': '#2458bd', 'communication': '#0f766e', 'wait': '#a8550a',
           'record': '#64748b', 'launch': '#6d4db0', 'copy': '#16738b'}


def text(x, y, value, size=17, fill=INK, weight=400, anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'


def x(value):
    return LEFT + D(value) * SCALE


def kind(event):
    if 'collective' in event:
        return 'communication'
    name = event['name']
    if 'WAIT' in name:
        return 'wait'
    if 'RECORD' in name:
        return 'record'
    if name == 'RmsNorm':
        return 'compute'
    if name == 'HcclLaunchAicpuKernel':
        return 'launch'
    return 'copy'


def subrows(events):
    ends, placements = [], {}
    for event in sorted(events, key=lambda e: (D(e['relative_start_us']), -D(e['duration_us']))):
        start, end = D(event['relative_start_us']), D(event['relative_start_us']) + D(event['duration_us'])
        lane = next((i for i, previous in enumerate(ends) if previous <= start), len(ends))
        if lane == len(ends):
            ends.append(end)
        else:
            ends[lane] = end
        placements[event['event_index']] = lane
    return placements, len(ends)


def render(data):
    assert not data['association']['cross_rank_clock_calibrated']
    assert len(data['selected_windows']) == 4
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
           '<title id="title">SP 前后：两卡的真实计算、通信与等待</title>',
           '<desc id="desc">2026年11月18日A5双卡SP微实验。A为AllReduce加全量RMSNorm，B为ReduceScatter加半量RMSNorm再AllGather。BF16，128行，hidden4096。展示Level1 iteration2 input0四个独立窗口及全部导出设备轨道，统一0至500微秒比例，零点分别为各自Host scope开始。保留Stream61、59、27以及导出Stream65535或39和HCCL聚合轨道；未匹配flow的SQE和搬运事件用虚线表示，仅作为窗口上下文。两个rank均为B无profiler中位耗时更长。设备有既有81AF8009 UB bus error告警，结论只限此次微实验，不推断健康设备、整模型或服务性能。</desc>',
           '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>',
           f'<rect width="{W}" height="{H}" fill="#f8fafc"/>',
           text(28, 40, 'SP 前后：两卡的真实计算、通信与等待', 27, weight=700),
           text(28, 75, 'A：AllReduce → 全量 RMSNorm　｜　B：ReduceScatter → 半量 RMSNorm → AllGather', 19),
           text(28, 105, 'N=128 · H=4096 · BF16 · TP2 · A5 · Level1 · iteration 2 / input 0', 18, MUTED),
           '<rect x="28" y="126" width="1064" height="87" rx="8" fill="#ffffff" stroke="#cbd5e1"/>',
           text(44, 154, '关闭 profiler 后的 NPU event 中位数（每路每卡 50 个样本）', 18, weight=600)]
    metrics = {}
    for r in data['ranks']:
        for stat in r['statistics']:
            if stat['valid_rows'] == 128:
                metrics[(r['rank'], stat['mode'])] = stat['npu_event_ms']['median'] * 1000
    out += [text(44, 187, f"rank 0：A {metrics[(0,'A')]:.2f} → B {metrics[(0,'B')]:.2f} µs", 19),
            text(550, 187, f"rank 1：A {metrics[(1,'A')]:.2f} → B {metrics[(1,'B')]:.2f} µs", 19),
            '<rect x="28" y="225" width="1064" height="50" rx="8" fill="#fff7ed" stroke="#c99157"/>',
            text(44, 256, '设备已有 81AF8009（UB bus error）告警；仅描述本次结果，不外推健康设备表现。', 18, '#925019'),
            text(28, 304, '各窗口从本地 Host 范围起点计时，所有轨道同一比例；四个零点不作跨 rank 对齐。', 18, MUTED)]
    windows = {(w['mode'], w['rank']): w for w in data['selected_windows']}
    panel_tops = [335, 665, 995, 1325]
    for (mode, rank), top in zip([('A', 0), ('A', 1), ('B', 0), ('B', 1)], panel_tops):
        window = windows[(mode, rank)]
        assert window['valid_rows'] == 128 and window['iteration'] == 2 and window['input_variant'] == 0
        title = f"{mode} · rank {rank}　" + ('全量 RMSNorm [128,4096]' if mode == 'A' else '本地 RMSNorm [64,4096]，再恢复完整行')
        out += [f'<rect x="20" y="{top}" width="1080" height="316" rx="8" fill="#ffffff" stroke="#cbd5e1"/>',
                text(34, top+27, title, 20, weight=700)]
        for tick in range(0, 501, 100):
            xx = x(tick)
            out += [text(xx, top+53, tick, 15, MUTED, anchor='middle'),
                    f'<line x1="{xx}" x2="{xx}" y1="{top+60}" y2="{top+288}" stroke="#dce4ed" stroke-dasharray="3 4"/>']
        out.append(text(1080, top+27, 'µs · 独立时间原点', 15, MUTED, anchor='end'))
        ys = {'Host scope': top+77, 'Host phases': top+109}
        tracks = ['Stream 61', 'Stream 59', 'Stream 27']
        tracks += sorted(t for t in window['tracks'] if t.startswith('Group'))
        tracks += sorted(t for t in window['tracks'] if t.startswith('Stream') and t not in tracks)
        assert len(tracks) == 5 and set(tracks) == set(window['tracks'])
        for i, track in enumerate(tracks):
            ys[track] = top + 141 + i * 32
        labels = {'Host scope': 'Host：本轮范围', 'Host phases': 'Host：提交阶段',
                  'Stream 61': 'Stream 61｜计算 / 同步', 'Stream 59': 'Stream 59｜通信控制',
                  'Stream 27': 'Stream 27｜AICPU',
                  'Group group_name_0 Communication': 'HCCL 聚合｜group_name_0',
                  'Stream 65535': '导出 Stream 65535＊', 'Stream 39': 'Stream 39＊'}
        for track, yy in ys.items():
            out += [text(34, yy+5, labels.get(track, track), 16),
                    f'<line x1="{LEFT}" x2="{RIGHT}" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>']
        duration = D(window['host_scope_duration_us'])
        assert 0 < duration <= END
        out.append(f'<rect x="{LEFT}" y="{ys["Host scope"]-9}" width="{duration*SCALE}" height="18" rx="3" fill="#f1f5f9" stroke="#64748b"><title>Host scope {duration} µs；包含本地完成同步，非纯提交时间</title></rect>')
        out.append(text(x(duration)+8, ys['Host scope']+5, f'{duration:.2f}', 14, MUTED))
        phase_labels = {'all_reduce': 'AllReduce', 'full_rmsnorm': 'Norm', 'reduce_scatter': 'RS', 'local_rmsnorm': 'Norm', 'all_gather': 'AG'}
        for phase in window['host_phases']:
            key = phase['name'].split('/')[-1]
            color = PALETTE['compute' if 'norm' in key else 'communication']
            xx, width = x(phase['relative_start_us']), D(phase['duration_us']) * SCALE
            yy = ys['Host phases']
            out += [f'<rect x="{xx}" y="{yy-9}" width="{width}" height="18" rx="2" fill="{color}"><title>{escape(phase["name"])}，Host {phase["duration_us"]} µs</title></rect>',
                    text(xx+width/2, yy+5, phase_labels[key], 13, '#ffffff', 500, 'middle')]
        for track in tracks:
            track_events = [e for e in window['events'] if e['track_name'] == track]
            placements, levels = subrows(track_events)
            for event in track_events:
                start, dur = D(event['relative_start_us']), D(event['duration_us'])
                assert start >= 0 and start+dur <= END
                xx, width = x(start), dur*SCALE
                bar_height = D(18) / max(levels, 1)
                yy = D(ys[track]-9) + placements[event['event_index']] * bar_height
                color = PALETTE[kind(event)]
                context = event['association']['method'] == 'device_window_context_only'
                fill, dash = ('#f1f5f9', ' stroke-dasharray="2 1"') if context else (color, '')
                explanation = '仅窗口上下文，未建立flow/connection归属' if context else event['association']['method']
                tip = f'{event["name"]} · {track} · {start}–{start+dur} µs · duration {dur} µs · event {event["event_index"]} · {explanation}'
                out.append(f'<rect data-case="{window["case"]}" data-event-index="{event["event_index"]}" data-track="{escape(track)}" data-start-us="{start}" data-duration-us="{dur}" x="{xx}" y="{yy}" width="{width}" height="{bar_height}" fill="{fill}" stroke="{color}" stroke-width="0.8"{dash}><title>{escape(tip)}</title></rect>')
                if width < 1:
                    out.append(f'<circle cx="{xx}" cy="{yy+bar_height/2}" r="1.4" fill="{color}"><title>{escape(tip)}；圆点仅定位，不表示持续时间</title></circle>')
                if 'collective' in event:
                    short = 'AllReduce' if 'allReduce' in event['name'] else 'RS' if 'reduceScatter' in event['name'] else 'AG'
                    label = f'{short} {dur:.2f}'
                    out.append(text(xx+width/2, yy+bar_height-3, label, 13, '#ffffff', 500, 'middle'))
                elif event['name'] == 'RmsNorm':
                    # Labels can be wider than the true task; the colored bar retains its real duration.
                    out.append(text(xx+width+7, yy+15, f'Norm {dur:.2f} µs', 14, color, 600))
        counts = window['event_counts_by_association']
        linked = counts['exact_local_flow'] + counts['same_local_connection_within_parent']
        out.append(text(34, top+303, f"{len(window['events'])} 个设备事件：{linked} 个有来源关联，{counts['device_window_context_only']} 个仅作窗口上下文。RS = ReduceScatter；AG = AllGather。", 14, MUTED))
    out += [text(28, 1692, '图中 B 的 Norm 行数减半，同时多出 AllGather 及同步；不能只用 Norm 的变短判断整段收益。', 17),
            text(28, 1722, '青：HCCL 父区间　蓝：Norm　橙：等待　灰：记录　紫：AICPU；虚线表示未归属的窗口上下文。', 16, MUTED),
            text(28, 1751, '＊SQE / UBDMA 按导出轨道保留，重叠事件分层画；65535 的物理含义未确认。圆点不代表时长。', 16, MUTED),
            text(28, 1780, '父区间与内部任务有重叠，不累加求耗时；等待没有归因到具体对端，空白也不补作等待。', 16, MUTED),
            text(28, 1809, 'Host / Device 时间关系未经独立校准；此图是单次 profiling，不能代替上方无 profiler 的重复测量。', 16, MUTED),
            text(28, 1843, '采集：2026-11-18 · 原始事件索引、来源 SHA、N129 数字：experiments/profiling-ab/results/sp-ab.json', 14, MUTED),
            '</svg>']
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    OUTPUT.write_text(render(json.loads(DATA.read_text())), encoding='utf-8')
    print(OUTPUT.name)
