#!/usr/bin/env python3
"""Render both ranks from saved flow-attributed MC2/MegaMoE events.

Each row has its own origin. A shared scale is not a synchronized time base.
Bars preserve duration exactly; annotations never change event geometry.
"""
from decimal import Decimal as D
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'experiments/results/a5-moe-rank-comparison.json'
OUTPUT = ROOT / 'dist/assets/a5-mc2-megamoe-measured.svg'
X0, WIDTH, END = D(250), D(650), D(900)
SCALE = WIDTH / END
INK, MUTED = '#172b4d', '#52627a'
COLORS = {'dispatch_communication': '#0f766e', 'combine_communication': '#0f766e',
          'grouped_matmul_compute': '#2563eb', 'activation_quantization': '#b45309',
          'fused_moe_opaque': '#7355a4'}


def text(x, y, s, size=16, fill=INK, weight=400, anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>'


def render(data):
    assert not data['association']['cross_rank_clock_calibration_established']
    assert data['conditions']['profile_iteration'] == 2
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="750" viewBox="0 0 960 750" role="img" aria-labelledby="title desc">',
           '<title id="title">MC2 与 CANN MegaMoE：两个 rank 的同尺度设备时间线</title>',
           '<desc id="desc">真实事件复绘，采集于2026年11月4日。EP2，每源rank32个Token，W4A8MXFP，SiLU，eager，iteration2。四行均为实际Stream61，各自首任务归零，统一0至900微秒比例；未校准跨rank时钟。保留12个外部设备任务的实际起点与时长。没有独立通信流，融合算子内部未展开。空白未归因，不能从本图计算稳态或服务加速比。来源a5-moe-rank-comparison.json。</desc>',
           '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>',
           '<rect width="960" height="750" fill="#f8fafc"/>',
           text(30, 40, 'MC2 → CANN MegaMoE：把两个 rank 放在一起看', 24, weight=700),
           text(30, 71, '真实事件复绘 · EP2 · 每源 rank 32 Token · W4A8MXFP / SiLU · eager · iteration 2', 16, MUTED),
           '<rect x="30" y="89" width="900" height="58" rx="8" fill="#fff7ed"/>',
           text(46, 113, '统一 0～900 µs 比例；每行以自己的首任务为零。', 17, '#9a4f18', 600),
           text(46, 136, '四个零点不表示同一时刻，不能从上下位置判断跨 rank 的先后或等待原因。', 15, '#9a4f18')]
    rows = [(data['paired_windows'][0], 179, (238, 325)),
            (data['paired_windows'][1], 420, (479, 566))]
    for pair, top, ys in rows:
        impl = pair['implementation']
        assert [r['rank'] for r in pair['ranks']] == [0, 1]
        out.append(text(30, top, 'MC2：五个外部任务' if impl == 'MC2' else 'CANN MegaMoE：一个外部融合任务', 19, weight=700))
        for tick in range(0, 901, 150):
            x = X0 + tick * SCALE
            out += [text(x, top + 27, str(tick), 14, MUTED, anchor='middle'),
                    f'<path d="M{x} {top+35}V{ys[1]+34}" stroke="#dce4ed" stroke-dasharray="3 4"/>']
        for rank, y in zip(pair['ranks'], ys):
            rid = rank['rank']; tasks = rank['tasks']
            assert len(tasks) == (5 if impl == 'MC2' else 1)
            assert rank['physical_stream_ids'] == [61]
            assert min(D(t['relative_start_us_from_this_rank_first_task']) for t in tasks) == 0
            assert max(D(t['relative_end_us_from_this_rank_first_task']) for t in tasks) == D(rank['device_span_us'])
            out += [text(44, y+13, f'rank {rid} · Stream 61', 17, weight=600),
                    text(44, y+36, f'专家任务 {71 if rid == 0 else 57} 次', 14, MUTED)]
            for i, t in enumerate(tasks, 1):
                start = D(t['relative_start_us_from_this_rank_first_task']); dur = D(t['duration_us'])
                assert start + dur == D(t['relative_end_us_from_this_rank_first_task'])
                assert 0 <= start < start+dur <= END
                assert t['physical_stream_id'] == 61
                x, w = X0 + start*SCALE, dur*SCALE
                color = COLORS[t['category']]
                title = f"{t['kernel']} · rank {rid} · Stream 61 · {start}–{start+dur} µs · duration {dur} µs"
                out.append(f'<rect data-rank="{rid}" data-implementation="{impl}" data-start-us="{start}" data-duration-us="{dur}" x="{x}" y="{y}" width="{w}" height="25" fill="{color}"><title>{escape(title)}</title></rect>')
                if impl == 'MC2' and (rid == 0 or i in (1, 5)):
                    out.append(text(x+w/2, y-9, str(i), 15, color, 600, 'middle'))
                if impl == 'MC2' and rid == 1 and i in (1, 5):
                    out.append(text(x+w/2, y+18, f'{dur:.2f} µs', 15, '#ffffff', 500, 'middle'))
                if impl != 'MC2':
                    out.append(text(x+w+10, y+18, f'{dur:.2f} µs · 内部阶段未展开', 15, color))
            if impl == 'MC2' and rid == 1:
                x = X0+D(tasks[1]['relative_start_us_from_this_rank_first_task'])*SCALE
                out += [f'<path d="M{x+7} {y+28}V{y+65}H{x+65}" fill="none" stroke="#2563eb"/>',
                        text(x+72, y+70, '2–4：矩阵计算、激活与量化', 14, '#2563eb')]
            out.append(text(900, y+49, f"本行首尾跨度 {D(rank['device_span_us']):.2f} µs", 14, MUTED, anchor='end'))
    out += [text(30, 651, '1 Dispatch　2 GMM1　3 SwiGLU + 量化　4 GMM2　5 Combine', 16),
            text(30, 680, '本次通信与计算均记录在 Stream 61；没有单独导出的 HCCL 收发轨道。', 16, MUTED),
            text(30, 705, '留白只表示没有归属本轮的已记录任务。一次采集的跨度不等于稳态或服务时延。', 15, MUTED),
            text(30, 732, '采集：2026-11-04 · 数据：experiments/results/a5-moe-rank-comparison.json', 14, MUTED), '</svg>']
    return '\n'.join(out)+'\n'


if __name__ == '__main__':
    OUTPUT.write_text(render(json.loads(DATA.read_text())), encoding='utf-8')
    print(OUTPUT)
