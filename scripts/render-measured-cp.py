#!/usr/bin/env python3
"""Render DCP off/on, both ranks and all flow-attributed device tracks.

Decimal arithmetic preserves exported durations. No minimum event width and no
cross-rank clock alignment. HCCL aggregate tracks retain their real identity.
"""
from decimal import Decimal as D
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'experiments/context-parallel/results/profiling-events.json'
OUTPUT = ROOT / 'dist/assets/cp-dcp-ranks-measured.svg'
X0, WIDTH, END = D(240), D(660), D(600)
SCALE = WIDTH / END
INK, MUTED = '#172b4d', '#52627a'


def text(x, y, s, size=16, fill=INK, weight=400, anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>'


def render(data):
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="1110" viewBox="0 0 960 1110" role="img" aria-labelledby="title desc">',
           '<title id="title">DCP 关闭与开启：两个 rank 的计算、通信和同步时间线</title>',
           '<desc id="desc">A5真实事件复绘。TP2，batch4，总Q heads8，KV head1，dim128，历史4096，BF16，iteration2。DCP1各rank一个FIA。DCP2各rank展示Stream61计算和同步、Stream59同步、Group group_name_3 Communication聚合轨道，共17个事件。四个rank/配置窗口分别以首事件归零，统一0至600微秒比例；不是同步起点，不能判断跨rank先后。蓝色计算与布局转换、青色通信父区间、橙色EVENT WAIT、灰色EVENT RECORD。通信父区间含同步，不是纯传输时间。单次算子实验不代表服务加速比。来源profiling-events.json。</desc>',
           '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>',
           '<rect width="960" height="1110" fill="#f8fafc"/>',
           text(30, 40, 'DCP 关闭 → 开启：两卡的计算、通信与同步', 24, weight=700),
           text(30, 70, '真实事件复绘 · TP2 · batch 4 · Q heads 8 / KV head 1 · dim 128 · 历史 4096 · BF16', 15, MUTED),
           '<rect x="30" y="88" width="900" height="58" rx="8" fill="#fff7ed"/>',
           text(46, 112, '同一时间比例，每个 rank / 配置以自己的首事件归零。', 17, '#9a4f18', 600),
           text(46, 135, '未校准跨 rank 时钟；以下上下对照只能比较本地任务组成与区间。', 15, '#9a4f18')]
    windows = {w['scene_rank']:w for w in data['selected_operator_windows']}
    def axis(top,bottom):
        for tick in range(0,601,100):
            x=X0+tick*SCALE
            out.extend([text(x,top,str(tick),14,MUTED,anchor='middle'),
                        f'<path d="M{x} {top+8}V{bottom}" stroke="#dce4ed" stroke-dasharray="3 4"/>'])
    def bars(w, ys):
        events = w['events']; assert len(events)==w['selected_scope_summary']['device_event_count']
        assert min(D(e['relative_start_us']) for e in events)==0
        assert max(D(e['relative_start_us'])+D(e['duration_us']) for e in events)==D(w['selected_scope_summary']['task_span_us'])
        for e in events:
            start,dur=D(e['relative_start_us']),D(e['duration_us'])
            assert D(e['start_us'])-D(w['selected_scope_summary']['device_origin_us'])==start
            assert 0<=start<start+dur<=END
            y=ys[e['track_name']]; x=X0+start*SCALE; width=dur*SCALE
            name=e['kernel']
            color='#b45309' if name=='EVENT WAIT' else '#64748b' if name=='EVENT RECORD' else '#0f766e' if e['task_type']=='HCCL parent/AICPU' else '#2563eb'
            tip=f"{w['scene_rank']} · {e['track_name']} · {name} · {start}–{start+dur} µs · duration {dur} µs"
            out.append(f'<rect data-window="{w["scene_rank"]}" data-track="{escape(e["track_name"])}" data-start-us="{start}" data-duration-us="{dur}" x="{x}" y="{y}" width="{width}" height="24" fill="{color}"><title>{escape(tip)}</title></rect>')
            if name in ('EVENT RECORD', 'EVENT WAIT') and dur < D('0.1'):
                out.append(f'<circle cx="{x}" cy="{y+12}" r="2" fill="{color}"><title>{escape(tip)}；圆点仅定位起点，不表示持续时间</title></circle>')
            if name=='FusedInferAttentionScore':
                label=f'FIA {dur:.2f} µs'
                out.append(text(x,y-9,label,14,'#1d4ed8'))
            elif name=='_pack_token_peer':
                out.append(text(x,y-9,'打包',14,'#1d4ed8'))
            elif name=='_fused_merge_kernel':
                out.append(text(x+width,y-9,'归并 / 转换',14,'#1d4ed8',anchor='end'))
            elif e['task_type']=='HCCL parent/AICPU':
                label=('Q AllGather' if 'allGather' in name else 'output/LSE AllToAll')+f' · {dur:.2f} µs'
                right='alltoall' in name
                out.append(text(x+width if right else x,y+48,label,14,'#0f766e',anchor='end' if right else 'start'))
    out.append(text(30,179,'DCP1：两卡各用 4 个 Q heads，读取完整历史',19,weight=700))
    axis(206,333)
    for rank,y in [(0,238),(1,300)]:
        w=windows[f'TP2_DCP1/rank{rank}']
        out+= [text(44,y+15,f'rank {rank} · Stream 61',16,weight=600)]
        bars(w,{'Stream 61':y})
    for rank,top in [(0,377),(1,669)]:
        w=windows[f'TP2_DCP2/rank{rank}']
        out += [f'<rect x="30" y="{top-24}" width="900" height="280" rx="8" fill="#ffffff" stroke="#dce4ed"/>',
                text(44,top,f'DCP2 · rank {rank}：收集 8 个 Q heads，读取本地 2048 个历史位置',18,weight=600)]
        axis(top+30,top+208)
        ys={'Stream 61':top+65,'Stream 59':top+119,'Group group_name_3 Communication':top+173}
        for key,y in ys.items():
            if key.startswith('Group'):
                out += [text(44,y+10,'HCCL 通信聚合轨道',15,weight=600),text(44,y+31,'group_name_3',14,MUTED)]
            else:
                out += [text(44,y+13,key,16,weight=600),text(44,y+34,'计算 / 转换 / 同步' if '61' in key else '同步控制',14,MUTED)]
        bars(w,ys)
        out.append(text(900,top+244,f"本窗口首尾跨度 {D(w['selected_scope_summary']['task_span_us']):.2f} µs",14,MUTED,anchor='end'))
    colors=[('#2563eb','计算 / 布局转换'),('#0f766e','HCCL 父区间'),('#b45309','EVENT WAIT'),('#64748b','EVENT RECORD')]
    for x,(c,label) in zip([30,265,490,715],colors):
        out += [f'<rect x="{x}" y="960" width="14" height="14" fill="{c}"/>',text(x+23,973,label,15)]
    out += [text(30,1005,'横轴：µs。小圆点定位极短同步事件，不表示时长；所有条宽保持真实比例。',15,MUTED),
            text(30,1031,'HCCL 是通信聚合轨道，不是 stream 编号；父区间包含同步，不与内部传输任务累加。',15,MUTED),
            text(30,1057,'只含选定 Attention 调用，未包含完整模型；空白未归因，不据此计算服务加速比。',15,MUTED),
            text(30,1088,'采集：2026-11-05 · 数据：experiments/context-parallel/results/profiling-events.json',14,MUTED),'</svg>']
    return '\n'.join(out)+'\n'


if __name__ == '__main__':
    OUTPUT.write_text(render(json.loads(DATA.read_text())),encoding='utf-8')
    print(OUTPUT)
