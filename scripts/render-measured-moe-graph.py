#!/usr/bin/env python3
"""Draw real MC2 eager/replay tasks, preserving independent local time origins.

Overview panels share one scale; the explicitly labelled replay detail has 10x
the scale. No missing graph flow or per-iteration task association is inferred.
"""
from decimal import Decimal as D
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'experiments/profiling-ab/results/moe-graph-ab.json'
OUTPUT = ROOT / 'dist/assets/moe-graph-rank-streams.svg'
W, H = 1120, 1940
LEFT, RIGHT = D(270), D(1080)
INK, MUTED = '#172b4d', '#52627a'
COLORS = {'dispatch': '#0f766e', 'gmm': '#2458bd', 'swiglu': '#6d4db0',
          'combine': '#0f766e', 'execute': '#16738b', 'wait': '#a8550a', 'record': '#64748b'}


def text(x, y, value, size=17, fill=INK, weight=400, anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'


def kind(name):
    if 'Dispatch' in name:
        return 'dispatch'
    if 'Combine' in name:
        return 'combine'
    if 'GroupedMatmul' in name:
        return 'gmm'
    if 'Swiglu' in name:
        return 'swiglu'
    return {'MODEL_EXECUTE': 'execute', 'MODEL_WAIT_COMPLETE': 'wait', 'NOTIFY RECORD': 'record'}[name]


def panel(window, top, end, zoom=False):
    scale = (RIGHT - LEFT) / D(end)
    def x(t):
        return LEFT + D(t) * scale
    mode, rank = window['mode'], window['rank']
    eager = mode == 'eager'
    height = 170 if eager else 230 if zoom else 190
    label = f'{mode} · rank {rank}'
    suffix = '全部 3 次 active 调用的导出事件'
    if zoom:
        label += ' · 10× 放大'
    out = [f'<rect x="20" y="{top}" width="1080" height="{height}" rx="8" fill="#ffffff" stroke="#cbd5e1"/>',
           text(34, top+27, f'{label}　{suffix}', 19, weight=700),
           text(1080, top+27, 'µs · 独立时间原点', 14, MUTED, anchor='end')]
    for tick in range(0, end+1, end//4):
        xx = x(tick)
        out += [text(xx, top+53, tick, 14, MUTED, anchor='middle'),
                f'<line x1="{xx}" x2="{xx}" y1="{top+61}" y2="{top+height-35}" stroke="#dce4ed" stroke-dasharray="3 4"/>']
    ys = {'Host': top+81, 'Stream 61': top+121}
    if not eager:
        ys['Stream 26'] = top+161 if zoom else top+150
    labels = {'Host':'Host：调用范围', 'Stream 61':'Stream 61｜算子' if eager else 'Stream 61｜图执行 / 等待',
              'Stream 26':'Stream 26｜图内算子'}
    for track, yy in ys.items():
        out += [text(34, yy+5, labels[track], 17),
                f'<line x1="{LEFT}" x2="{RIGHT}" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>']
    for scope in window['host_scopes']:
        start, duration = D(scope['relative_start_us']), D(scope['duration_us'])
        assert 0 <= start and start + duration <= end
        xx, width = x(start), duration*scale
        out.append(f'<rect data-scope="{mode}-{rank}-{scope["iteration"]}" x="{xx}" y="{ys["Host"]-9}" width="{width}" height="18" fill="#e2e8f0" stroke="#64748b"><title>Host iteration {scope["iteration"]}，{start}–{start+duration} µs；仅 executor.run() 调用，完成同步在范围外</title></rect>')
        out.append(text(xx+width/2, ys['Host']+5 if width >= 18 else ys['Host']+24,
                        scope['iteration'], 13, MUTED, 500, 'middle'))
    # Draw only the exact local eager framework relations. Lines denote identity,
    # not measured launch latency; the exporter can place a task before its source.
    for link in window['flow_links']:
        assert eager
        out.append(f'<line data-flow-target="{link["device_event_index"]}" x1="{x(link["source_relative_us"])}" y1="{ys["Host"]+9}" x2="{x(link["destination_relative_us"])}" y2="{ys["Stream 61"]-10}" stroke="#64748b" stroke-width="1" stroke-dasharray="2 2"><title>精确本地 torch_to_npu flow；Host iteration {link["host_iteration"]}，不把端点差值解释为提交延迟</title></line>')
    for event in window['device_events']:
        start, duration = D(event['relative_start_us']), D(event['duration_us'])
        assert 0 <= start and start+duration <= end
        xx, width, yy = x(start), duration*scale, ys[event['track_name']]
        category = kind(event['name']); color = COLORS[category]
        association = f'精确 flow：Host iteration {event["host_iteration"]}' if eager else '完整采集上下文；没有单次 Host iteration 关联'
        tip = f'{event["name"]} · {event["track_name"]} · {start}–{start+duration} µs · duration {duration} µs · event {event["event_index"]} · {association}'
        out.append(f'<rect data-panel="{mode}-{rank}-{"zoom" if zoom else "overview"}" data-event-index="{event["event_index"]}" data-track="{escape(event["track_name"])}" data-start-us="{start}" data-duration-us="{duration}" x="{xx}" y="{yy-10}" width="{width}" height="20" fill="{color}"><title>{escape(tip)}</title></rect>')
        if width < 1:
            out.append(f'<circle cx="{xx}" cy="{yy}" r="1.4" fill="{color}"><title>{escape(tip)}；圆点仅定位，不代表时长</title></circle>')
        if zoom:
            short = {'dispatch':'D', 'gmm':'G', 'swiglu':'S', 'combine':'C', 'wait':'WAIT'}.get(category)
            if short:
                out.append(text(xx+width/2, yy+4, short, 12, '#ffffff', 500, 'middle'))
    foot = '15 个设备任务均有精确 flow；虚线连接只表示事件关联，不表示提交延迟。' if eager else \
        '24 个设备任务全部保留；未建立 Host→设备 flow，不按时间接近或顺序指定 iteration。'
    out.append(text(34, top+height-12, foot, 14, MUTED))
    if zoom:
        out.append(text(34, top+196, 'Model ID 63；Task ID 重复出现。空白不补画为等待，MODEL_WAIT 不归因到具体对端。', 15, MUTED))
    return out


def render(data):
    assert len(data['trace_windows']) == 4
    windows = {(w['mode'],w['rank']):w for w in data['trace_windows']}
    for w in windows.values():
        assert len(w['host_scopes']) == 3
        if w['mode'] == 'replay':
            assert not w['flow_links'] and all(e['host_iteration'] is None for e in w['device_events'])
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">MC2 eager 与真实图回放：两卡的完整采集轨道</title>',
        '<desc id="desc">2026年11月18日A5双卡MC2微实验，32个Token每rank，8专家、topk2、W4A8MXFP。关闭profiler的重复测量中图回放的NPU event及Host提交中位数均较小。四个概览使用相同0至4000微秒比例，各自从本地首个Host scope开始；两幅回放放大图统一0至400微秒。eager每卡15个任务及对应flow；回放每卡24个任务，保留真实Stream61图执行和等待以及Stream26算子。回放导出缺少Host到设备flow，不把设备事件对应到某次Host调用。设备已有81AF8009告警，仅说明本次小形状算子链，不推断整模型或健康设备性能。</desc>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>',
        f'<rect width="{W}" height="{H}" fill="#f8fafc"/>',
        text(28,40,'MC2 eager 与真实图回放：两卡的完整采集轨道',27,weight=700),
        text(28,76,'每 rank 32 Token · EP2 · 8 专家 / topk2 · H=1024 · intermediate=1024 · W4A8MXFP',18),
        text(28,106,'同一 MC2 算子链、权重、输入和路由；仅切换 eager / NPUGraph replay。',18,MUTED),
        '<rect x="28" y="126" width="1064" height="104" rx="8" fill="#ffffff" stroke="#cbd5e1"/>',
        text(44,153,'关闭 profiler 后的中位数 · 每模式每卡 120 个样本 · ABBA 两轮',18,weight=600),
        text(44,183,'NPU event（µs）',17,MUTED),text(44,214,'Host 提交（µs）',17,MUTED)]
    for rank,xx in [(0,275),(1,690)]:
        s=data['ranks'][rank]['summary']
        out += [text(xx,183,f"rank {rank}：{s['mc2/eager']['npu_event']['median_us']:.2f} → {s['mc2/replay']['npu_event']['median_us']:.2f}",19),
                text(xx,214,f"rank {rank}：{s['mc2/eager']['host_submit']['median_us']:.2f} → {s['mc2/replay']['host_submit']['median_us']:.2f}",19)]
    out += ['<rect x="28" y="242" width="1064" height="67" rx="8" fill="#fff7ed" stroke="#c99157"/>',
            text(44,268,'设备已有 81AF8009（UB bus error）告警；结果仅限本次微实验。',17,'#925019'),
            text(44,294,'replay 导出缺少 ACL→NPU flow：下图展示完整采集，不声称某次回放的完整因果链。',17,'#925019'),
            text(28,339,'概览：四个窗口统一 0–4000 µs，均保留全部 3 次 active 调用的导出轨道',20,weight=600),
            text(28,365,'各 rank / 模式分别以首个 Host 范围为零点；不据此对齐跨 rank 时钟。',17,MUTED)]
    for mode,rank,top in [('eager',0,385),('eager',1,565),('replay',0,745),('replay',1,945)]:
        out += panel(windows[(mode,rank)],top,4000)
    out += [text(28,1173,'回放细节：下面两幅统一放大 10×，横轴改为 0–400 µs',21,weight=600),
            text(28,1201,'D = Dispatch　G = GroupedMatmul　S = SwigluGroupQuant　C = Combine',17,MUTED)]
    for rank,top in [(0,1220),(1,1460)]:
        out += panel(windows[('replay',rank)],top,400,zoom=True)
    out += [text(28,1723,'青绿：Dispatch / Combine　蓝：GMM　紫：SwiGLU　深青：MODEL_EXECUTE　橙：MODEL_WAIT',16,MUTED),
            text(28,1752,'灰点：NOTIFY RECORD（仅定位，非时长）；所有矩形的长度均来自原始持续时间。',17,MUTED),
            text(28,1781,'Dispatch / Combine 含 MC2 通信，但未导出独立的对端传输轨道；图中不另造通信流。',17),
            text(28,1810,'NPU event 包含 Host 提交与对端间隙；Host 提交不含完成同步。该小形状容易受 Host 开销影响。',17),
            text(28,1839,'Host / Device 时间关系未经独立校准；上下两种比例的 profiling 图不能代替无 profiler 的统计。',17,MUTED),
            text(28,1868,'图回放与各自 eager 输出精确一致；专家任务共 128 次（rank 0：60，rank 1：68）。不外推模型加速。',17),
            text(28,1906,'采集：2026-11-18 · 完整样本、原始事件索引与来源 SHA：experiments/profiling-ab/results/moe-graph-ab.json',14,MUTED),
            '</svg>']
    return '\n'.join(out)+'\n'


if __name__ == '__main__':
    OUTPUT.write_text(render(json.loads(DATA.read_text())),encoding='utf-8')
    print(OUTPUT.name)
