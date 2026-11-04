#!/usr/bin/env python3
"""Render a device-time snapshot from the public relative-time evidence file.

No task receives a minimum display width. Decimal arithmetic preserves the
exported durations when projecting the shared 0..500 microsecond axis.
"""
from decimal import Decimal
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/results/a5-moe-swimlane.json"
OUTPUT = ROOT / "dist/assets/a5-mc2-megamoe-measured.svg"
D = Decimal
X0, PLOT_WIDTH, AXIS_END = D(250), D(800), D(500)
SCALE = PLOT_WIDTH / AXIS_END
COLORS = {"dispatch": "#0f766e", "compute": "#1d4ed8",
          "activation_quant": "#b45309", "combine": "#0f766e",
          "fused_opaque": "#475569"}


def number(value):
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def text(x, y, value, size=20, fill="#172b4d", weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def render(data):
    assert data["axis"]["unit"] == "us"
    assert D(data["axis"]["start"]) == 0 and D(data["axis"]["end"]) == AXIS_END
    scopes = {scope["implementation"]: scope for scope in data["scopes"]}
    assert len(scopes["MC2"]["tasks"]) == 5
    assert len(scopes["MegaMoe"]["tasks"]) == 1
    for scope in scopes.values():
        assert scope["physical_stream_id"] == 61
        for task in scope["tasks"]:
            start, end, duration = (D(task[key]) for key in ("start_us", "end_us", "duration_us"))
            assert 0 <= start < end <= AXIS_END
            assert end - start == duration
            assert task["physical_stream_id"] == 61
        assert min(D(task["start_us"]) for task in scope["tasks"]) == 0
        assert max(D(task["end_us"]) for task in scope["tasks"]) == D(scope["device_span_us"])

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="640" '
        'viewBox="0 0 1100 640" role="img" aria-labelledby="title desc">',
        '<title id="title">A5 实测：MC2 与 CANN MegaMoe 的外部设备轨迹</title>',
        '<desc id="desc">真实 profiling 的单次执行快照。条件为 EP2、每 rank 32 tokens、8 专家、'
        'top-k 2、hidden 和 intermediate 均为 1024、W4A8MXFP、SiLU、eager。'
        '选取 rank 0 的 iteration 2，两种实现各自以第一个设备任务归零，统一按 0 到 500 微秒绘制。'
        'MC2 的 stream 61 依次有 dispatch、GMM1、SwiGLU 量化、GMM2、combine 五个任务；'
        'CANN MegaMoe 的 stream 61 有一个外部融合任务。所有条宽严格按设备时间比例绘制。'
        '空白区间的原因未归因。没有跨 rank 对齐，也没有展开融合 kernel 内部阶段；'
        '该快照不能用于计算稳态加速比。精确数据与源 trace 校验值见 a5-moe-swimlane.json。</desc>',
        '<defs><style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",'
        '"Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style></defs>',
        '<rect width="1100" height="640" fill="#f8fafc"/>',
        text(32, 44, 'A5 实测：MC2 与 CANN MegaMoe 的外部设备轨迹', 28, weight=700),
        text(32, 78, '实测快照 · tokens/rank 32 · EP2 · W4A8MXFP · eager · rank 0 / iteration 2', 19, '#475569'),
        text(32, 110, '两种实现独立归零，共用 0～500 µs 刻度；条宽保持真实比例。', 20, '#475569'),
        '<rect x="32" y="132" width="1036" height="278" rx="8" fill="#ffffff" stroke="#cbd5e1"/>',
        text(250, 162, '设备局部时间 / µs', 18, '#475569'),
    ]
    for tick in range(0, 501, 100):
        x = number(X0 + D(tick) * SCALE)
        out.extend([
            f'<path d="M{x} 194V394" stroke="#cbd5e1" stroke-dasharray="3 5" fill="none"/>',
            text(x, 187, str(tick), 18, '#475569', anchor='middle'),
        ])

    for implementation, y in (("MC2", 222), ("MegaMoe", 332)):
        scope = scopes[implementation]
        out.extend([
            text(52, y + 24, 'MC2' if implementation == 'MC2' else 'CANN MegaMoe', 22, weight=700),
            text(52, y + 52, 'rank 0 · stream 61', 18, '#475569'),
        ])
        for task in scope["tasks"]:
            x = X0 + D(task["start_us"]) * SCALE
            width = D(task["duration_us"]) * SCALE
            color = COLORS[task["category"]]
            tooltip = (f'{task["id"]} {task["kernel"]}; physical stream 61; '
                       f'{task["start_us"]} to {task["end_us"]} us; duration {task["duration_us"]} us')
            out.extend([
                f'<rect id="{implementation}-{task["id"]}" x="{number(x)}" y="{y}" '
                f'width="{number(width)}" height="36" fill="{color}" '
                f'data-start-us="{task["start_us"]}" data-duration-us="{task["duration_us"]}">'
                f'<title>{escape(tooltip)}</title></rect>',
                text(number(x + width / 2), y - 10, task["id"], 20, color, 700, 'middle'),
            ])
        span = f'{D(scope["device_span_us"]):.2f}'
        out.append(text(1048, y + 58, f'末任务结束 {span} µs', 18, '#475569', anchor='end'))
    out.append(text(326, 356, '一个外部设备任务，内部未展开', 20, '#475569'))

    legend = [
        (52, 438, '1', 'Dispatch', 'dispatch'),
        (350, 438, '2', 'GMM1', 'compute'),
        (640, 438, '3', 'SwiGLU + 量化', 'activation_quant'),
        (52, 478, '4', 'GMM2', 'compute'),
        (350, 478, '5', 'Combine', 'combine'),
        (640, 478, 'M', 'MegaMoe（融合任务）', 'fused_opaque'),
    ]
    for x, y, ident, label, kind in legend:
        out.extend([
            f'<rect x="{x}" y="{y-19}" width="25" height="25" rx="3" fill="{COLORS[kind]}"/>',
            text(x + 12.5, y, ident, 18, '#ffffff', 700, 'middle'),
            text(x + 37, y, label, 20),
        ])
    out.extend([
        text(32, 528, '空白表示此 stream 无已采集任务，原因未归因。', 19, '#475569'),
        text(32, 558, '各实现独立归零；未对齐其他 rank，也未绘制 Host→NPU 微秒延迟。', 19, '#475569'),
        text(32, 588, '本图含采集扰动，不用于计算稳态加速比；融合算子的内部流水未展开。', 19, '#475569'),
        text(32, 620, '采集：2026-11-04 · 可复算数据：experiments/results/a5-moe-swimlane.json', 16, '#475569'),
        '</svg>',
    ])
    return '\n'.join(out) + '\n'


if __name__ == "__main__":
    evidence = json.loads(DATA.read_text(encoding="utf-8"))
    OUTPUT.write_text(render(evidence), encoding="utf-8")
    print(OUTPUT)
