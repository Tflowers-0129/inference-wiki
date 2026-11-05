#!/usr/bin/env python3
"""Render editable teaching diagrams for KV ownership and Attention merging."""
from fractions import Fraction
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'dist' / 'assets'
INK, MUTED, LINE = '#172b4d', '#475569', '#cbd5e1'
BLUE, BLUE_BG = '#1d4ed8', '#eff6ff'
TEAL, TEAL_BG = '#0f766e', '#e6f4f1'
AMBER, AMBER_BG = '#92400e', '#fff7ed'


def text(x, y, value, size=20, fill=INK, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def rect(x, y, w, h, fill='#fff', stroke=LINE, radius=8, dashed=False):
    dash = ' stroke-dasharray="4 4"' if dashed else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')


def begin(height, title, desc):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
        f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">' + escape(title) + '</title>',
        '<desc id="desc">' + escape(desc) + '</desc>',
        '<defs><style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",'
        '"Microsoft YaHei","Noto Sans CJK SC",sans-serif}</style>'
        '<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
        'orient="auto"><path d="M1 1L8 4.5L1 8" fill="none" stroke="#475569" '
        'stroke-width="1.5"/></marker></defs>',
        rect(0, 0, 1120, height, '#f8fafc', '#f8fafc', 0),
        text(32, 48, title, 28, weight=700),
    ]


def finish(parts):
    return '\n'.join(parts + ['</svg>']) + '\n'


def tp_shards():
    parts = begin(
        1020, 'TP 按 KV head 划分；DCP 再沿序列划分',
        '教学拓扑，GQA有16个query heads、2个KV heads、TP4。rank0的初始Q heads为0到3，'
        'rank1为4到7，rank2为8到11，rank3为12到15。TP4时rank0和rank1各保存KV head0的全序列，'
        'rank2和rank3各保存KV head1的全序列。加入DCP2，组0、1和组2、3分别按位置奇偶切分同一KV head，'
        '每rank的本地KV位置数从8变为4，不增加设备。query还需组内交换，局部Attention结果需要归并。'
        '图表示数据归属，不是profiling或执行时间线。')
    parts += [text(32, 82, '教学拓扑 · GQA：Q heads=16，KV heads=2，TP=4；示例序列位置 0…7', 19, MUTED)]
    q_ranges = ['0…3', '4…7', '8…11', '12…15']
    for panel_y, dcp in [(112, False), (524, True)]:
        parts.append(rect(32, panel_y, 1056, 380))
        if dcp:
            heading = 'TP4 + DCP2：同一 KV head 的两个副本，改为保存不同位置'
            subheading = 'DCP 组 [0,1] 对应 KV head 0；DCP 组 [2,3] 对应 KV head 1'
        else:
            heading = '只有 TP4：KV head 数少于 rank 数，同一 head 仍会复制'
            subheading = '每个 rank 只持有自己的 Q heads，但保存所需 KV head 的完整序列'
        parts += [text(52, panel_y + 35, heading, 22, weight=700),
                  text(52, panel_y + 68, subheading, 18, MUTED)]
        for x, label in [(52, 'rank'), (150, '初始 Q heads'), (330, '本地 KV head'), (540, '本地 KV 的位置')]:
            parts.append(text(x, panel_y + 105, label, 18, MUTED, 600))
        for rank in range(4):
            row_y = panel_y + 125 + rank * 60
            head = rank // 2
            color, bg = (BLUE, BLUE_BG) if head == 0 else (TEAL, TEAL_BG)
            parts += [text(52, row_y + 27, str(rank), 21, weight=700),
                      text(150, row_y + 27, q_ranges[rank], 21),
                      text(330, row_y + 27, 'KV head ' + str(head), 20, color, 600)]
            for pos in range(8):
                x = 540 + pos * 64
                local = not dcp or pos % 2 == rank % 2
                parts.append(rect(x, row_y, 52, 40, bg if local else '#f8fafc',
                                  color if local else LINE, 5, not local))
                parts.append(text(x + 26, row_y + 27, pos if local else '·', 21,
                                  color if local else MUTED, 600, 'middle'))
        if dcp:
            parts.append(text(52, panel_y + 365, '实色：本地保存 K/V；虚线：该位置由组内另一 rank 保存。', 18, MUTED))
    parts += [text(32, 945, '每 rank 的 KV 位置数：8 → 4；设备数仍为 4。', 21, TEAL, 700),
              text(32, 980, '分片后还需要 query 组内交换与 Attention 结果归并；此图不表示执行时序。', 19, MUTED)]
    return finish(parts)


def pcp_head_tail():
    rank_positions = [[0, 1, 6, 7], [2, 3, 4, 5]]
    assert [sum(pos + 1 for pos in group) for group in rank_positions] == [18, 18]
    assert sum(range(1, 5)) == 10 and sum(range(5, 9)) == 26
    parts = begin(
        800, 'PCP 的 head-tail 分片：query 数相同，因果工作也要看位置',
        '教学构造，无历史前缀，8个位置做包含自身的因果Attention。PCP2把8个query切成4个连续chunk，'
        '每chunk2个位置。rank0持有chunk0和chunk3，即原位置0、1、6、7；rank1持有chunk1和chunk2，'
        '即2、3、4、5。每head的query-key位置对数分别为1加2加7加8等于18，3加4加5加6等于18。'
        '若按前4和后4连续分片则为10和26。本例无padding，所有位置身份保持不变，不是设备耗时。')
    parts += [text(32, 82, '教学构造 · PCP=2，L=8，每个 chunk 两个 query；无历史前缀、无 padding', 19, MUTED)]
    xs = [256 + pos * 96 for pos in range(8)]
    for chunk in range(4):
        rank = 0 if chunk in (0, 3) else 1
        color, bg = (BLUE, BLUE_BG) if rank == 0 else (TEAL, TEAL_BG)
        parts.append(rect(xs[2 * chunk] - 4, 120, 184, 54, bg, color, 6))
        parts.append(text(xs[2 * chunk] + 88, 154, 'chunk ' + str(chunk), 20, color, 600, 'middle'))
    parts.append(text(32, 215, '原始 query 位置', 20, weight=600))
    for pos, x in enumerate(xs):
        rank = 0 if pos in rank_positions[0] else 1
        color, bg = (BLUE, BLUE_BG) if rank == 0 else (TEAL, TEAL_BG)
        parts += [rect(x, 188, 80, 42, bg, color, 5), text(x + 40, 216, pos, 22, color, 600, 'middle')]
    parts.append(text(32, 272, '每行可见 key 数', 20, weight=600))
    for pos, x in enumerate(xs):
        parts.append(text(x + 40, 272, pos + 1, 21, MUTED, anchor='middle'))
    parts.append('<line x1="32" y1="302" x2="1088" y2="302" stroke="#cbd5e1" stroke-width="1.5"/>')
    for rank, y in [(0, 342), (1, 440)]:
        color, bg = (BLUE, BLUE_BG) if rank == 0 else (TEAL, TEAL_BG)
        parts += [text(32, y + 27, 'PCP rank ' + str(rank), 22, color, 700),
                  text(32, y + 59, 'chunk 0 + 3' if rank == 0 else 'chunk 1 + 2', 19, MUTED)]
        for pos, x in enumerate(xs):
            local = pos in rank_positions[rank]
            parts.append(rect(x, y, 80, 42, bg if local else '#f8fafc', color if local else LINE, 5, not local))
            if local:
                parts.append(text(x + 40, y + 28, pos, 22, color, 600, 'middle'))
    for rank, x in [(0, 32), (1, 576)]:
        color, bg = (BLUE, BLUE_BG) if rank == 0 else (TEAL, TEAL_BG)
        parts += [rect(x, 548, 512, 102, bg, color),
                  text(x + 20, 584, 'rank ' + str(rank) + ' 的因果位置对 / head', 20, color, 600),
                  text(x + 20, 624, '1 + 2 + 7 + 8 = 18' if rank == 0 else '3 + 4 + 5 + 6 = 18', 25, color, 700)]
    parts += [text(32, 700, '连续前后切分：10 / 26；head-tail 分片：18 / 18。', 22, weight=700),
              text(32, 740, 'query 保留原始 position 与因果可见范围，不按本地行号重新编号。', 20, MUTED),
              text(32, 775, '这里只数每 head 的 query-key 位置对；相同计数不等于相同实测耗时。', 18, MUTED)]
    return finish(parts)


def attention_merge():
    z = [2, 6]
    local = [1, 12]
    merged = sum(Fraction(zj, sum(z)) * oj for zj, oj in zip(z, local))
    assert merged == Fraction(37, 4)
    assert Fraction(sum(local), 2) == Fraction(13, 2)
    parts = begin(
        810, '局部 Attention 输出，为什么不能直接平均？',
        '教学算例，同一个query和head。分片0的scores是0、0，values是0、2，归一化分母Z0为2，'
        '局部输出O0为1，LSE为ln2。分片1的scores是ln3、ln3，values是10、14，Z1为6，O1为12，'
        'LSE为ln6。稳定归并以ln6为最大LSE，得到未归一化权重1/3和1，归一化权重1/4和3/4，'
        '最终输出9.25。直接平均局部输出得到6.5，是错误结果。')
    parts += [text(32, 82, '教学算例 · 同一 query / head，两片 KV；LSE 记录各片 softmax 的归一化规模', 19, MUTED)]
    rows = [
        (0, 32, BLUE, BLUE_BG, ['scores = [0, 0]', 'values = [0, 2]', 'exp(scores) = [1, 1]', 'Z₀ = 2，L₀ = ln 2', 'O₀ = (0 + 2) / 2 = 1']),
        (1, 608, TEAL, TEAL_BG, ['scores = [ln 3, ln 3]', 'values = [10, 14]', 'exp(scores) = [3, 3]', 'Z₁ = 6，L₁ = ln 6', 'O₁ = (3×10 + 3×14) / 6 = 12']),
    ]
    for rank, x, color, bg, lines in rows:
        parts += [rect(x, 112, 480, 264, bg, color), text(x + 20, 150, 'KV 分片 ' + str(rank), 23, color, 700)]
        for i, value in enumerate(lines):
            parts.append(text(x + 20, 192 + i * 37, value, 21, color if i >= 3 else INK, 600 if i >= 3 else 400))
    parts += [
        '<path d="M272 376V404H478V440" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>',
        '<path d="M848 376V404H642V440" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>',
        text(318, 398, 'O₀、L₀', 19, BLUE, 600), text(718, 398, 'O₁、L₁', 19, TEAL, 600),
        rect(168, 452, 784, 194, '#fff', LINE),
        text(192, 488, '稳定归并：m = max(L₀, L₁) = ln 6', 22, weight=600),
        text(192, 526, 'w₀ = exp(L₀ − m) = 1/3，w₁ = exp(L₁ − m) = 1', 21),
        text(192, 564, '归一化权重：[1/4, 3/4]', 22, weight=600),
        text(192, 612, 'O = (1/4)×1 + (3/4)×12 = 9.25', 27, BLUE, 700),
        text(32, 696, '四项一起算：(0 + 2 + 3×10 + 3×14) / (1+1+3+3) = 9.25', 22, TEAL, 600),
        rect(32, 722, 1056, 54, AMBER_BG, '#d7ae63', 7),
        text(52, 758, '错误做法：直接平均局部输出，(1 + 12) / 2 = 6.5。', 22, AMBER, 600),
    ]
    return finish(parts)


def main():
    for name, render in [('cp-tp-kv-shards.svg', tp_shards),
                         ('cp-pcp-head-tail.svg', pcp_head_tail),
                         ('cp-attention-merge.svg', attention_merge)]:
        (ASSETS / name).write_text(render(), encoding='utf-8')
        print('Rendered ' + name)


if __name__ == '__main__':
    main()
