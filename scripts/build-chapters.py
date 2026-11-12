#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render this Wiki's controlled Markdown subset with the Python standard library.

Supported blocks: paragraphs, headings, flat lists, quotes, fenced code, tables,
and standalone local SVG/PNG figures. The chapter manuscripts remain the source.
"""
from pathlib import Path
import html
import hashlib
import re
from image_assets import image_dimensions, resolve_asset, rewrite_local_link

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
CHAPTERS = [
    ('execution', '从一轮调度到一次前向.md', '从一轮调度到一次前向', '从一轮调度<br><span>到一次前向</span>'),
    ('speculation', '投机解码与流式指标.md', '投机解码：调度、元数据与流式时延', '投机解码<br><span>与流式指标</span>'),
    ('load-balancing', '负载均衡与专家并行.md', '负载均衡：DPLB 与 EPLB', '负载均衡<br><span>DPLB 与 EPLB</span>'),
    ('ep-communication', 'EP通信与通算融合.md', 'EP 通信与计算重叠：算子选择与融合执行', 'EP 通信与计算重叠<br><span>算子选择与融合执行</span>'),
    ('context-parallel', '上下文并行与Attention归并.md', '上下文并行：KV 分片与 Attention 归并', '上下文并行<br><span>KV 分片与 Attention 归并</span>'),
    ('pd-disaggregation', 'PD分离系统与KV生命周期.md', 'PD 分离系统：请求交接与 KV 生命周期', 'PD 分离系统<br><span>请求交接与 KV 生命周期</span>'),
    ('kv-cache-layouts', 'KV缓存布局与递归状态.md', 'KV 缓存布局与递归状态', 'KV 缓存布局<br><span>与递归状态</span>'),
    ('sparse-offload', '稀疏KV卸载与HiSparse.md', '稀疏 KV：Decode Offload 与 HiSparse', '稀疏 KV 卸载<br><span>走向 HiSparse</span>'),
]
LINKS = {name: key + '.html' for key, name, *_ in CHAPTERS}
LINKS['LLM推理调度与执行.md'] = 'index.html'
TOKEN = re.compile(r'(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*.+?\*\*)')


def asset_url(name):
    digest = hashlib.sha256((DIST / name).read_bytes()).hexdigest()[:8]
    return name + '?rev=' + digest


def chapter_navigation(current, local_links):
    chapters = [('index', '调度、PD 与 KV')] + [(slug, title) for slug, _, title, _ in CHAPTERS]
    items = []
    for slug, title in chapters:
        label = f'<span>{html.escape(title)}</span>'
        if slug == current:
            items.append(f'<details class="wiki-chapter-current" data-nav-chapter="{slug}" open><summary aria-current="page">{label}</summary><nav class="main-nav chapter-nav" aria-label="本章目录">{local_links}</nav></details>')
        else:
            items.append(f'<a class="wiki-chapter-link" data-nav-chapter="{slug}" href="{slug}.html">{label}</a>')
    return '<nav class="main-nav wiki-chapters" aria-label="学习篇章">' + ''.join(items) + '</nav>'


def inline(source):
    result = []
    for part in TOKEN.split(source):
        if part.startswith('`') and part.endswith('`'):
            result.append('<code>' + html.escape(part[1:-1]) + '</code>')
        elif part.startswith('**') and part.endswith('**'):
            result.append('<strong>' + inline(part[2:-2]) + '</strong>')
        elif part.startswith('[') and (link := re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', part)):
            label, url = link.groups()
            url = LINKS.get(url, rewrite_local_link(url))
            external = ' target="_blank" rel="noopener"' if url.startswith('https://') else ''
            result.append(f'<a class="inline-source" href="{html.escape(url, quote=True)}"{external}>{inline(label)}</a>')
        else:
            result.append(html.escape(part))
    return ''.join(result)


def figure(alt, reference):
    path = resolve_asset(reference, ROOT / 'content', DIST / 'assets')
    width, height = image_dimensions(path)
    name = html.escape(path.name, quote=True)
    original = path.suffix.lower() == '.png'
    label = '原图（PNG）' if original else '矢量图源（SVG）'
    alt = html.escape(f'{label}：{alt}' if original else alt, quote=True)
    fits_phone = not original and width <= 350
    width, height = (format(value, '.15f').rstrip('0').rstrip('.') for value in (width, height))
    figure_class = "diagram-figure technical-figure" + (" raster-figure" if original else "")
    if fits_phone:
        figure_class += " compact-figure"
    scroll_hint = '' if fits_phone else '；窄屏可横向滚动'
    src = asset_url("assets/" + path.name)
    return f'<figure class="{figure_class}"><a class="diagram-expand" href="{src}" target="_blank" rel="noopener" title="打开完整{label}"><img width="{width}" height="{height}" src="{src}" alt="{alt}" loading="lazy"></a><figcaption>{alt}<em>{label}；点击查看完整图源{scroll_hint} ↗</em></figcaption></figure>'


def render_markdown(source, key):
    lines = source.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
    parts, headings, paragraph = [], [], []
    section_open = False
    spec_player = spec_stream = ''
    execution_lab = (ROOT / 'scripts' / 'execution-lab.html').read_text(encoding='utf-8') if key == 'execution' else ''
    lb_panels = {kind: (ROOT / 'scripts' / f'{kind}-lab.html').read_text(encoding='utf-8') for kind in ('dplb', 'eplb')} if key == 'load-balancing' else {}
    offload_lab = (ROOT / 'scripts' / 'offload-lab.html').read_text(encoding='utf-8') if key == 'sparse-offload' else ''
    ep_lab = (ROOT / 'scripts' / 'ep-lab.html').read_text(encoding='utf-8') if key == 'ep-communication' else ''
    ep_timeline = (ROOT / 'scripts' / 'ep-timeline.html').read_text(encoding='utf-8') if key == 'ep-communication' else ''
    kv_decode_player = (ROOT / 'scripts' / 'kv-decode-player.html').read_text(encoding='utf-8') if key == 'kv-cache-layouts' else ''
    figure_summary = None
    if key == 'speculation':
        fragment = (ROOT / 'scripts' / 'spec-player.html').read_text(encoding='utf-8')
        split = fragment.index('<section class="spec-lab" id="spec-stream"')
        spec_player, spec_stream = fragment[:split], fragment[split:]

    def flush():
        if paragraph:
            parts.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush(); i += 1; continue
        if line == '<!-- SPEC_PLAYER -->':
            flush(); parts.append(spec_player); i += 1; continue
        if line == '<!-- SPEC_STREAM -->':
            flush()
            if section_open:
                parts.append('</section>')
                section_open = False
            parts.append(spec_stream)
            i += 1; continue
        if line == '<!-- KV_DECODE_PLAYER -->':
            flush(); parts.append(kv_decode_player); i += 1; continue
        if line == '<!-- EXECUTION_LAB -->':
            flush(); parts.append(execution_lab); i += 1; continue
        if line == '<!-- OFFLOAD_LAB -->':
            flush(); parts.append(offload_lab); i += 1; continue
        if line == '<!-- EP_LAB -->':
            flush(); parts.append(ep_lab); i += 1; continue
        if line == '<!-- EP_TIMELINE -->':
            flush(); parts.append(ep_timeline); i += 1; continue
        if line in ('<!-- DPLB_LAB -->', '<!-- EPLB_LAB -->'):
            flush(); parts.append(lb_panels.get('dplb' if 'DPLB' in line else 'eplb', '')); i += 1; continue
        if match := re.fullmatch(r'<!-- DETAILS: (.+) -->', line):
            flush(); parts.append('<details class="execution-source-details"><summary>' + html.escape(match.group(1)) + '</summary>'); i += 1; continue
        if line == '<!-- /DETAILS -->':
            flush(); parts.append('</details>'); i += 1; continue
        if match := re.fullmatch(r'<!-- COLLAPSIBLE_FIGURE: (.+) -->', line):
            flush(); figure_summary = match.group(1); i += 1; continue
        if line.startswith('```'):
            flush(); code = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i]); i += 1
            parts.append('<pre class="source-code"><code>' + html.escape('\n'.join(code)) + '</code></pre>')
            i += 1; continue
        if match := re.match(r'^(#{2,3}) (.+)$', line):
            flush(); marks, title = match.groups()
            if marks == '##':
                if section_open:
                    parts.append('</section>')
                identifier = key + '-' + str(len(headings) + 1)
                headings.append((identifier, title))
                parts.append(f'<section class="lesson observed" id="{identifier}" data-title="{html.escape(title, quote=True)}"><h2>{inline(title)}</h2>')
                section_open = True
            else:
                parts.append('<h3>' + inline(title) + '</h3>')
            i += 1; continue
        if match := re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line):
            flush(); rendered = figure(*match.groups())
            if figure_summary:
                rendered = '<details class="chapter-figure-details"><summary>' + html.escape(figure_summary) + '</summary>' + rendered + '</details>'
                figure_summary = None
            parts.append(rendered); i += 1; continue
        if line.startswith('>'):
            flush(); quote = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                quote.append(lines[i].strip()[1:].strip()); i += 1
            parts.append('<blockquote class="chapter-scope"><p>' + inline(' '.join(quote)) + '</p></blockquote>')
            continue
        if line.startswith('|') and i + 1 < len(lines) and re.fullmatch(r'[| :\-]+', lines[i + 1].strip()):
            flush()
            headers = [cell.strip() for cell in line.strip('|').split('|')]
            parts.append('<div class="comparison-table"><table><thead><tr>' + ''.join('<th>' + inline(c) + '</th>' for c in headers) + '</tr></thead><tbody>')
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [cell.strip() for cell in lines[i].strip().strip('|').split('|')]
                if len(cells) != len(headers):
                    raise ValueError(f'Inconsistent chapter table: {lines[i]}')
                parts.append('<tr>' + ''.join('<td>' + inline(c) + '</td>' for c in cells) + '</tr>')
                i += 1
            parts.append('</tbody></table></div>'); continue
        if re.match(r'^(?:- |\d+\. )', line):
            flush(); ordered = bool(re.match(r'^\d+\. ', line)); tag = 'ol' if ordered else 'ul'
            parts.append('<' + tag + ' class="chapter-list">')
            pattern = r'^\d+\. (.*)' if ordered else r'^- (.*)'
            while i < len(lines) and (match := re.match(pattern, lines[i].strip())):
                parts.append('<li>' + inline(match[1]) + '</li>'); i += 1
            parts.append('</' + tag + '>'); continue
        paragraph.append(line); i += 1
    flush()
    if section_open:
        parts.append('</section>')
    return '\n'.join(parts), headings


def build_chapters():
    for key, name, title, display_title in CHAPTERS:
        source = (ROOT / 'content' / name).read_text(encoding='utf-8')
        body, headings = render_markdown(source, key)
        local_nav = ''.join(f'<a href="#{identifier}">{html.escape(label)}</a>' for identifier, label in headings)
        if key == 'speculation':
            player_links = '<a href="#spec-player">投机验证播放器</a><a href="#spec-stream">流式指标实验室</a>'
            extra = f'<link rel="stylesheet" href="{asset_url("spec.css")}"><link rel="stylesheet" href="{asset_url("spec-scene.css")}"><script defer src="{asset_url("spec-model.js")}"></script><script defer src="{asset_url("spec-scene.js")}"></script><script defer src="{asset_url("spec-app.js")}"></script>'
            next_link = '<a href="load-balancing.html">继续：DPLB 与 EPLB →</a>'
            action = '<a class="primary-button" href="#spec-player">逐步看接受与回退 →</a><a class="text-button" href="#spec-stream">对照 ITL 与 TPOT ↓</a>'
        elif key == 'execution':
            player_links = '<a href="#fusion-lab">Triton 输入准备图解</a>'
            extra = f'<link rel="stylesheet" href="{asset_url("execution-lab.css")}"><script defer src="{asset_url("execution-lab.js")}"></script>'
            next_link = '<a href="speculation.html">继续：投机解码与流式指标 →</a>'
            action = '<a class="primary-button" href="#execution-1">从输入打包开始 →</a><a class="text-button" href="#fusion-lab">看 Triton 怎样融合准备 ↓</a>'
        elif key == 'load-balancing':
            player_links = '<a href="#dplb-lab">DPLB 请求分发</a><a href="#eplb-lab">EPLB rank 泳道</a>'
            extra = f'<link rel="stylesheet" href="{asset_url("lb.css")}"><script defer src="{asset_url("lb-model.js")}"></script><script defer src="{asset_url("lb-app.js")}"></script>'
            next_link = '<a href="ep-communication.html">继续：EP 通信与计算重叠 →</a>'
            action = '<a class="primary-button" href="#dplb-lab">分发一批新请求 →</a><a class="text-button" href="#eplb-lab">对照专家放置与 rank 泳道 ↓</a>'
        elif key == 'sparse-offload':
            player_links = '<a href="#offload-lab">TopK 缓存与地址实验室</a>'
            extra = f'<link rel="stylesheet" href="{asset_url("offload.css")}"><script defer src="{asset_url("offload-model.js")}"></script><script defer src="{asset_url("offload-app.js")}"></script>'
            next_link = '<a href="kv-cache-layouts.html">← 回看 KV 缓存布局与递归状态</a>'
            action = '<a class="primary-button" href="#offload-lab">解析一次 TopK →</a><a class="text-button" href="#sparse-offload-8">看 HiSparse 依赖泳道 ↓</a>'
        elif key == 'ep-communication':
            player_links = '<a href="#ep-lab">专家任务与激活副本实验室</a><a href="#ep-timeline">MC2 / Fused-MC2 模拟 profiling</a>'
            extra = f'<link rel="stylesheet" href="{asset_url("ep.css")}"><link rel="stylesheet" href="{asset_url("ep-timeline.css")}"><script defer src="{asset_url("ep-model.js")}"></script><script defer src="{asset_url("ep-app.js")}"></script><script defer src="{asset_url("ep-timeline-model.js")}"></script><script defer src="{asset_url("ep-timeline-app.js")}"></script>'
            next_link = '<a href="context-parallel.html">继续：上下文并行与 Attention 归并 →</a>'
            action = '<a class="primary-button" href="#ep-communication-1">沿 EP 数据路径开始 →</a><a class="text-button" href="#ep-timeline">对照 MC2 与 Fused-MC2 泳道 ↓</a>'
        elif key == 'context-parallel':
            player_links = ''
            extra = ''
            next_link = '<a href="pd-disaggregation.html">继续：PD 分离系统与 KV 生命周期 →</a>'
            action = '<a class="primary-button" href="#context-parallel-1">从 TP 的 KV 切分开始 →</a>'
        elif key == 'pd-disaggregation':
            player_links = ''
            extra = ''
            next_link = '<a href="kv-cache-layouts.html">继续：KV 缓存布局与递归状态 →</a>'
            action = '<a class="primary-button" href="#pd-disaggregation-1">从请求交接开始 →</a>'
        elif key == 'kv-cache-layouts':
            player_links = '<a href="#kv-decode-player">KV 寻址播放器</a>'
            extra = f'<link rel="stylesheet" href="{asset_url("kv-decode.css")}"><script defer src="{asset_url("kv-decode-model.js")}"></script><script defer src="{asset_url("kv-decode-app.js")}"></script>'
            next_link = '<a href="sparse-offload.html">继续：稀疏 KV 卸载与 HiSparse →</a>'
            action = '<a class="primary-button" href="#kv-decode-player">播放一次 Decode 的 KV 寻址 →</a><a class="text-button" href="#kv-cache-layouts-5">按后端查缓存形状 ↓</a>'
        else:
            raise ValueError(f'Unknown chapter: {key}')
        chapter_nav = chapter_navigation(key, player_links + local_nav)
        page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{html.escape(title)}：沿输入字段、状态记账和源码调用定位推理性能问题。"><title>{html.escape(title)} · Inference Wiki</title><link rel="icon" href="assets/favicon.svg"><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="reading.css"><link rel="stylesheet" href="{asset_url("chapters.css")}"><script defer src="{asset_url("chapter.js")}"></script>{extra}</head>
<body><div class="reading-progress" id="readingProgress"></div>
<aside class="sidebar" id="sidebar"><a class="brand" href="index.html"><span class="brand-mark"><i></i><i></i><i></i></span><span>Inference<span class="brand-sub">图解推理系统</span></span><b>WIKI</b></a><button class="search-trigger" data-search>搜索本章 <kbd>⌘ K</kbd></button><div class="nav-group-label">学习篇章</div>{chapter_nav}<div class="sidebar-bottom">字段、调用链与 Trace 排查</div></aside>
<div class="page-shell"><header class="topbar"><button class="mobile-menu" id="menuToggle" aria-label="展开目录" aria-expanded="false">☰</button><div class="breadcrumb"><a href="index.html">推理系统</a><span>/</span><strong>{html.escape(title)}</strong></div><a class="source-link" href="{key}.md" download>下载本章 Markdown ↗</a></header>
<main><header class="chapter-hero"><div class="eyebrow">INFERENCE SYSTEMS / 源码与排查</div><h1>{display_title}</h1><div class="hero-actions">{action}</div></header><div class="article-layout"><article class="chapter-article">{body}<nav class="chapter-next" aria-label="继续阅读"><a href="index.html">← 调度与 PD 总览</a>{next_link}</nav><a class="download-source" href="technical-wiki-source.zip" download>下载全部 Markdown 与图源（SVG / PNG） ↓</a></article><aside class="page-toc"><span>本章导读</span>{player_links}{local_nav}</aside></div></main></div>
<dialog id="searchDialog"><div class="search-dialog-input"><input id="searchInput" type="search" aria-label="搜索本章知识点" placeholder="搜索本章内容"><button id="closeSearch" aria-label="关闭搜索">Esc</button></div><div id="searchResults" class="search-results"></div></dialog></body></html>'''
        (DIST / (key + '.html')).write_text(page, encoding='utf-8')
    print(f'已从 Markdown 生成 {len(CHAPTERS)} 篇图解网页。')


if __name__ == '__main__':
    build_chapters()
