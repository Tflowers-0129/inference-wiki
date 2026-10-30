#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render this Wiki's controlled Markdown subset with the Python standard library.

Supported blocks: paragraphs, headings, flat lists, quotes, fenced code, tables,
and standalone local SVG figures. The chapter manuscripts remain the source.
"""
from pathlib import Path
import html
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
CHAPTERS = [
    ('execution', '从一轮调度到一次前向.md', '从一轮调度到一次前向', '从一轮调度<br><span>到一次前向</span>'),
    ('speculation', '投机解码与流式指标.md', '投机解码：调度、元数据与流式时延', '投机解码<br><span>与流式指标</span>'),
]
LINKS = {name: key + '.html' for key, name, *_ in CHAPTERS}
LINKS['LLM推理调度与执行.md'] = 'index.html'
TOKEN = re.compile(r'(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*.+?\*\*)')


def inline(source):
    result = []
    for part in TOKEN.split(source):
        if part.startswith('`') and part.endswith('`'):
            result.append('<code>' + html.escape(part[1:-1]) + '</code>')
        elif part.startswith('**') and part.endswith('**'):
            result.append('<strong>' + inline(part[2:-2]) + '</strong>')
        elif part.startswith('[') and (link := re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', part)):
            label, url = link.groups()
            url = LINKS.get(url, url)
            external = ' target="_blank" rel="noopener"' if url.startswith('https://') else ''
            result.append(f'<a class="inline-source" href="{html.escape(url, quote=True)}"{external}>{inline(label)}</a>')
        else:
            result.append(html.escape(part))
    return ''.join(result)


def figure(alt, reference):
    name = Path(reference).name
    svg = ET.parse(DIST / 'assets' / name).getroot()
    width, height = svg.attrib.get('width', '1120'), svg.attrib.get('height', '900')
    alt = html.escape(alt, quote=True)
    return f'<figure class="diagram-figure technical-figure"><a class="diagram-expand" href="assets/{name}" target="_blank" rel="noopener" title="打开完整 SVG"><img width="{width}" height="{height}" src="assets/{name}" alt="{alt}" loading="lazy"></a><figcaption>{alt}<em>点击查看完整 SVG；窄屏可横向滚动 ↗</em></figcaption></figure>'


def render_markdown(source, key):
    lines = source.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
    parts, headings, paragraph = [], [], []
    section_open = False
    injected = {}
    if key == 'speculation':
        fragment = (ROOT / 'scripts' / 'spec-player.html').read_text(encoding='utf-8')
        split = fragment.index('<section class="spec-lab" id="spec-stream"')
        injected = {'5.': fragment[:split], '7.': fragment[split:]}

    def flush():
        if paragraph:
            parts.append('<p>' + inline(' '.join(paragraph)) + '</p>')
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush(); i += 1; continue
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
                for prefix, fragment in injected.items():
                    if title.startswith(prefix):
                        parts.append(fragment)
                identifier = key + '-' + str(len(headings) + 1)
                headings.append((identifier, title))
                parts.append(f'<section class="lesson observed" id="{identifier}" data-title="{html.escape(title, quote=True)}"><h2>{inline(title)}</h2>')
                section_open = True
            else:
                parts.append('<h3>' + inline(title) + '</h3>')
            i += 1; continue
        if match := re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line):
            flush(); parts.append(figure(*match.groups())); i += 1; continue
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
        chapter_nav = '<a href="index.html">01　调度、PD 与 KV</a>' + ''.join(f'<a href="{slug}.html"' + (' class="active" aria-current="page"' if slug == key else '') + f'>{number:02d}　{html.escape(label)}</a>' for number, (slug, _, label, _) in enumerate(CHAPTERS, 2))
        if key == 'speculation':
            player_links = '<a href="#spec-player">投机验证播放器</a><a href="#spec-stream">流式指标实验室</a>'
            extra = '<link rel="stylesheet" href="spec.css"><script defer src="spec-model.js"></script><script defer src="spec-app.js"></script>'
            next_link = '<a href="execution.html">← 回看一次前向</a>'
            action = '<a class="primary-button" href="#spec-player">逐步看接受与回退 →</a><a class="text-button" href="#spec-stream">对照 ITL 与 TPOT ↓</a>'
        else:
            player_links, extra = '', ''
            next_link = '<a href="speculation.html">继续：投机解码与流式指标 →</a>'
            action = '<a class="primary-button" href="#execution-1">从输入打包开始 →</a><a class="text-button" href="speculation.html">继续看投机解码 ↗</a>'
        page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{html.escape(title)}：以图解和构造例学习 LLM 推理机制。"><title>{html.escape(title)} · Inference Wiki</title><link rel="icon" href="assets/favicon.svg"><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="reading.css"><link rel="stylesheet" href="chapters.css"><script defer src="chapter.js"></script>{extra}</head>
<body><div class="reading-progress" id="readingProgress"></div>
<aside class="sidebar" id="sidebar"><a class="brand" href="index.html"><span class="brand-mark"><i></i><i></i><i></i></span><span>Inference<span class="brand-sub">图解推理系统</span></span><b>WIKI</b></a><button class="search-trigger" data-search>搜索本章 <kbd>⌘ K</kbd></button><div class="nav-group-label">学习篇章</div><nav class="main-nav" aria-label="学习篇章">{chapter_nav}</nav><div class="nav-group-label resources-label">本章目录</div><nav class="main-nav chapter-nav" aria-label="本章目录">{player_links}{local_nav}</nav><div class="sidebar-bottom">构造例与实际性能分开说明</div></aside>
<div class="page-shell"><header class="topbar"><button class="mobile-menu" id="menuToggle" aria-label="展开目录" aria-expanded="false">☰</button><div class="breadcrumb"><a href="index.html">推理系统</a><span>/</span><strong>{html.escape(title)}</strong></div><a class="source-link" href="{key}.md" download>下载本章 Markdown ↗</a></header>
<main><header class="chapter-hero"><div class="eyebrow">INFERENCE SYSTEMS / 延伸阅读</div><h1>{display_title}</h1><div class="hero-actions">{action}</div></header><div class="article-layout"><article class="chapter-article">{body}<nav class="chapter-next" aria-label="继续阅读"><a href="index.html">← 调度与 PD 总览</a>{next_link}</nav><a class="download-source" href="technical-wiki-source.zip" download>下载全部 Markdown 与可编辑 SVG 图源 ↓</a><footer><p>教学构造例用于解释机制；性能判断以目标负载下的实际测量为依据。</p></footer></article><aside class="page-toc"><span>本章导读</span>{player_links}{local_nav}</aside></div></main></div>
<dialog id="searchDialog"><div class="search-dialog-input"><input id="searchInput" type="search" aria-label="搜索本章知识点" placeholder="搜索本章内容"><button id="closeSearch" aria-label="关闭搜索">Esc</button></div><div id="searchResults" class="search-results"></div></dialog></body></html>'''
        (DIST / (key + '.html')).write_text(page, encoding='utf-8')
    print('已从 Markdown 生成 2 篇图解网页。')


if __name__ == '__main__':
    build_chapters()
