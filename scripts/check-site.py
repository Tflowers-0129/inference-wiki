#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check all Wiki pages, cross-page anchors, player bindings and portable downloads."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
DOWNLOADS = {'LLM推理调度与执行.md':'source.md', '从一轮调度到一次前向.md':'execution.md', '投机解码与流式指标.md':'speculation.md'}


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids, self.references, self.scripts = [], [], []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        for attribute in ('src', 'href', 'poster'):
            if attrs.get(attribute):
                self.references.append(attrs[attribute])
        if tag == 'script' and attrs.get('src'):
            self.scripts.append(attrs['src'])


def main():
    errors = []
    pages = {p.resolve():Page(p) for p in DIST.glob('*.html')}

    def check_reference(reference, base, current=None):
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc:
            return
        if parsed.path.startswith('/'):
            errors.append(f'资源应支持仓库子路径：{reference}')
            return
        target = (base / unquote(parsed.path)).resolve() if parsed.path else current
        if parsed.path and (not target or not target.is_file()):
            errors.append(f'缺少本地资源：{reference}')
        if parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
            errors.append(f'缺少锚点：{target.name}#{parsed.fragment}')

    for path, page in pages.items():
        for identifier, count in Counter(page.ids).items():
            if count > 1:
                errors.append(f'{path.name} 重复 id：{identifier}')
        for reference in page.references:
            check_reference(reference,DIST,path)
        for script in page.scripts:
            js = DIST / script
            if not js.is_file():
                continue
            for identifier in re.findall(r"(?:\$|byId)\(['\"]([^'\"]+)['\"]\)",js.read_text(encoding='utf-8')):
                if identifier not in page.ids:
                    errors.append(f'{path.name} 的 {script} 引用缺失 DOM：{identifier}')
    for filename, anchors in {'index.html':['scheduler','pdPlayer'],'speculation.html':['spec-player','spec-stream']}.items():
        page = pages.get((DIST/filename).resolve())
        if not page or any(anchor not in page.ids for anchor in anchors):
            errors.append(f'{filename} 缺少播放器入口')
    for css in DIST.glob('*.css'):
        for reference in re.findall(r'url\(\s*["\']?([^"\')]+)',css.read_text(encoding='utf-8')):
            check_reference(reference.strip(),css.parent)

    all_images, manuscripts = set(), {}
    for name, download in DOWNLOADS.items():
        expected = None
        for path in (ROOT/'content'/name,DIST/download):
            if not path.is_file():
                errors.append(f'缺少正文 {path.name}；请运行 sync-content.py')
                continue
            body = path.read_text(encoding='utf-8')
            if re.search(r'\bv\d+\.\d+(?:\.\d+)?|\b[0-9a-f]{40}\b',body):
                errors.append(f'{path.name} 不应展示具体产品版本或提交号')
            images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)',body)
            names = {Path(urlsplit(ref).path).name for ref in images}
            if not names:
                errors.append(f'{path.name} 没有图解')
            if expected is None:
                expected = names
                manuscripts[name] = names
            elif names != expected:
                errors.append(f'{path.name} 图源未同步')
            all_images.update(names)
            for reference in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',body)+images:
                check_reference(reference,path.parent)
    for image in all_images:
        try:
            ET.parse(DIST/'assets'/image)
        except (OSError,ET.ParseError) as error:
            errors.append(f'SVG 无效：{image}：{error}')
    for path in list(DIST.glob('*.html'))+list((DIST/'assets').glob('*.svg')):
        if re.search(r'\bv\d+\.\d+(?:\.\d+)?|\b[0-9a-f]{40}\b',path.read_text(encoding='utf-8')):
            errors.append(f'{path.name} 仍展示具体产品版本或提交号')

    bundle_path = DIST/'technical-wiki-source.zip'
    if not bundle_path.is_file():
        errors.append('缺少图文源包')
    else:
        with zipfile.ZipFile(bundle_path) as bundle:
            if bundle.testzip():
                errors.append('图文源包校验失败')
            for name, expected in manuscripts.items():
                if name not in bundle.namelist():
                    errors.append(f'源包缺少 {name}')
                    continue
                refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)',bundle.read(name).decode('utf-8'))
                if {Path(ref).name for ref in refs} != expected:
                    errors.append(f'源包 {name} 图源不一致')
                for reference in refs:
                    if not reference.startswith('figures/') or reference not in bundle.namelist():
                        errors.append(f'源包图源缺失 {reference}')
    if errors:
        print('\n'.join('错误：'+error for error in errors),file=sys.stderr)
        return 1
    print(f'检查通过：{len(pages)} 篇网页、{len(DOWNLOADS)} 篇 Markdown、{len(all_images)} 张 SVG，跨页锚点及播放器 DOM 引用有效。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
