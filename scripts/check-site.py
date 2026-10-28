#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查静态站点的本地资源、页内锚点、图文下载和播放器 DOM 引用。"""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = []
        self.references = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        for attribute in ('src', 'href', 'poster'):
            if attrs.get(attribute):
                self.references.append(attrs[attribute])


def main():
    errors = []
    page = Page((DIST / 'index.html').read_text(encoding='utf-8'))
    ids = set(page.ids)
    for identifier, count in Counter(page.ids).items():
        if count > 1:
            errors.append(f'重复的 HTML id：{identifier}')

    def check_reference(reference, base, available_ids=None):
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc:
            return
        if parsed.path.startswith('/'):
            errors.append(f'应使用兼容 GitHub Pages 子路径的相对资源路径：{reference}')
            return
        target = (base / unquote(parsed.path)).resolve() if parsed.path else base
        if parsed.path and not target.is_file():
            errors.append(f'缺少本地资源：{reference}')
        if parsed.fragment and not parsed.path and available_ids is not None:
            if unquote(parsed.fragment) not in available_ids:
                errors.append(f'缺少页内锚点：#{parsed.fragment}')

    for reference in page.references:
        check_reference(reference, DIST, ids)
    for css in DIST.glob('*.css'):
        for reference in re.findall(r'url\(\s*[\"\']?([^\"\')]+)[\"\']?\s*\)', css.read_text(encoding='utf-8')):
            check_reference(reference, css.parent)
    for js in (DIST / 'app.js', DIST / 'pd-app.js'):
        for identifier in re.findall(r"\$\(['\"]([^'\"]+)['\"]\)", js.read_text(encoding='utf-8')):
            if identifier not in ids:
                errors.append(f'{js.name} 引用了不存在的 DOM id：{identifier}')
    for anchor in ('scheduler', 'pdPlayer'):
        if anchor not in ids:
            errors.append(f'缺少演示入口：#{anchor}')

    for manuscript in (ROOT / 'content' / '混部调度与稳态.md', DIST / 'source.md'):
        if not manuscript.is_file():
            errors.append(f'缺少正文文件：{manuscript.relative_to(ROOT)}；请先运行 sync-content.py')
            continue
        images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', manuscript.read_text(encoding='utf-8'))
        if len(set(images)) != 5:
            errors.append(f'{manuscript.name} 应包含 5 张独立 SVG 图源')
        for reference in images:
            check_reference(reference, manuscript.parent)

    bundle_path = DIST / 'technical-wiki-source.zip'
    if not bundle_path.is_file():
        errors.append('缺少图文源包；请先运行 sync-content.py')
    else:
        with zipfile.ZipFile(bundle_path) as bundle:
            if bundle.testzip():
                errors.append('图文源包校验失败')
            body = bundle.read('混部调度与稳态.md').decode('utf-8')
            for reference in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', body):
                if not reference.startswith('figures/') or reference not in bundle.namelist():
                    errors.append(f'图文源包内图源路径无效：{reference}')

    if errors:
        print('\n'.join(f'错误：{error}' for error in errors), file=sys.stderr)
        return 1
    print(f'站点检查通过：{len(page.references)} 个资源/链接引用、{len(ids)} 个 DOM 锚点、5 张图源和 2 个下载文件。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
