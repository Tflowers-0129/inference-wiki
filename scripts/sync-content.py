#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步仓库内的正文，并生成网页提供下载的 Markdown 与 SVG 源文件包。"""

import argparse
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / 'content' / '混部调度与稳态.md'


def sync_content(manuscript=DEFAULT_MANUSCRIPT):
    """Resolve manuscript images locally; keep generated downloads portable."""
    manuscript = Path(manuscript).resolve()
    markdown = manuscript.read_text(encoding='utf-8')
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', markdown)
    diagrams = {}
    for image in images:
        if '://' in image or image.startswith('data:'):
            raise ValueError(f'正文图源应为仓库内的本地 SVG：{image}')
        image_path = (manuscript.parent / image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f'找不到正文图源：{image_path}')
        expected_path = ROOT / 'dist' / 'assets' / image_path.name
        if image_path != expected_path.resolve() or image_path.suffix.lower() != '.svg':
            raise ValueError(f'正文图源应保存在 dist/assets/：{image}')
        diagrams[image] = image_path

    web_markdown = markdown
    bundle_markdown = markdown
    for reference, image_path in diagrams.items():
        web_markdown = web_markdown.replace(f']({reference})', f'](assets/{image_path.name})')
        bundle_markdown = bundle_markdown.replace(f']({reference})', f'](figures/{image_path.name})')
    (ROOT / 'dist' / 'source.md').write_text(web_markdown, encoding='utf-8')
    bundle_readme = '''# 混部调度与稳态：可编辑图文源文件

打开 `混部调度与稳态.md` 阅读，解压时保留 `figures/` 目录。
`figures/*.svg` 是自包含的矢量图源，中文采用系统字体。

图中的 token 算术、时间线与资源公式均附前提；示意时长和 100 ms 停顿不代表设备实测。
机制引用及待验证清单在正文中。复现实验前请记录引擎、Ascend 插件、连接器和压测工具版本。
'''
    with zipfile.ZipFile(ROOT / 'dist' / 'technical-wiki-source.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('混部调度与稳态.md', bundle_markdown)
        bundle.writestr('README.md', bundle_readme)
        for image_path in sorted(set(diagrams.values())):
            bundle.write(image_path, f'figures/{image_path.name}')
    print(f'已同步正文和 {len(set(diagrams.values()))} 张 SVG，网页下载文件已生成。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, default=DEFAULT_MANUSCRIPT,
                        help='正文路径；默认使用本仓库 content/混部调度与稳态.md')
    sync_content(parser.parse_args().manuscript)
