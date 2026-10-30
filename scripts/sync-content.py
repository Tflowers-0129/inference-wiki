#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步仓库内的正文，并生成网页提供下载的 Markdown 与 SVG 源文件包。"""

import argparse
from pathlib import Path
import re
import runpy
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / 'content' / 'LLM推理调度与执行.md'
CHAPTER_DOWNLOADS = {
    'LLM推理调度与执行.md': 'source.md',
    '从一轮调度到一次前向.md': 'execution.md',
    '投机解码与流式指标.md': 'speculation.md',
}


def sync_content(manuscript=DEFAULT_MANUSCRIPT):
    """Resolve manuscript images locally; keep generated downloads portable."""
    manuscript = Path(manuscript).resolve()
    bundle_chapters, all_diagrams = {}, set()
    for name, output in CHAPTER_DOWNLOADS.items():
        chapter = manuscript if name == DEFAULT_MANUSCRIPT.name else ROOT / 'content' / name
        markdown = chapter.read_text(encoding='utf-8')
        web_markdown = bundle_markdown = markdown
        for reference in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', markdown):
            if '://' in reference or reference.startswith('data:'):
                raise ValueError(f'正文图源应为仓库内的本地 SVG：{reference}')
            image_path = (chapter.parent / reference).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f'找不到正文图源：{image_path}')
            expected = ROOT / 'dist' / 'assets' / image_path.name
            if image_path != expected.resolve() or image_path.suffix.lower() != '.svg':
                raise ValueError(f'正文图源应保存在 dist/assets/：{reference}')
            web_markdown = web_markdown.replace(f']({reference})', f'](assets/{image_path.name})')
            bundle_markdown = bundle_markdown.replace(f']({reference})', f'](figures/{image_path.name})')
            all_diagrams.add(image_path)
        for chapter_name, download_name in CHAPTER_DOWNLOADS.items():
            web_markdown = web_markdown.replace(f']({chapter_name})', f']({download_name})')
        (ROOT / 'dist' / output).write_text(web_markdown, encoding='utf-8')
        bundle_chapters[name] = bundle_markdown
    bundle_readme = '''# LLM 推理调度与执行：可编辑图文源文件

依次阅读 `LLM推理调度与执行.md`、`从一轮调度到一次前向.md`、`投机解码与流式指标.md`。
解压时保留 `figures/` 目录。
`figures/*.svg` 是自包含的矢量图源，中文采用系统字体。

图中的 token 算术、时间线与资源公式均附前提；示意时长和 100 ms 停顿不代表设备实测。
机制引用及待验证清单在正文中。复现实验前请记录引擎、Ascend 后端、连接器和压测工具的有效配置。
'''
    with zipfile.ZipFile(ROOT / 'dist' / 'technical-wiki-source.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, body in bundle_chapters.items():
            bundle.writestr(name, body)
        bundle.writestr('README.md', bundle_readme)
        for image_path in sorted(all_diagrams):
            bundle.write(image_path, f'figures/{image_path.name}')
    runpy.run_path(str(ROOT / 'scripts' / 'build-chapters.py'))['build_chapters']()
    print(f'已同步 {len(bundle_chapters)} 篇正文和 {len(all_diagrams)} 张 SVG，网页下载文件已生成。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, default=DEFAULT_MANUSCRIPT,
                        help='正文路径；默认使用本仓库 content/LLM推理调度与执行.md')
    sync_content(parser.parse_args().manuscript)
