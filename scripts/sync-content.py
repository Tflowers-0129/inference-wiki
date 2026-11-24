#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步仓库内的正文，并生成网页提供下载的 Markdown 与图源文件包。"""

import argparse
from pathlib import Path
import re
import runpy
import zipfile
from image_assets import SOURCE_MANIFEST, image_dimensions, resolve_asset, rewrite_local_link, source_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / 'content' / 'LLM推理调度与执行.md'
CHAPTER_DOWNLOADS = {
    'LLM推理调度与执行.md': 'source.md',
    '从一轮调度到一次前向.md': 'execution.md',
    '模型层中的Attention与MoE.md': 'model-layers.md',
    '投机解码与流式指标.md': 'speculation.md',
    '负载均衡与专家并行.md': 'load-balancing.md',
    'EP通信与通算融合.md': 'ep-communication.md',
    '上下文并行与Attention归并.md': 'context-parallel.md',
    'PD分离系统与KV生命周期.md': 'pd-disaggregation.md',
    'KV缓存布局与递归状态.md': 'kv-cache-layouts.md',
    '稀疏KV卸载与HiSparse.md': 'sparse-offload.md',
    'KV池化与跨机PD.md': 'kv-pooling.md',
    '量化推理与算子执行.md': 'quantization.md',
    '微批次流水与DBO.md': 'dbo.md',
}


def sync_content(manuscript=DEFAULT_MANUSCRIPT):
    """Resolve manuscript images locally; keep generated downloads portable."""
    manuscript = Path(manuscript).resolve()
    bundle_chapters, all_diagrams = {}, set()
    manifest = source_manifest(ROOT / 'dist' / 'assets')

    def rewrite_links(markdown, *, bundle=False):
        return re.sub(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)',
                      lambda match: f'[{match[1]}]({rewrite_local_link(match[2], bundle=bundle)})', markdown)

    for name, output in CHAPTER_DOWNLOADS.items():
        chapter = manuscript if name == DEFAULT_MANUSCRIPT.name else ROOT / 'content' / name
        markdown = chapter.read_text(encoding='utf-8')
        web_markdown = bundle_markdown = markdown
        for reference in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', markdown):
            image_path = resolve_asset(reference, chapter.parent, ROOT / 'dist' / 'assets')
            image_dimensions(image_path)
            web_markdown = web_markdown.replace(f']({reference})', f'](assets/{image_path.name})')
            bundle_markdown = bundle_markdown.replace(f']({reference})', f'](figures/{image_path.name})')
            all_diagrams.add(image_path)
        for chapter_name, download_name in CHAPTER_DOWNLOADS.items():
            web_markdown = web_markdown.replace(f']({chapter_name})', f']({download_name})')
        web_markdown = rewrite_links(web_markdown)
        bundle_markdown = rewrite_links(bundle_markdown, bundle=True)
        if f'](figures/{SOURCE_MANIFEST})' in bundle_markdown and manifest is None:
            raise FileNotFoundError(f'找不到素材来源清单：dist/assets/{SOURCE_MANIFEST}')
        (ROOT / 'dist' / output).write_text(web_markdown, encoding='utf-8')
        bundle_chapters[name] = bundle_markdown
    bundle_readme = '''# LLM 推理调度与执行：图文源文件

第一次读，可以先在[调度篇](LLM推理调度与执行.md)算两轮预算，再跟着[一次前向](从一轮调度到一次前向.md)和[模型的一层](模型层中的Attention与MoE.md)走完模型计算，最后看[投机解码](投机解码与流式指标.md)怎样把一组输出发到客户端。

- 想追 KV 的去向，从[缓存页的读写](KV缓存布局与递归状态.md)开始，再看 [P/D 交接](PD分离系统与KV生命周期.md)、[前缀复用](KV池化与跨机PD.md)或[稀疏 KV 的取回](稀疏KV卸载与HiSparse.md)。
- 想弄懂多卡分工，可以看[请求与专家放置](负载均衡与专家并行.md)、[专家输入怎样往返](EP通信与通算融合.md)和[上下文怎样分片与归并](上下文并行与Attention归并.md)。
- 想估算算子成本，可以先[手算量化点积](量化推理与算子执行.md)，再看[两批工作怎样重叠](微批次流水与DBO.md)。

解压时保留 `figures/` 目录。
`figures/*.svg` 是自包含的矢量图源，中文采用系统字体。
`figures/*.png` 是随正文收录的原图；图源、上下文与解释范围见对应正文。
`figures/profile-sources.json`（若有）保留素材来源清单；章节链接指向同包 Markdown。

图中的 token 算术、时间线与资源公式均附前提；示意时长和 100 ms 停顿不代表设备实测。
机制引用及待验证清单在正文中。复现实验前请记录引擎、Ascend 后端、连接器和压测工具的有效配置。
'''
    with zipfile.ZipFile(ROOT / 'dist' / 'technical-wiki-source.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, body in bundle_chapters.items():
            bundle.writestr(name, body)
        bundle.writestr('README.md', bundle_readme)
        for image_path in sorted(all_diagrams):
            bundle.write(image_path, f'figures/{image_path.name}')
        if manifest is not None:
            bundle.write(manifest, f'figures/{SOURCE_MANIFEST}')
    runpy.run_path(str(ROOT / 'scripts' / 'build-chapters.py'))['build_chapters']()
    print(f'已同步 {len(bundle_chapters)} 篇正文和 {len(all_diagrams)} 张图源，网页下载文件已生成。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, default=DEFAULT_MANUSCRIPT,
                        help='正文路径；默认使用本仓库 content/LLM推理调度与执行.md')
    sync_content(parser.parse_args().manuscript)
