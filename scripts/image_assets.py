"""Validate local Wiki figures and read intrinsic dimensions without dependencies."""
from pathlib import Path
import math
import re
import struct
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
import zlib

IMAGE_SUFFIXES = {'.svg', '.png'}
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
PAGE_MANUSCRIPTS = {
    'index.html': 'LLM推理调度与执行.md',
    'execution.html': '从一轮调度到一次前向.md',
    'speculation.html': '投机解码与流式指标.md',
}
SOURCE_MANIFEST = 'profile-sources.json'


def rewrite_local_link(reference, *, bundle=False):
    """Rewrite only known generated-page links and the source manifest."""
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or parsed.query:
        return reference
    if parsed.path.startswith('../dist/'):
        page = parsed.path[len('../dist/'):]
        if page in PAGE_MANUSCRIPTS:
            if bundle:
                # Generated section IDs differ from Markdown heading anchors.
                return PAGE_MANUSCRIPTS[page]
            return page + ('#' + parsed.fragment if parsed.fragment else '')
    if parsed.path == '../dist/assets/' + SOURCE_MANIFEST and not parsed.fragment:
        return ('figures/' if bundle else 'assets/') + SOURCE_MANIFEST
    return reference


def source_manifest(assets):
    """Return the optional, known manifest; reject links outside assets."""
    path = Path(assets) / SOURCE_MANIFEST
    if path.exists():
        if path.resolve().parent != Path(assets).resolve() or not path.is_file():
            raise ValueError(f'素材来源清单应保存在 dist/assets/ 顶层：{path.name}')
        return path
    return None


def resolve_asset(reference, base, assets):
    """Only allow an existing, top-level SVG or PNG inside the assets directory."""
    parsed = urlsplit(reference)
    if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
            or Path(reference).is_absolute() or '\\' in reference):
        raise ValueError(f'正文图源应为仓库内的本地图源：{reference}')
    path = (Path(base) / reference).resolve()
    if path.parent != Path(assets).resolve() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f'正文图源应为 dist/assets/ 顶层的 SVG 或 PNG：{reference}')
    if not path.is_file():
        raise FileNotFoundError(f'找不到正文图源：{path}')
    return path


def image_dimensions(path):
    """Return numeric width/height; validate the PNG signature and first IHDR."""
    path = Path(path)
    if path.suffix.lower() == '.png':
        with path.open('rb') as image:
            header = image.read(33)
        if (len(header) != 33 or header[:8] != PNG_SIGNATURE
                or header[8:12] != struct.pack('>I', 13) or header[12:16] != b'IHDR'):
            raise ValueError(f'PNG 签名或 IHDR 无效：{path.name}')
        width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', header[16:29])
        depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
        if (not 0 < width < 2**31 or not 0 < height < 2**31
                or depth not in depths.get(color, set()) or compression != 0
                or filtering != 0 or interlace not in (0, 1)
                or zlib.crc32(header[12:29]) != struct.unpack('>I', header[29:33])[0]):
            raise ValueError(f'PNG IHDR 字段或校验和无效：{path.name}')
        return width, height
    if path.suffix.lower() != '.svg':
        raise ValueError(f'不支持的图源格式：{path.name}')
    svg = ET.parse(path).getroot()
    if svg.tag.rsplit('}', 1)[-1] != 'svg':
        raise ValueError(f'SVG 根元素无效：{path.name}')

    def length(value):
        if not re.fullmatch(r'\+?(?:\d+(?:\.\d*)?|\.\d+)(?:px)?', value):
            return None
        number = float(value.removesuffix('px'))
        return number if math.isfinite(number) and number > 0 else None

    width, height = length(svg.get('width', '')), length(svg.get('height', ''))
    if width is None or height is None:
        try:
            viewbox = [float(value) for value in re.split(r'[\s,]+', svg.get('viewBox', '').strip())]
        except ValueError:
            viewbox = []
        if len(viewbox) != 4 or not all(math.isfinite(value) for value in viewbox) or min(viewbox[2:]) <= 0:
            raise ValueError(f'SVG 缺少有效尺寸或 viewBox：{path.name}')
        width, height = viewbox[2:]
    return width, height
