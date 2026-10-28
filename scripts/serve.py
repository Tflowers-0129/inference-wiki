#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步图文并启动本地 Wiki 预览；只使用 Python 标准库。"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=4173, help='本地端口（默认：4173）')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port 必须在 1 到 65535 之间')

    runpy.run_path(str(ROOT / 'scripts' / 'sync-content.py'))['sync_content']()
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / 'dist'))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    except OSError as error:
        parser.exit(1, f'无法使用端口 {args.port}：{error}\n可尝试 --port 4174。\n')
    print(f'Wiki 预览：http://127.0.0.1:{args.port}/', flush=True)
    print('按 Ctrl+C 停止；修改正文后重新运行此命令以同步下载文件。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n预览已停止。')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
