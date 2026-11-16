#!/usr/bin/env python3
"""Fetch the exact Apache-2.0 vLLM source files listed in sources.json."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
manifest_path = Path(__file__).with_name("sources.json")
manifest = json.loads(manifest_path.read_text())
a.output.mkdir(parents=True, exist_ok=True)
for item in manifest["sources"]:
    target = a.output / item["file"]
    if target.exists():
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise SystemExit(f"Refusing to replace changed source: {target}")
    else:
        with urllib.request.urlopen(item["url"], timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise SystemExit(f"Source hash mismatch: {item['file']}")
        temp = target.with_suffix(target.suffix + ".download")
        temp.write_bytes(data)
        temp.replace(target)
    print(item["file"], "verified")
(a.output / "sources.json").write_bytes(manifest_path.read_bytes())
