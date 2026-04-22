#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
import argparse

HEADER_RE = re.compile(r"^<Image:([^>]+)>$")  # <Image:xxxx.jpg>

def normalize_key(raw: str) -> str:
    """
    '100249371_00010_1.jpg' -> '100249371_00010_1'
    """
    s = raw.strip()
    if s.lower().endswith(".jpg"):
        s = s[:-4]
    elif s.lower().endswith(".jpeg"):
        s = s[:-5]
    elif s.lower().endswith(".png"):
        s = s[:-4]
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input combined text file (your pasted data)")
    ap.add_argument("--out-dir", default="gt_pages", help="output dir (default: ./gt_pages)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing txt")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = inp.read_text(encoding="utf-8", errors="ignore").splitlines()

    cur_key = None
    buf: list[str] = []
    wrote = 0
    skipped = 0

    def flush():
        nonlocal wrote, skipped, cur_key, buf
        if cur_key is None:
            return
        content = "\n".join([line.rstrip() for line in buf]).strip()
        out_path = out_dir / f"{cur_key}.txt"

        if not content:
            skipped += 1
            return

        if out_path.exists() and (not args.overwrite):
            # 既存は触らない
            skipped += 1
            return

        out_path.write_text(content + "\n", encoding="utf-8")
        wrote += 1

    for line in text:
        line = line.rstrip("\n")
        m = HEADER_RE.match(line.strip())
        if m:
            # 次のページに切り替え
            flush()
            cur_key = normalize_key(m.group(1))
            buf = []
        else:
            if cur_key is not None:
                buf.append(line)

    flush()

    print("===== split_gt_pages DONE =====")
    print(f"[INFO] input   : {inp}")
    print(f"[INFO] out_dir : {out_dir.resolve()}")
    print(f"[INFO] wrote   : {wrote} files")
    print(f"[INFO] skipped : {skipped} (empty or exists without --overwrite)")

if __name__ == "__main__":
    main()
