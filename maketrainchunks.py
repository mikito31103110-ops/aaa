#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_BOOK_IDS_DEFAULT = [
    "100241706",
    "100249371",
    "100249376",
    "100249416",
    "100249476",
    "200004107",
    "200004148",
    "200005598",
    "200005798",
    "200006663",
    "200008003",
    "200008316",
    "200014685",
    "200015779",
    "200015843",
]

def normalize_text(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())

def chunk_text(s: str, chunk_len: int, stride: int) -> list[tuple[int, str]]:
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    n = len(s)
    if n == 0:
        return []

    if n <= chunk_len:
        return [(0, s)]

    out: list[tuple[int, str]] = []
    for off in range(0, max(1, n - chunk_len + 1), stride):
        out.append((off, s[off:off + chunk_len]))

    last_off = n - chunk_len
    if out and out[-1][0] != last_off:
        out.append((last_off, s[last_off:last_off + chunk_len]))

    return out

def iter_txt_files_for_books(gt_root: Path, book_ids: list[str]) -> list[Path]:
    files: list[Path] = []
    for bid in book_ids:
        d = gt_root / bid
        if not d.exists():
            print(f"[WARN] missing dir: {d}")
            continue
        files.extend([p for p in d.rglob("*.txt") if p.is_file()])
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True, help="gt_pages root dir (contains book_id subdirs)")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--chunk-len", type=int, default=300)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--min-len", type=int, default=80)
    ap.add_argument("--book-ids", default="", help="comma-separated book ids. empty => use built-in TRAIN_BOOK_IDS_DEFAULT")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    out_path = Path(args.out)

    if not gt_root.exists():
        raise FileNotFoundError(gt_root)

    if args.book_ids.strip():
        book_ids = [x.strip() for x in args.book_ids.split(",") if x.strip()]
    else:
        book_ids = TRAIN_BOOK_IDS_DEFAULT

    files = iter_txt_files_for_books(gt_root, book_ids)
    if not files:
        raise SystemExit("[ERROR] no txt files found. check gt-root and book dirs.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_chunks = 0
    n_skipped_empty = 0
    n_skipped_short = 0

    with out_path.open("w", encoding="utf-8") as f:
        for p in files:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            text = normalize_text(raw)

            if len(text) == 0:
                n_skipped_empty += 1
                continue

            chunks = chunk_text(text, int(args.chunk_len), int(args.stride))

            # sourceは "bookid/filename" にしておくと後で追跡しやすい
            try:
                rel = p.relative_to(gt_root)
                source = str(rel)
            except Exception:
                source = p.name

            for j, (off, ch) in enumerate(chunks):
                if len(ch) < int(args.min_len):
                    n_skipped_short += 1
                    continue

                rec = {
                    "id": f"{source}#{j:04d}",
                    "source": source,
                    "offset": int(off),
                    "text": ch,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1

            n_files += 1

    print("=== DONE ===")
    print(f"gt_root         : {gt_root}")
    print(f"books           : {len(book_ids)}")
    print(f"files_used      : {n_files}/{len(files)}")
    print(f"chunks_written  : {n_chunks}")
    print(f"skipped_empty   : {n_skipped_empty}")
    print(f"skipped_short   : {n_skipped_short}")
    print(f"chunk_len/stride: {args.chunk_len}/{args.stride}")
    print(f"out             : {out_path}")

if __name__ == "__main__":
    main()

"""

python maketrainchunks.py \
  --gt-root /home/ihpc/Documents/saito/KODAI2/gt_pages \
  --out /home/ihpc/Documents/saito/KODAI2/rag/train_chunks_train15.jsonl \
  --chunk-len 300 \
  --stride 100 \
  --min-len 80

"""