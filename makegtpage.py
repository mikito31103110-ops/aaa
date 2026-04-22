#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import re
from pathlib import Path


IMAGE_TAG_RE = re.compile(r"<Image:([^>]+)>")
FILENAME_RE = re.compile(r"^(?P<page_id>\d+_\d{5}_\d)\.jpg$")  # e.g. 100241706_00004_2.jpg

# 全角カッコ（...）を削除（最短一致）
PAREN_RE = re.compile(r"（.*?）", flags=re.DOTALL)


def parse_sections(raw_text: str) -> list[tuple[str, str]]:
    """
    Returns: list of (image_filename, content_str)
    content_str is the text between this <Image:...> and the next <Image:...>.
    """
    matches = list(IMAGE_TAG_RE.finditer(raw_text))
    sections: list[tuple[str, str]] = []

    for i, m in enumerate(matches):
        img_fname = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

        content = raw_text[start:end]
        # trim only outer newlines (keep internal formatting)
        content = content.strip("\n")

        sections.append((img_fname, content))

    return sections


def normalize_content(content: str) -> str:
    """
    1) 〓 を削除
    2) 〳〵, 〴〵 を置換（幅ズレ対策）
    3) 全角カッコ（...）を丸ごと削除
    """
    if not content:
        return content

    # 1) ゲタ除去
    content = content.replace("〓", "")

    # 2) 置換（順序はこのままでOK）
    content = content.replace("〳〵", "〱")
    content = content.replace("〴〵", "〲")

    # 3) 全角カッコ（...）を削除（複数個対応）
    content = PAREN_RE.sub("", content)

    return content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="貼り付けテキストを保存したファイル")
    ap.add_argument("--out-root", type=str, required=True, help="KODAI2/gt_pages のパス")
    ap.add_argument("--book-id", type=str, required=True, help="例: 100241706")
    ap.add_argument("--overwrite", action="store_true", help="既存txtがあっても上書きする")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_root = Path(args.out_root)
    book_id = args.book_id.strip()

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8")

    # 出力ディレクトリ
    out_dir = out_root / book_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = parse_sections(raw_text)

    written = 0
    skipped = 0
    badname = 0

    for img_fname, content in sections:
        m = FILENAME_RE.match(img_fname)
        if not m:
            badname += 1
            continue

        page_id = m.group("page_id")  # e.g. 100241706_00004_2
        # book_id が違うものが混ざってたら弾く（安全策）
        if not page_id.startswith(book_id + "_"):
            badname += 1
            continue

        out_path = out_dir / f"{page_id}.txt"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        # === 正規化（追加）===
        content = normalize_content(content)
        # =====================

        # 仕様：本文が空でもファイルは作る（空ページの記録）
        # 末尾改行は1つだけ付けておく（行末なしを避ける）
        if content:
            out_path.write_text(content + "\n", encoding="utf-8")
        else:
            out_path.write_text("", encoding="utf-8")

        written += 1

    print(f"[DONE] out_dir   : {out_dir}")
    print(f"[DONE] written   : {written}")
    print(f"[DONE] skipped   : {skipped}  (use --overwrite to overwrite)")
    print(f"[WARN] badname   : {badname}  (unexpected <Image:...> filename format)")


if __name__ == "__main__":
    main()
