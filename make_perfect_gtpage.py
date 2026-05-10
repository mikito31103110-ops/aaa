#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_gt_pages_from_charid.py

CODHくずし字座標CSVから、Char ID順にUnicodeを並べてGTテキストを作成する。

想定CSV列:
Unicode,Image,X,Y,Block ID,Char ID,Width,Height

出力:
1. image単位GT:
   out_root/by_image/<book_id>/<image_id>.txt
   例: by_image/100249376/100249376_00004_1.txt

2. 見開き/丁単位GT:
   out_root/by_page/<book_id>/<book_id>_00004.txt
   例: by_page/100249376/100249376_00004.txt
   ※ 100249376_00004_1 と 100249376_00004_2 を統合

3. 文献全体GT:
   out_root/by_book/<book_id>.txt

4. ログ:
   out_root/report.csv

使い方例:
python make_gt_pages_from_charid.py \
  --csv-root /home/ihpc-9/Documents/saito/KODAI/full \
  --out-root ./gt_from_charid

またはCSVだけを集めたフォルダなら:
python make_gt_pages_from_charid.py \
  --csv-root ./coordinate_csvs \
  --out-root ./gt_from_charid
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


RE_CHAR_ID_NUM = re.compile(r"(\d+)")
RE_BOOK_ID = re.compile(r"^(\d+)_")
RE_PAGE_BASE = re.compile(r"^(.+_\d+)(?:_[12])?$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv-root",
        type=Path,
        required=True,
        help="座標CSVを探すルートディレクトリ。配下を再帰的に検索する。",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="GTテキスト出力先。",
    )
    ap.add_argument(
        "--csv-pattern",
        type=str,
        default="*.csv",
        help="検索するCSVパターン。default: *.csv",
    )
    ap.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="CSV読み込み文字コード。default: utf-8",
    )
    ap.add_argument(
        "--normalize-newline",
        action="store_true",
        help="文献全体出力でページ間に空行を入れる。",
    )
    ap.add_argument(
        "--skip-empty-unicode",
        action="store_true",
        help="Unicode列が空の行をスキップする。",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="欠損列や不正Unicodeがあれば即停止する。",
    )
    return ap.parse_args()


def find_csvs(csv_root: Path, pattern: str) -> List[Path]:
    if not csv_root.exists():
        raise FileNotFoundError(f"csv-root not found: {csv_root}")

    csvs = sorted(csv_root.rglob(pattern))

    # coordinate.csvっぽいものを優先
    coord_like = [
        p for p in csvs
        if "coordinate" in p.name.lower() or "coord" in p.name.lower()
    ]

    return coord_like if coord_like else csvs


def book_id_from_path_or_image(csv_path: Path, image_name: Optional[str]) -> str:
    if image_name:
        m = RE_BOOK_ID.match(image_name)
        if m:
            return m.group(1)

    # ファイル名や親ディレクトリから数字IDを拾う
    candidates = [csv_path.stem] + [p.name for p in csv_path.parents]
    for c in candidates:
        m = re.search(r"\d{6,}", c)
        if m:
            return m.group(0)

    return csv_path.stem


def char_id_to_int(char_id: str) -> int:
    """
    C0001 -> 1
    C12   -> 12
    """
    if char_id is None:
        raise ValueError("Char ID is None")

    m = RE_CHAR_ID_NUM.search(str(char_id))
    if not m:
        raise ValueError(f"Invalid Char ID: {char_id}")

    return int(m.group(1))


def unicode_to_char(u: str) -> str:
    """
    U+53E4 -> 古
    複数コードポイントが空白区切りで来ても一応対応:
    U+XXXX U+YYYY
    """
    if u is None:
        raise ValueError("Unicode is None")

    u = str(u).strip()
    if not u:
        raise ValueError("Unicode is empty")

    parts = re.split(r"\s+", u)
    chars = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("U+"):
            hex_part = part[2:]
        elif part.startswith("u+"):
            hex_part = part[2:]
        else:
            # すでに実文字で入っている場合に備える
            if len(part) == 1:
                chars.append(part)
                continue
            raise ValueError(f"Invalid Unicode format: {u}")

        codepoint = int(hex_part, 16)
        chars.append(chr(codepoint))

    return "".join(chars)


def page_base_from_image(image: str) -> str:
    """
    100249376_00004_1 -> 100249376_00004
    100249376_00004_2 -> 100249376_00004
    100249376_00004   -> 100249376_00004
    """
    m = RE_PAGE_BASE.match(image)
    if not m:
        return image
    return m.group(1)


def side_order_from_image(image: str) -> int:
    """
    *_1 を先、*_2 を後にする。
    末尾が取れない場合は 0。
    """
    if image.endswith("_1"):
        return 1
    if image.endswith("_2"):
        return 2
    return 0


def natural_image_key(image: str) -> Tuple:
    """
    画像名の自然順ソート用。
    例:
    100249376_00004_1
    100249376_00004_2
    100249376_00005_1
    """
    parts = re.split(r"(\d+)", image)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


def read_coordinate_csv(
    csv_path: Path,
    encoding: str,
    skip_empty_unicode: bool,
    strict: bool,
) -> Tuple[str, Dict[str, List[Tuple[int, str, str]]], List[str]]:
    """
    return:
      book_id
      image_to_items: image -> [(char_id_num, char, raw_unicode), ...]
      warnings
    """
    warnings = []
    image_to_items: Dict[str, List[Tuple[int, str, str]]] = {}

    required_cols = {"Unicode", "Image", "Char ID"}

    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"No header found: {csv_path}")

        missing = required_cols - set(reader.fieldnames)
        if missing:
            msg = f"{csv_path}: missing columns: {sorted(missing)}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            return csv_path.stem, image_to_items, warnings

        first_image = None

        for line_no, row in enumerate(reader, start=2):
            image = (row.get("Image") or "").strip()
            raw_u = (row.get("Unicode") or "").strip()
            char_id = (row.get("Char ID") or "").strip()

            if not image:
                msg = f"{csv_path}:{line_no}: empty Image"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue

            if first_image is None:
                first_image = image

            if not raw_u:
                msg = f"{csv_path}:{line_no}: empty Unicode"
                if skip_empty_unicode:
                    warnings.append(msg)
                    continue
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue

            try:
                cid_num = char_id_to_int(char_id)
            except Exception as e:
                msg = f"{csv_path}:{line_no}: invalid Char ID {char_id!r}: {e}"
                if strict:
                    raise
                warnings.append(msg)
                continue

            try:
                ch = unicode_to_char(raw_u)
            except Exception as e:
                msg = f"{csv_path}:{line_no}: invalid Unicode {raw_u!r}: {e}"
                if strict:
                    raise
                warnings.append(msg)
                continue

            image_to_items.setdefault(image, []).append((cid_num, ch, raw_u))

    book_id = book_id_from_path_or_image(csv_path, first_image)
    return book_id, image_to_items, warnings


def check_char_id_sequence(items: List[Tuple[int, str, str]]) -> List[str]:
    """
    Char IDの重複・欠番をチェック。
    """
    warnings = []
    ids = [x[0] for x in items]

    if not ids:
        return warnings

    seen = set()
    dup = sorted({x for x in ids if x in seen or seen.add(x)})
    if dup:
        warnings.append(f"duplicate Char ID numbers: {dup[:20]}")

    expected = list(range(1, max(ids) + 1))
    missing = sorted(set(expected) - set(ids))
    if missing:
        warnings.append(f"missing Char ID numbers: {missing[:20]}")

    return warnings


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def process_one_csv(
    csv_path: Path,
    out_root: Path,
    encoding: str,
    skip_empty_unicode: bool,
    strict: bool,
    normalize_newline: bool,
) -> List[Dict[str, str]]:
    report_rows: List[Dict[str, str]] = []

    book_id, image_to_items, warnings = read_coordinate_csv(
        csv_path=csv_path,
        encoding=encoding,
        skip_empty_unicode=skip_empty_unicode,
        strict=strict,
    )

    if not image_to_items:
        report_rows.append({
            "csv": str(csv_path),
            "book_id": book_id,
            "unit": "csv",
            "name": "",
            "chars": "0",
            "status": "no_data",
            "warnings": " | ".join(warnings),
        })
        return report_rows

    by_image_dir = out_root / "by_image" / book_id
    by_page_dir = out_root / "by_page" / book_id
    by_book_dir = out_root / "by_book"

    image_texts: Dict[str, str] = {}

    # 1. Image単位GT
    for image in sorted(image_to_items.keys(), key=natural_image_key):
        items = image_to_items[image]
        seq_warnings = check_char_id_sequence(items)

        # Char ID順。同じChar IDがあれば元順をなるべく保つため stable sort
        items_sorted = sorted(items, key=lambda x: x[0])
        text = "".join(ch for _, ch, _ in items_sorted)
        image_texts[image] = text

        out_path = by_image_dir / f"{image}.txt"
        write_text(out_path, text)

        report_rows.append({
            "csv": str(csv_path),
            "book_id": book_id,
            "unit": "image",
            "name": image,
            "chars": str(len(text)),
            "status": "ok",
            "warnings": " | ".join(seq_warnings),
        })

    # 2. ページ単位GT: *_1 と *_2 を統合
    page_to_images: Dict[str, List[str]] = {}
    for image in image_texts.keys():
        page_base = page_base_from_image(image)
        page_to_images.setdefault(page_base, []).append(image)

    page_texts: Dict[str, str] = {}

    for page_base in sorted(page_to_images.keys(), key=natural_image_key):
        images = sorted(
            page_to_images[page_base],
            key=lambda x: (natural_image_key(page_base_from_image(x)), side_order_from_image(x), natural_image_key(x)),
        )

        # 片面ごとは改行で分ける。LM学習用に完全連結したければ "\n" を "" に変える。
        page_text = "\n".join(image_texts[img] for img in images)
        page_texts[page_base] = page_text

        out_path = by_page_dir / f"{page_base}.txt"
        write_text(out_path, page_text)

        report_rows.append({
            "csv": str(csv_path),
            "book_id": book_id,
            "unit": "page",
            "name": page_base,
            "chars": str(len(page_text.replace('\n', ''))),
            "status": "ok",
            "warnings": f"merged_images={','.join(images)}",
        })

    # 3. 文献全体GT
    sep = "\n\n" if normalize_newline else "\n"
    book_text = sep.join(page_texts[k] for k in sorted(page_texts.keys(), key=natural_image_key))
    write_text(by_book_dir / f"{book_id}.txt", book_text)

    report_rows.append({
        "csv": str(csv_path),
        "book_id": book_id,
        "unit": "book",
        "name": book_id,
        "chars": str(len(book_text.replace('\n', ''))),
        "status": "ok",
        "warnings": " | ".join(warnings),
    })

    return report_rows


def write_report(out_root: Path, rows: List[Dict[str, str]]):
    report_path = out_root / "report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["csv", "book_id", "unit", "name", "chars", "status", "warnings"]

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()

    csvs = find_csvs(args.csv_root, args.csv_pattern)

    if not csvs:
        print(f"[ERROR] No CSV files found under: {args.csv_root}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] found CSV files: {len(csvs)}")

    all_report_rows: List[Dict[str, str]] = []

    for i, csv_path in enumerate(csvs, start=1):
        print(f"[INFO] ({i}/{len(csvs)}) processing: {csv_path}")

        try:
            rows = process_one_csv(
                csv_path=csv_path,
                out_root=args.out_root,
                encoding=args.encoding,
                skip_empty_unicode=args.skip_empty_unicode,
                strict=args.strict,
                normalize_newline=args.normalize_newline,
            )
            all_report_rows.extend(rows)

        except Exception as e:
            print(f"[ERROR] failed: {csv_path}: {e}", file=sys.stderr)
            all_report_rows.append({
                "csv": str(csv_path),
                "book_id": "",
                "unit": "csv",
                "name": "",
                "chars": "0",
                "status": "error",
                "warnings": str(e),
            })

            if args.strict:
                write_report(args.out_root, all_report_rows)
                raise

    write_report(args.out_root, all_report_rows)

    print("[DONE]")
    print(f"[OUT] {args.out_root}")
    print(f"[REPORT] {args.out_root / 'report.csv'}")


if __name__ == "__main__":
    main()



"""
python make_perfect_gtpage.py \
  --csv-root /home/ihpc-9/Documents/saito/KODAI/full \
  --out-root ./gt_perfect_pages
"""