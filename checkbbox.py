#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BBOXとGTが一致しているかの確認
"""

from pathlib import Path
import pandas as pd
import numpy as np

from xycut import recursive_xy_cut


# ===== 設定 =====
COORD_CSV = Path("../KODAI/full/100249537/100249537_coordinate.csv")
GT_DIR = Path("./gt_pages/100249537")  # KODAI2/gt_pages
REMOVE_WHITESPACE = True    # Trueで改行・空白除去

# 追加：数えたい文字（必要なら変更）
COUNT_CHAR = "?"


# === 追加：不可視系文字の除去（空白判定のため） ===
# よく混入する「空白っぽいのにlen>0」犯人たち
INVISIBLE_CHARS = {
    "\ufeff",  # BOM
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\u00ad",  # SOFT HYPHEN
}

def strip_invisible_and_space(txt: str) -> str:
    # 空白(Unicode whitespace) + 制御文字 + 上の不可視を除去して「実質中身」を得る
    out = []
    for ch in txt:
        if ch in INVISIBLE_CHARS:
            continue
        # issapce(): 改行/タブ/全角スペース等も含む
        if ch.isspace():
            continue
        # 制御文字（カテゴリCc）をざっくり排除：ord<32 と DEL(127)
        o = ord(ch)
        if o < 32 or o == 127:
            continue
        out.append(ch)
    return "".join(out)


def normalize_text(txt: str) -> str:
    if REMOVE_WHITESPACE:
        return "".join(ch for ch in txt if not ch.isspace())
    return txt


def count_bboxes_xycut(df_page: pd.DataFrame) -> int:
    x1 = df_page["X"].to_numpy()
    y1 = df_page["Y"].to_numpy()
    x2 = x1 + df_page["Width"].to_numpy()
    y2 = y1 + df_page["Height"].to_numpy()

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)

    res = []
    recursive_xy_cut(boxes, np.arange(len(boxes)), res)

    return len(res)


def count_target_char(txt: str, target: str = COUNT_CHAR) -> int:
    return txt.count(target)


def main():
    df = pd.read_csv(COORD_CSV)
    pages = sorted(df["Image"].astype(str).unique())

    print("===== BBOX vs GT LENGTH CHECK =====")
    print(f"Total pages in CSV : {len(pages)}")
    print(f"GT directory       : {GT_DIR.resolve()}")
    print(f"Count target char  : {COUNT_CHAR!r}")
    print()

    ok_pages = []
    ng_pages = []

    q_pages = []
    total_q = 0

    # === 作り直し：空白ページ検出（堅牢版） ===
    blank_pages = []  # (page, bbox_n, reason)

    for page in pages:
        gt_path = GT_DIR / f"{page}.txt"
        if not gt_path.exists():
            print(f"[SKIP] {page} : GT txt not found")
            continue

        df_page = df[df["Image"] == page]
        bbox_n = count_bboxes_xycut(df_page)

        # まず「ファイルサイズ0」を直撃で検出
        try:
            size = gt_path.stat().st_size
        except OSError:
            size = -1

        gt_text_raw = gt_path.read_text(encoding="utf-8", errors="replace")

        # 実質空判定用（不可視・空白・制御を除去）
        gt_effective = strip_invisible_and_space(gt_text_raw)
        is_blank = (size == 0) or (len(gt_effective) == 0)

        # 既存の評価用テキスト（これまで通り）
        gt_text = normalize_text(gt_text_raw)
        gt_n = len(gt_text)

        if is_blank:
            reason = "size==0" if size == 0 else "only_whitespace/invisible"
            blank_pages.append((page, bbox_n, reason))

        q_n = count_target_char(gt_text, COUNT_CHAR)
        if q_n > 0:
            q_pages.append((page, q_n))
            total_q += q_n

        if bbox_n == gt_n:
            ok_pages.append(page)
            status = "OK"
        else:
            ng_pages.append((page, bbox_n, gt_n))
            status = "NG"

        # BLANKフラグも表示（大枠は維持しつつ情報追加）
        blank_flag = " BLANK" if is_blank else ""
        print(f"[{status}]{blank_flag} {page} : bbox={bbox_n}, gt_chars={gt_n}, {COUNT_CHAR}={q_n}")

    print("\n===== SUMMARY =====")
    print(f"Matched pages : {len(ok_pages)}")
    print(f"Mismatched    : {len(ng_pages)}")

    print(f"Pages with {COUNT_CHAR!r} : {len(q_pages)}")
    print(f"Total {COUNT_CHAR!r}      : {total_q}")

    print(f"Blank GT pages : {len(blank_pages)}")

    if blank_pages:
        print("\n--- BLANK PAGE DETAIL ---")
        for p, b, reason in blank_pages:
            print(f"{p} : bbox={b}, reason={reason}")

    if q_pages:
        print(f"\n--- {COUNT_CHAR!r} DETAIL ---")
        for p, qn in q_pages:
            print(f"{p} : {COUNT_CHAR}={qn}")

    if ng_pages:
        print("\n--- MISMATCH DETAIL ---")
        for p, b, g in ng_pages:
            print(f"{p} : bbox={b}, gt={g}, diff={b-g}")


if __name__ == "__main__":
    main()
