#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BBOXとGTが一致しているかの確認

現在の想定:
- coordinate.csv の Image 列:
    200003076_00004_1
    200003076_00004_2

- GT_DIR:
    gt_pages_q/200003076/200003076_00004.txt

つまり、CSV側の片面データ(_1, _2)をページ単位に統合して、
GTの統合済みtxtと比較する。

追加機能:
- coordinate.csv 側に文字ラベル列がある場合、
  CSVラベル文字列とGT文字列を比較し、
  「GTに余分な文字」「GTに足りない文字」を表示する。
"""

from pathlib import Path
import re
from collections import Counter

import pandas as pd
import numpy as np

from xycut import recursive_xy_cut


# ===== 設定 =====
COORD_CSV = Path("../KODAI/full/200006665/200006665_coordinate.csv")
GT_DIR = Path("./gt_pages_q/200006665")
REMOVE_WHITESPACE = True

# 数えたい文字
COUNT_CHAR = "?"

# 文字差分を表示するか
SHOW_CHAR_DIFF = True

# 1ページあたり何種類まで差分文字を表示するか
CHAR_DIFF_TOPN = 50

# Image名: 200003076_00004_1 -> 200003076_00004
MERGED_PAGE_RE = re.compile(r"^(?P<merged_id>\d+_\d{5})_[12]$")


# === 不可視系文字の除去（空白判定のため） ===
INVISIBLE_CHARS = {
    "\ufeff",  # BOM
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\u00ad",  # SOFT HYPHEN
}


# CSV側の文字ラベル列候補
# 実際のcoordinate.csvの列名が違う場合はここに追加する
LABEL_COL_CANDIDATES = [
    "Unicode",
    "unicode",
    "UNICODE",
    "Codepoint",
    "codepoint",
    "CODEPOINT",
    "Label",
    "label",
    "LABEL",
    "Char",
    "char",
    "CHAR",
    "Character",
    "character",
    "Transcription",
    "transcription",
    "Text",
    "text",
    "Class",
    "class",
    "CLASS",
]


def strip_invisible_and_space(txt: str) -> str:
    """
    空白・制御文字・不可視文字を除去して、実質的に空かどうか判定するための文字列を返す。
    """
    out = []
    for ch in txt:
        if ch in INVISIBLE_CHARS:
            continue
        if ch.isspace():
            continue
        o = ord(ch)
        if o < 32 or o == 127:
            continue
        out.append(ch)
    return "".join(out)


def normalize_text(txt: str) -> str:
    """
    評価用の文字列正規化。
    REMOVE_WHITESPACE=True の場合、改行・空白を除去する。
    """
    if REMOVE_WHITESPACE:
        return "".join(ch for ch in txt if not ch.isspace())
    return txt


def normalize_label_text(txt: str) -> str:
    """
    CSVラベル側とGT側の比較用正規化。
    現状は normalize_text と同じだが、必要ならここだけ別調整できる。
    """
    return normalize_text(txt)


def to_merged_page_id(image_name: str) -> str:
    """
    CSVの Image 名を、統合GT用のページIDに変換する。

    例:
        200003076_00004_1 -> 200003076_00004
        200003076_00004_2 -> 200003076_00004

    もしすでに 200003076_00004 の形なら、そのまま返す。
    """
    image_name = str(image_name)

    m = MERGED_PAGE_RE.match(image_name)
    if m:
        return m.group("merged_id")

    # すでに統合済み形式の場合
    if re.match(r"^\d+_\d{5}$", image_name):
        return image_name

    # 想定外形式の場合も落とさず、そのまま返す
    return image_name


def count_bboxes_xycut(df_page: pd.DataFrame) -> int:
    """
    df_page 内のbbox数を、xycutの並び順処理後の個数として数える。
    基本的には len(df_page) と一致するはずだが、
    元コードの挙動を維持するため recursive_xy_cut を通す。
    """
    if len(df_page) == 0:
        return 0

    x1 = df_page["X"].to_numpy()
    y1 = df_page["Y"].to_numpy()
    x2 = x1 + df_page["Width"].to_numpy()
    y2 = y1 + df_page["Height"].to_numpy()

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)

    res = []
    recursive_xy_cut(boxes, np.arange(len(boxes)), res)

    return len(res)


def count_bboxes_for_merged_page(df_merged: pd.DataFrame) -> int:
    """
    統合ページ単位のbbox数を数える。

    注意:
    _1 と _2 は別画像なので、本来は画像ごとにxycutしてから合計する方が安全。
    ここで _1/_2 を混ぜて一発xycutすると、座標系が同じ画像内前提になり、
    並び順や分割が不自然になる可能性がある。

    したがって、
        Imageごとに count_bboxes_xycut()
        それを合計
    とする。
    """
    total = 0
    for image_name in sorted(df_merged["Image"].astype(str).unique()):
        df_one_side = df_merged[df_merged["Image"].astype(str) == image_name]
        total += count_bboxes_xycut(df_one_side)
    return total


def count_target_char(txt: str, target: str = COUNT_CHAR) -> int:
    return txt.count(target)


def detect_label_column(df: pd.DataFrame) -> str | None:
    """
    coordinate.csv から文字ラベル列を自動検出する。
    見つからなければ None。
    """
    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def unicode_codepoint_to_char(s: str) -> str:
    """
    CSVラベルが U+4E00 のような形式なら実文字に変換する。
    それ以外なら元の文字列を返す。

    対応例:
        U+3042 -> あ
        u+3042 -> あ
        3042   -> 3042 のまま
        あ     -> あ
    """
    s = str(s).strip()

    if s == "" or s.lower() == "nan":
        return ""

    # U+3042
    m = re.fullmatch(r"U\+([0-9A-Fa-f]{4,6})", s)
    if m:
        return chr(int(m.group(1), 16))

    # 0x3042
    m = re.fullmatch(r"0x([0-9A-Fa-f]{4,6})", s)
    if m:
        return chr(int(m.group(1), 16))

    # 複数コードポイント: U+3042 U+3099 など
    if "U+" in s:
        parts = re.findall(r"U\+([0-9A-Fa-f]{4,6})", s)
        if parts:
            return "".join(chr(int(p, 16)) for p in parts)

    return s


def get_csv_label_sequence_for_one_side(df_page: pd.DataFrame, label_col: str) -> str:
    """
    片面画像内のCSVラベル列をxycut順に並べて文字列化する。
    """
    if len(df_page) == 0:
        return ""

    x1 = df_page["X"].to_numpy()
    y1 = df_page["Y"].to_numpy()
    x2 = x1 + df_page["Width"].to_numpy()
    y2 = y1 + df_page["Height"].to_numpy()

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)

    res = []
    recursive_xy_cut(boxes, np.arange(len(boxes)), res)

    labels = []
    values = df_page[label_col].tolist()

    for idx in res:
        raw = values[int(idx)]
        ch = unicode_codepoint_to_char(raw)
        labels.append(ch)

    return "".join(labels)


def get_csv_label_sequence_for_merged_page(df_merged: pd.DataFrame, label_col: str) -> str:
    """
    統合ページ単位のCSVラベル文字列を作る。

    _1 と _2 は別画像なので、Imageごとにxycutしてから、
    Image名順に連結する。

    現在は:
        200021712_00004_1
        200021712_00004_2
    の順。
    """
    parts = []

    for image_name in sorted(df_merged["Image"].astype(str).unique()):
        df_one_side = df_merged[df_merged["Image"].astype(str) == image_name]
        side_text = get_csv_label_sequence_for_one_side(df_one_side, label_col)
        parts.append(side_text)

    return "".join(parts)


def format_counter_items(counter: Counter, topn: int = CHAR_DIFF_TOPN) -> str:
    """
    Counterを見やすく表示する。
    例:
        の:2, し:1, 鳥:1
    """
    if not counter:
        return "-"

    items = counter.most_common(topn)
    text = ", ".join([f"{repr(ch)}:{n}" for ch, n in items])

    rest = len(counter) - len(items)
    if rest > 0:
        text += f", ...(+{rest} kinds)"

    return text


def diff_char_counts(csv_text: str, gt_text: str) -> tuple[Counter, Counter]:
    """
    CSVラベル列とGT文字列を文字カウントで比較する。

    Returns:
        missing_in_gt:
            CSVにはあるがGTに足りない文字
        extra_in_gt:
            GTにあるがCSVには余分な文字
    """
    csv_counter = Counter(csv_text)
    gt_counter = Counter(gt_text)

    missing_in_gt = csv_counter - gt_counter
    extra_in_gt = gt_counter - csv_counter

    return missing_in_gt, extra_in_gt


def main():
    df = pd.read_csv(COORD_CSV)

    if "Image" not in df.columns:
        raise KeyError("coordinate.csv に 'Image' 列がありません。")

    label_col = detect_label_column(df)

    # CSV側に統合ページIDを追加
    df["MergedPage"] = df["Image"].astype(str).map(to_merged_page_id)

    merged_pages = sorted(df["MergedPage"].astype(str).unique())

    print("===== BBOX vs MERGED GT LENGTH CHECK =====")
    print(f"CSV              : {COORD_CSV.resolve()}")
    print(f"Total merged pages in CSV : {len(merged_pages)}")
    print(f"GT directory     : {GT_DIR.resolve()}")
    print(f"Count target char: {COUNT_CHAR!r}")

    if label_col is None:
        print("[WARN] label column : not found")
        print("[WARN] char diff    : disabled")
        print(f"[WARN] columns      : {list(df.columns)}")
    else:
        print(f"[INFO] label column : {label_col}")
        print("[INFO] char diff    : enabled")

    print()

    ok_pages = []
    ng_pages = []
    q_pages = []
    blank_pages = []
    missing_pages = []

    # 追加: 文字差分があるページ
    char_diff_pages = []

    total_q = 0

    for merged_page in merged_pages:
        gt_path = GT_DIR / f"{merged_page}.txt"

        if not gt_path.exists():
            # 互換用: もし片面GTが残っている場合も一応探す
            side_paths = [
                GT_DIR / f"{merged_page}_1.txt",
                GT_DIR / f"{merged_page}_2.txt",
            ]

            existing_side_paths = [p for p in side_paths if p.exists()]

            if existing_side_paths:
                # 片面GTがある場合は連結して評価
                gt_text_raw = ""
                for p in existing_side_paths:
                    gt_text_raw += p.read_text(encoding="utf-8", errors="replace")
            else:
                print(f"[SKIP] {merged_page} : GT txt not found")
                missing_pages.append(merged_page)
                continue
        else:
            gt_text_raw = gt_path.read_text(encoding="utf-8", errors="replace")

        df_merged = df[df["MergedPage"] == merged_page]

        # _1/_2 を画像ごとにxycutして合計
        bbox_n = count_bboxes_for_merged_page(df_merged)

        # ファイルサイズ確認
        if gt_path.exists():
            try:
                size = gt_path.stat().st_size
            except OSError:
                size = -1
        else:
            # 片面GTを連結した場合
            size = len(gt_text_raw.encode("utf-8"))

        gt_effective = strip_invisible_and_space(gt_text_raw)
        is_blank = (size == 0) or (len(gt_effective) == 0)

        gt_text = normalize_text(gt_text_raw)
        gt_n = len(gt_text)

        if is_blank:
            reason = "size==0" if size == 0 else "only_whitespace/invisible"
            blank_pages.append((merged_page, bbox_n, reason))

        q_n = count_target_char(gt_text, COUNT_CHAR)
        if q_n > 0:
            q_pages.append((merged_page, q_n))
            total_q += q_n

        if bbox_n == gt_n:
            ok_pages.append(merged_page)
            status = "OK"
        else:
            ng_pages.append((merged_page, bbox_n, gt_n))
            status = "NG"

        # どのImageがこの統合ページに含まれているかも表示
        image_list = sorted(df_merged["Image"].astype(str).unique())
        sides = ",".join(image_list)

        blank_flag = " BLANK" if is_blank else ""

        # 既存出力は維持
        print(
            f"[{status}]{blank_flag} {merged_page} : "
            f"bbox={bbox_n}, gt_chars={gt_n}, {COUNT_CHAR}={q_n}, images=[{sides}]"
        )

        # ===== 追加: 文字差分チェック =====
        if SHOW_CHAR_DIFF and label_col is not None:
            csv_label_text = get_csv_label_sequence_for_merged_page(df_merged, label_col)
            csv_label_text = normalize_label_text(csv_label_text)

            missing_in_gt, extra_in_gt = diff_char_counts(csv_label_text, gt_text)

            if missing_in_gt or extra_in_gt:
                char_diff_pages.append(
                    (
                        merged_page,
                        len(csv_label_text),
                        len(gt_text),
                        missing_in_gt,
                        extra_in_gt,
                    )
                )

    print("\n===== SUMMARY =====")
    print(f"Matched pages  : {len(ok_pages)}")
    print(f"Mismatched     : {len(ng_pages)}")
    print(f"Missing GT     : {len(missing_pages)}")
    print(f"Pages with {COUNT_CHAR!r}: {len(q_pages)}")
    print(f"Total {COUNT_CHAR!r}     : {total_q}")
    print(f"Blank GT pages : {len(blank_pages)}")

    if SHOW_CHAR_DIFF and label_col is not None:
        print(f"Char diff pages: {len(char_diff_pages)}")

    if missing_pages:
        print("\n--- MISSING GT DETAIL ---")
        for p in missing_pages:
            print(f"{p} : {GT_DIR / (p + '.txt')} not found")

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
            print(f"{p} : bbox={b}, gt={g}, diff={b - g}")

    # ===== 追加: 文字差分詳細 =====
    if SHOW_CHAR_DIFF and label_col is not None and char_diff_pages:
        print("\n--- CHAR COUNT DIFF DETAIL ---")
        print("missing_in_gt = CSVにはあるがGTに足りない文字")
        print("extra_in_gt   = GTにあるがCSVより余分な文字")
        print()

        for p, csv_n, gt_n, missing_in_gt, extra_in_gt in char_diff_pages:
            print(f"{p} : csv_chars={csv_n}, gt_chars={gt_n}, diff={csv_n - gt_n}")
            print(f"  missing_in_gt: {format_counter_items(missing_in_gt)}")
            print(f"  extra_in_gt  : {format_counter_items(extra_in_gt)}")


if __name__ == "__main__":
    main()