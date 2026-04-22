#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import subprocess
import sys


# =========================
# 設定（必要ならここだけ変更）
# =========================
BASE_DIR = Path(__file__).resolve().parent

# gt_pages のルート
GT_ROOT = BASE_DIR / "gt_pages"

# ========= 学習用 =========
TRAIN_BOOK_IDS = [
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

TRAIN_TXT = BASE_DIR / "handtrain_charo3.txt"
ARPA = BASE_DIR / "handchar_kenlmo3.arpa"

NGRAM_ORDER = 3        # trigram
ENCODING = "utf-8"

# 空白・改行などを除去して「文字列」化したいなら True
REMOVE_WHITESPACE = True

# =========================
# ★向きの選択（ここだけ切り替え）
# =========================
REVERSE_TRAIN = True  # False: 現状（上→下） / True: 反転（下→上）

# 反転時にファイルを上書きしたくないなら、ここを有効化
if REVERSE_TRAIN:
    TRAIN_TXT = BASE_DIR / "handtrain_charo3_rev.txt"
    ARPA = BASE_DIR / "handchar_kenlmo3_rev.arpa"


def normalize_text(s: str) -> str:
    s = s.strip()
    if REMOVE_WHITESPACE:
        s = "".join(ch for ch in s if not ch.isspace())
    return s


def _maybe_reverse_tokens(line: str) -> str:
    """
    line: normalize_text後の1ページ文字列（基本は空白なし）
    - REVERSE_TRAIN=False: そのまま
    - REVERSE_TRAIN=True : 文字（またはトークン）列を逆順にする
    """
    if not REVERSE_TRAIN:
        return line

    # 既にスペース区切りの可能性にも対応
    if " " in line:
        toks = [t for t in line.split(" ") if t]
        toks = toks[::-1]
        return " ".join(toks)
    else:
        return line[::-1]


# =========================
# 1. gt_pages (学習book群) → train_char.txt
# =========================
def build_train_text():
    pages = []
    missing_dirs = []
    missing_books = []

    for bid in TRAIN_BOOK_IDS:
        d = GT_ROOT / bid
        if not d.exists():
            missing_dirs.append(str(d))
            missing_books.append(bid)
            continue
        pages.extend(sorted(d.glob("*.txt")))

    if missing_dirs:
        print("[WARN] missing gt_pages dirs (TRAIN):")
        for m in missing_dirs:
            print("  ", m)
        print("[WARN] missing book_ids:", ", ".join(missing_books))

    if not pages:
        print(f"[ERROR] no txt files found under {GT_ROOT} for TRAIN_BOOK_IDS")
        sys.exit(1)

    print(f"[INFO] found {len(pages)} txt files from {len(TRAIN_BOOK_IDS)} TRAIN books")
    print(f"[INFO] REVERSE_TRAIN: {REVERSE_TRAIN}")
    print(f"[INFO] writing: {TRAIN_TXT}")

    kept = 0
    skipped_empty = 0

    with TRAIN_TXT.open("w", encoding=ENCODING) as fw:
        for p in pages:
            s = normalize_text(p.read_text(encoding=ENCODING, errors="ignore"))
            if not s:
                skipped_empty += 1
                continue

            # ★必要なら反転（下→上学習）
            s = _maybe_reverse_tokens(s)

            # すでにスペース区切りならそのまま
            if " " in s:
                fw.write(s + "\n")
            else:
                fw.write(" ".join(list(s)) + "\n")
            kept += 1

    print(f"[INFO] lines kept={kept}, empty skipped={skipped_empty}")
    print(f"[INFO] wrote training text: {TRAIN_TXT}")


# =========================
# 2. lmplz → char_kenlm.arpa
# =========================
def build_arpa():
    cmd = [
        "lmplz",
        "-o", str(NGRAM_ORDER),
    ]

    print("[INFO] running:", " ".join(cmd))
    try:
        with TRAIN_TXT.open("r", encoding=ENCODING) as fin, \
             ARPA.open("w", encoding=ENCODING) as fout:
            subprocess.run(
                cmd,
                stdin=fin,
                stdout=fout,
                check=True,
            )
    except FileNotFoundError:
        print("[ERROR] lmplz not found. Is KenLM built and in PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("[ERROR] lmplz failed")
        sys.exit(1)

    print(f"[INFO] wrote ARPA model: {ARPA}")


# =========================
# main
# =========================
def main():
    print("===== KenLM one-shot build (TRAIN books only) =====")
    print(f"[INFO] gt_pages root   : {GT_ROOT}")
    print(f"[INFO] train books     : {len(TRAIN_BOOK_IDS)}")
    print(f"[INFO] ngram order      : {NGRAM_ORDER}")
    print(f"[INFO] remove whitespace: {REMOVE_WHITESPACE}")
    print(f"[INFO] REVERSE_TRAIN    : {REVERSE_TRAIN}")
    print(f"[INFO] TRAIN_TXT        : {TRAIN_TXT}")
    print(f"[INFO] ARPA             : {ARPA}")

    build_train_text()
    build_arpa()

    print("===== DONE =====")
    print("Next step:")
    print("  use this file in kenlm.Model():")
    print(f"    {ARPA}")


if __name__ == "__main__":
    main()


"""
使用するとき（今のシェルのみ有効）
export PATH=/home/ihpc/Documents/saito/KODAI2/kenlm/build/bin:$PATH

確認
which lmplz
lmplz --help | head
"""