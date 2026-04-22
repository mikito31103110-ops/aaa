#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations



import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


# ---------- utilities ----------
def uplus_to_char(s: str) -> str:
    s = str(s)
    if s.startswith("U+") and len(s) >= 4:
        try:
            return chr(int(s[2:], 16))
        except Exception:
            return s
    return s


def normalize_text(txt: str, remove_whitespace: bool) -> str:
    if not remove_whitespace:
        return txt
    return "".join(ch for ch in txt if not ch.isspace())


# ---------- ordering (same idea as your script) ----------
def right_column_top_to_bottom_order(
    boxes: np.ndarray,
    col_tol: float = 1.6,
    row_band_scale: float = 0.8,
    row_band_min: int = 6,
) -> np.ndarray:
    boxes = boxes.astype(int)
    n = len(boxes)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = np.maximum(1, x2 - x1)
    h = np.maximum(1, y2 - y1)

    # column clustering tolerance
    col_gap = int(np.median(w) * 0.8 * float(col_tol))
    col_gap = max(col_gap, 5)

    order_x = np.argsort(cx)[::-1]  # right -> left
    cols: List[List[int]] = []
    reps: List[float] = []

    for idx in order_x:
        placed = False
        for c in range(len(cols)):
            if abs(cx[idx] - reps[c]) <= col_gap:
                cols[c].append(idx)
                reps[c] = float(np.median([cx[i] for i in cols[c]]))
                placed = True
                break
        if not placed:
            cols.append([idx])
            reps.append(float(cx[idx]))

    # normalize col order by rep cx desc
    col_order = np.argsort(np.array(reps))[::-1]
    cols = [cols[i] for i in col_order]

    # row band (absorb small y diff)
    row_band = int(np.median(h) * float(row_band_scale))
    row_band = max(row_band, int(row_band_min))

    out: List[int] = []
    for col in cols:
        col_sorted = sorted(col, key=lambda i: (int(cy[i] // row_band), -cx[i], cy[i]))
        out.extend(col_sorted)

    # defensive fallback
    if len(out) != n or len(set(out)) != n:
        fallback = sorted(range(n), key=lambda i: (-cx[i], cy[i]))
        return np.array(fallback, dtype=int)

    return np.array(out, dtype=int)


def make_text_for_one_df(
    df_group: pd.DataFrame,
    col_tol: float,
    row_band_scale: float,
    row_band_min: int,
) -> str:
    x1 = df_group["X"].to_numpy(dtype=float)
    y1 = df_group["Y"].to_numpy(dtype=float)
    x2 = x1 + df_group["Width"].to_numpy(dtype=float)
    y2 = y1 + df_group["Height"].to_numpy(dtype=float)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    order = right_column_top_to_bottom_order(
        boxes,
        col_tol=col_tol,
        row_band_scale=row_band_scale,
        row_band_min=row_band_min,
    )
    df_sorted = df_group.iloc[order].reset_index(drop=True)
    chars = [uplus_to_char(u) for u in df_sorted["Unicode"].astype(str).tolist()]
    return "".join(chars)


def process_one_book(
    coord_csv: Path,
    out_root: Path,
    book_id: str,
    remove_whitespace: bool,
    use_blocks: bool,
    col_tol: float,
    row_band_scale: float,
    row_band_min: int,
) -> Tuple[int, int]:
    """
    return: (pages_written, pages_total)
    """
    df = pd.read_csv(coord_csv)

    need_cols = ["Unicode", "Image", "X", "Y", "Width", "Height"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"[{book_id}] coord_csv missing column: {c}  in {coord_csv}")

    has_block = ("Block ID" in df.columns)

    out_dir = out_root / book_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(df["Image"].astype(str).unique().tolist())
    written = 0

    for page in pages:
        d = df[df["Image"].astype(str) == page].copy()
        if len(d) == 0:
            continue

        if use_blocks and has_block:
            blk = d["Block ID"].copy()
            blk_filled = pd.to_numeric(blk, errors="coerce").fillna(1e18)
            d = d.assign(_block_sort=blk_filled)

            parts: List[str] = []
            for _, dblk in d.sort_values("_block_sort").groupby("Block ID", dropna=False):
                parts.append(
                    make_text_for_one_df(
                        dblk, col_tol=col_tol, row_band_scale=row_band_scale, row_band_min=row_band_min
                    )
                )
            txt = "".join(parts)
        else:
            txt = make_text_for_one_df(d, col_tol=col_tol, row_band_scale=row_band_scale, row_band_min=row_band_min)

        txt = normalize_text(txt, remove_whitespace=remove_whitespace)

        stem = Path(str(page)).stem
        out_path = out_dir / f"{stem}.txt"
        out_path.write_text(txt, encoding="utf-8")
        written += 1

    return written, len(pages)


def find_coordinate_csvs(full_root: Path) -> List[Tuple[str, Path]]:
    """
    full_root/<book_id>/<book_id>_coordinate.csv を探索して返す
    return: [(book_id, csv_path), ...]
    """
    results: List[Tuple[str, Path]] = []
    if not full_root.exists():
        raise FileNotFoundError(f"full_root not found: {full_root}")

    for book_dir in sorted([p for p in full_root.iterdir() if p.is_dir()]):
        book_id = book_dir.name
        csv1 = book_dir / f"{book_id}_coordinate.csv"
        if csv1.exists():
            results.append((book_id, csv1))
            continue

        # 保険: *_coordinate.csv が1つだけある場合
        cands = list(book_dir.glob("*_coordinate.csv"))
        if len(cands) == 1:
            results.append((book_id, cands[0]))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-root", type=Path, required=True,
                    help="例: /home/ihpc/Documents/saito/KODAI/full")
    ap.add_argument("--out-root", type=Path, default=Path("./gt_pages"))
    ap.add_argument("--remove-whitespace", action="store_true")
    ap.add_argument("--use-blocks", action="store_true")

    # same knobs
    ap.add_argument("--col-tol", type=float, default=1.0)
    ap.add_argument("--row-band-scale", type=float, default=1.0)
    ap.add_argument("--row-band-min", type=int, default=6)

    # optional: only these book ids
    ap.add_argument("--only", type=str, default="",
                    help="例: 200014740,200003076 のようにカンマ区切り。空なら全て。")

    args = ap.parse_args()

    full_root = args.full_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    targets = find_coordinate_csvs(full_root)

    if args.only.strip():
        allow = set(x.strip() for x in args.only.split(",") if x.strip())
        targets = [(bid, p) for (bid, p) in targets if bid in allow]

    if not targets:
        raise SystemExit("[ERROR] coordinate.csv が見つかりませんでした。full_root を確認してください。")

    print(f"[INFO] found books: {len(targets)}")
    print(f"[INFO] out_root: {out_root}")
    print(f"[INFO] params: remove_ws={bool(args.remove_whitespace)} use_blocks={bool(args.use_blocks)} "
          f"col_tol={args.col_tol} row_band_scale={args.row_band_scale} row_band_min={args.row_band_min}")

    ok_books = 0
    total_pages = 0
    written_pages = 0
    failed: List[Tuple[str, str]] = []

    for book_id, csv_path in targets:
        try:
            w, t = process_one_book(
                coord_csv=csv_path,
                out_root=out_root,
                book_id=book_id,
                remove_whitespace=bool(args.remove_whitespace),
                use_blocks=bool(args.use_blocks),
                col_tol=float(args.col_tol),
                row_band_scale=float(args.row_band_scale),
                row_band_min=int(args.row_band_min),
            )
            ok_books += 1
            written_pages += w
            total_pages += t
            print(f"[OK] {book_id}: pages {w}/{t}")
        except Exception as e:
            failed.append((book_id, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {book_id}: {type(e).__name__}: {e}")

    print("\n===== SUMMARY =====")
    print(f"books ok     : {ok_books}/{len(targets)}")
    print(f"pages written: {written_pages}/{total_pages}")

    if failed:
        print("\n[FAILED LIST]")
        for bid, msg in failed:
            print(f"- {bid}: {msg}")


if __name__ == "__main__":
    main()


"""
python makegtpagesq.py \
  --full-root /home/ihpc/Documents/saito/KODAI/full \
  --out-root ./full_gt_pages \
  --remove-whitespace \
  --col-tol 1.0 \
  --row-band-scale 1.0
"""