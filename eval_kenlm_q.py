#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
変更点:
- overlay文字の表示順: 左=GT, 真ん中=Baseline(Greedy), 右=LM
- 〱対策: 日本語CJKフォントを優先ロード
- --text-y-offset を追加
    負の値: 文字を上へ
    正の値: 文字を下へ
    例: --text-y-offset -3
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

try:
    import timm
except Exception as e:
    raise SystemExit(f"timm が必要です: pip install timm\nImportError: {e}")

try:
    import kenlm  # type: ignore
except Exception as e:
    raise SystemExit(f"python kenlm が必要です: pip install kenlm\nImportError: {e}")


class LetterboxSquare:
    def __init__(self, size: int, fill: int = 255, interp=Image.BICUBIC):
        self.size = int(size)
        self.fill = int(fill)
        self.interp = interp

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))

        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        img_r = img.resize((nw, nh), resample=self.interp)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        canvas.paste(img_r, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def read_classes_txt(p: Path) -> list[str]:
    classes = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"classes.txt is empty: {p}")
    return classes


def uplus_to_char(s: str) -> str:
    s = str(s)
    if s.startswith("U+") and len(s) >= 4:
        try:
            return chr(int(s[2:], 16))
        except Exception:
            return s
    return s


def find_page_image(images_dir: Path, image_key: str) -> Optional[Path]:
    for ext in [".jpg", ".png", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"]:
        cand = images_dir / f"{image_key}{ext}"
        if cand.exists():
            return cand
    return None


def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(1, min(W, round(x2))))
    y2 = int(max(1, min(H, round(y2))))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def load_gt_chars(gt_path: Path) -> List[str]:
    txt = gt_path.read_text(encoding="utf-8", errors="ignore")
    txt = "".join(ch for ch in txt if not ch.isspace())
    return list(txt)


def right_column_top_to_bottom_order(boxes: np.ndarray) -> np.ndarray:
    boxes = boxes.astype(int)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    widths = np.maximum(1, boxes[:, 2] - boxes[:, 0])
    col_gap = max(int(np.median(widths) * 0.8), 5)

    order_x = np.argsort(cx)[::-1]
    col_id = np.zeros(len(boxes), dtype=int)

    cur = 0
    last_x = cx[order_x[0]]
    col_id[order_x[0]] = cur

    for idx in order_x[1:]:
        if abs(cx[idx] - last_x) > col_gap:
            cur += 1
        col_id[idx] = cur
        last_x = cx[idx]

    order = sorted(range(len(boxes)), key=lambda i: (col_id[i], cy[i]))
    return np.array(order, dtype=int)


@dataclass
class Item:
    page: str
    uni_true: str
    bbox: Tuple[int, int, int, int]
    topk_ids: List[int]
    topk_logp: List[float]


def beam_search_char_lm(
    items: List[Item],
    lm: kenlm.Model,
    classes_u: List[str],
    beam_size: int = 20,
    lam: float = 0.3,
) -> List[str]:
    state0 = kenlm.State()
    lm.BeginSentenceWrite(state0)

    beams = [([], 0.0, state0)]

    for it in items:
        new_beams = []
        for seq, score, st in beams:
            for cid, lp in zip(it.topk_ids, it.topk_logp):
                ch = uplus_to_char(classes_u[cid])
                out_st = kenlm.State()
                lm_score = lm.BaseScore(st, ch, out_st)
                s = score + float(lp) + lam * float(lm_score)
                new_beams.append((seq + [ch], s, out_st))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    return beams[0][0]


def _load_font(font_path: str, font_size: int) -> ImageFont.ImageFont:
    candidates = []

    if font_path:
        candidates.append(font_path)

    candidates += [
        "./fonts/NotoSerifCJKjp-Regular.otf",
        "/home/ihpc/Documents/saito/KODAI2/fonts/NotoSerifCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf",
    ]

    for fp in candidates:
        try:
            p = Path(fp)
            if p.exists():
                return ImageFont.truetype(str(p), font_size)
        except Exception as e:
            print(f"[WARN] failed to load font: {fp} ({e})")

    print("[WARN] Japanese font not found. Some characters such as 〱 may be garbled.")
    return ImageFont.load_default()


def draw_overlay_one_page(
    page_img: Image.Image,
    items: List[Item],
    gt_chars: List[str],
    greedy_chars: List[str],
    lm_chars: List[str],
    out_path: Path,
    font_path: str,
    font_size: int,
    box_width: int,
    max_labels: int,
    draw_ok: bool,
    text_y_offset: int,
) -> None:
    im = page_img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _load_font(font_path, font_size)

    n = min(len(items), len(gt_chars), len(greedy_chars), len(lm_chars), max_labels)

    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    OKC = (180, 180, 180)

    shown = 0
    for i in range(n):
        x1, y1, x2, y2 = items[i].bbox
        gt = gt_chars[i]
        g = greedy_chars[i]
        lm = lm_chars[i]

        g_ok = g == gt
        lm_ok = lm == gt

        if g_ok and lm_ok:
            if not draw_ok:
                continue
            color = OKC
        elif (not g_ok) and lm_ok:
            color = GREEN
        elif g_ok and (not lm_ok):
            color = RED
        else:
            color = YELLOW

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        labels = [gt, g, lm]

        tx, ty = x1, y1 - (font_size + 6)
        if ty < 0:
            ty = y1 + 2

        pad = 3
        gap = 4

        metrics = []
        for lab in labels:
            try:
                left, top, right, bottom = draw.textbbox((0, 0), lab, font=font)
                tw = right - left
                th = bottom - top
            except Exception:
                tw, th = draw.textsize(lab, font=font)
                left, top = 0, 0

            metrics.append({
                "text": lab,
                "left": left,
                "top": top,
                "width": tw,
                "height": th,
            })

        total_w = sum(m["width"] for m in metrics) + gap * (len(metrics) - 1)
        total_h = max(m["height"] for m in metrics)

        extra_up = max(0, -text_y_offset)
        extra_down = max(0, text_y_offset)

        bg_x1 = tx
        bg_y1 = ty - extra_up
        bg_x2 = tx + total_w + pad * 2
        bg_y2 = ty + total_h + pad * 2 + extra_down

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0))

        cur_x = tx + pad
        target_top_y = ty + pad + text_y_offset

        for m in metrics:
            draw_x = cur_x - m["left"]
            draw_y = target_top_y - m["top"]

            draw.text(
                (draw_x, draw_y),
                m["text"],
                fill=(255, 255, 255),
                font=font,
            )
            cur_x += m["width"] + gap

        shown += 1
        if shown >= max_labels:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def build_eval_transform(model, img_size: int):
    default_cfg = getattr(model, "default_cfg", {}) or {}
    mean = tuple(default_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(default_cfg.get("std", (0.229, 0.224, 0.225)))

    tfm = transforms.Compose([
        LetterboxSquare(img_size, fill=255, interp=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coord-csv", default="/home/ihpc/Documents/saito/KODAI/full/100249371/100249371_coordinate.csv")
    ap.add_argument("--images-dir", default="/home/ihpc/Documents/saito/KODAI/full/100249371/images")
    ap.add_argument("--gt-dir", default="/home/ihpc/Documents/saito/KODAI2/gt_pages/100241706")
    ap.add_argument("--arpa", default="/home/ihpc/Documents/saito/KODAI2/char_kenlm.arpa")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam-size", type=int, default=20)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.3)

    ap.add_argument("--pages", default="all")
    ap.add_argument("--out-dir", default="./eval_out_xycut")
    ap.add_argument("--demo-page", default="")

    ap.add_argument("--annotate-all", action="store_true")
    ap.add_argument("--font-path", default="")
    ap.add_argument("--font-size", type=int, default=18)
    ap.add_argument("--box-width", type=int, default=4)
    ap.add_argument("--max-labels", type=int, default=2000)
    ap.add_argument("--draw-ok", action="store_true")

    ap.add_argument(
        "--text-y-offset",
        type=int,
        default=-3,
        help="overlay文字の上下補正。負の値で上、正の値で下。例: -3",
    )

    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--min-char-count", type=int, default=5)
    ap.add_argument("--oracle-k", type=int, default=5)
    ap.add_argument("--confusion-topn", type=int, default=200)
    ap.add_argument("--confusion-min-count", type=int, default=5)

    args = ap.parse_args()

    coord_csv = Path(args.coord_csv)
    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    arpa = Path(args.arpa)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not coord_csv.exists():
        raise FileNotFoundError(coord_csv)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if not arpa.exists():
        raise FileNotFoundError(arpa)

    print("[INFO] loading kenlm...")
    lm = kenlm.Model(str(arpa))
    print(f"[INFO] kenlm order = {lm.order}")

    classes_u = read_classes_txt(Path(args.classes))
    K = len(classes_u)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] device: {device}")

    model = timm.create_model(args.model, pretrained=True, num_classes=K)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    tfm, mean, std = build_eval_transform(model, img_size=int(args.img_size))
    print(f"[INFO] eval tfm: LetterboxSquare({int(args.img_size)}) -> ToTensor -> Normalize")
    print(f"[INFO] normalize mean/std: mean={mean} std={std}")

    df = pd.read_csv(coord_csv)
    need_cols = ["Unicode", "Image", "X", "Y", "Width", "Height"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"coord_csv missing column: {c}")

    all_pages = sorted(df["Image"].astype(str).unique().tolist())

    if args.pages == "all":
        use_pages = all_pages
    else:
        use_pages = [p.strip() for p in args.pages.split(",") if p.strip()]

    use_pages2 = []
    for p in use_pages:
        if (gt_dir / f"{p}.txt").exists():
            use_pages2.append(p)
        else:
            print(f"[WARN] missing gt: {gt_dir / (p + '.txt')}")
    use_pages = use_pages2

    print(f"[INFO] pages (csv unique) : {len(all_pages)}")
    print(f"[INFO] pages (use)        : {len(use_pages)}")
    print(f"[INFO] text_y_offset      : {int(args.text_y_offset)}")

    topk = max(1, int(args.topk))
    beam_size = max(1, int(args.beam_size))
    lam = float(args.lam)

    total_chars = 0
    correct_greedy = 0
    correct_lm = 0
    pages_done = 0
    pages_skipped_len = 0
    pages_skipped_img = 0

    demo_result = None
    demo_page = args.demo_page.strip() or (use_pages[0] if use_pages else "")

    overlay_dir = out_dir / "overlays"
    if args.annotate_all:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)
    conf_g = Counter()
    conf_lm = Counter()
    oracle_k = max(1, int(args.oracle_k))

    with torch.no_grad():
        for page in use_pages:
            d = df[df["Image"].astype(str) == page].copy()
            if len(d) == 0:
                continue

            x1 = d["X"].to_numpy(dtype=float)
            y1 = d["Y"].to_numpy(dtype=float)
            x2 = x1 + d["Width"].to_numpy(dtype=float)
            y2 = y1 + d["Height"].to_numpy(dtype=float)
            boxes = np.stack([x1, y1, x2, y2], axis=1)

            order = right_column_top_to_bottom_order(boxes)
            d = d.iloc[order].reset_index(drop=True)
            boxes = boxes[order].astype(int)

            gt_path = gt_dir / f"{page}.txt"
            gt_chars = load_gt_chars(gt_path)

            if len(gt_chars) != len(d):
                pages_skipped_len += 1
                print(f"[WARN] length mismatch page={page}  bbox={len(d)}  gt={len(gt_chars)}  -> skip")
                continue

            img_path = find_page_image(images_dir, page)
            if img_path is None:
                pages_skipped_img += 1
                print(f"[WARN] missing image file for page: {page} -> skip")
                continue

            im = Image.open(img_path).convert("RGB")
            W, H = im.size

            crops = []
            for i in range(len(d)):
                r = d.iloc[i]
                x = float(r["X"])
                y = float(r["Y"])
                w = float(r["Width"])
                h = float(r["Height"])
                xx1, yy1, xx2, yy2 = clamp_bbox(x, y, x + w, y + h, W, H)
                crops.append(tfm(im.crop((xx1, yy1, xx2, yy2))))

            if not crops:
                continue

            items: List[Item] = []
            bs = int(args.batch_size)

            for s in range(0, len(crops), bs):
                batch = torch.stack(crops[s:s + bs], dim=0).to(device, non_blocking=True)
                logits = model(batch)

                for bi in range(logits.shape[0]):
                    logp = torch.log_softmax(logits[bi], dim=0)
                    k = min(topk, K)
                    topv, topi = torch.topk(logp, k=k)

                    items.append(
                        Item(
                            page=page,
                            uni_true=str(d.iloc[s + bi]["Unicode"]),
                            bbox=tuple(map(int, boxes[s + bi].tolist())),
                            topk_ids=topi.detach().cpu().numpy().tolist(),
                            topk_logp=topv.detach().cpu().numpy().tolist(),
                        )
                    )

            greedy_chars = [uplus_to_char(classes_u[it.topk_ids[0]]) for it in items]

            lm_chars = beam_search_char_lm(
                items=items,
                lm=lm,
                classes_u=classes_u,
                beam_size=beam_size,
                lam=lam,
            )

            g_ok = sum(1 for a, b in zip(greedy_chars, gt_chars) if a == b)
            l_ok = sum(1 for a, b in zip(lm_chars, gt_chars) if a == b)

            total_chars += len(gt_chars)
            correct_greedy += g_ok
            correct_lm += l_ok
            pages_done += 1

            if args.stats:
                for idx, (gt_ch, g_ch, lm_ch) in enumerate(zip(gt_chars, greedy_chars, lm_chars)):
                    overall["total"] += 1
                    counts[gt_ch]["total"] += 1

                    k_orc = min(oracle_k, len(items[idx].topk_ids))
                    top_chars_orc = [uplus_to_char(classes_u[cid]) for cid in items[idx].topk_ids[:k_orc]]
                    if gt_ch in top_chars_orc:
                        overall["oracle_in_topK"] += 1
                        counts[gt_ch]["oracle_in_topK"] += 1

                    g_ok1 = g_ch == gt_ch
                    lm_ok1 = lm_ch == gt_ch

                    conf_g[(gt_ch, g_ch)] += 1
                    conf_lm[(gt_ch, lm_ch)] += 1

                    if g_ok1 and lm_ok1:
                        overall["ok"] += 1
                        counts[gt_ch]["ok"] += 1
                    elif (not g_ok1) and lm_ok1:
                        overall["fix"] += 1
                        counts[gt_ch]["fix"] += 1
                        counts[gt_ch]["greedy_err"] += 1
                    elif g_ok1 and (not lm_ok1):
                        overall["worse"] += 1
                        counts[gt_ch]["worse"] += 1
                        counts[gt_ch]["lm_err"] += 1
                    else:
                        overall["bothx"] += 1
                        counts[gt_ch]["bothx"] += 1
                        counts[gt_ch]["greedy_err"] += 1
                        counts[gt_ch]["lm_err"] += 1

            if args.annotate_all:
                draw_overlay_one_page(
                    page_img=im,
                    items=items,
                    gt_chars=gt_chars,
                    greedy_chars=greedy_chars,
                    lm_chars=lm_chars,
                    out_path=overlay_dir / f"overlay_{page}.png",
                    font_path=args.font_path,
                    font_size=int(args.font_size),
                    box_width=int(args.box_width),
                    max_labels=int(args.max_labels),
                    draw_ok=bool(args.draw_ok),
                    text_y_offset=int(args.text_y_offset),
                )

            if page == demo_page:
                demo_result = (page, gt_chars, greedy_chars, lm_chars)

    print("\n===== OVERALL =====")
    print(f"pages evaluated : {pages_done}")
    print(f"pages skipped (len mismatch) : {pages_skipped_len}")
    print(f"pages skipped (no image)     : {pages_skipped_img}")

    if total_chars == 0:
        print("[ERROR] evaluated chars = 0. GT長さ不一致 / 画像なし / gt_pages不足 の可能性が高いです。")
        return

    print(f"Greedy acc    : {correct_greedy / total_chars:.4f}")
    print(f"Greedy+LM acc : {correct_lm / total_chars:.4f}")

    if args.stats:
        def ratio(x, d):
            return (x / d) if d > 0 else 0.0

        overall_txt = out_dir / "overall_stats.txt"
        with overall_txt.open("w", encoding="utf-8") as f:
            f.write("=== OVERALL CHAR STATS ===\n")
            f.write(f"total={overall['total']}\n")
            f.write(f"ok={overall['ok']}  ({ratio(overall['ok'], overall['total']):.6f})\n")
            f.write(f"fix={overall['fix']}  ({ratio(overall['fix'], overall['total']):.6f})\n")
            f.write(f"worse={overall['worse']}  ({ratio(overall['worse'], overall['total']):.6f})\n")
            f.write(f"bothx={overall['bothx']}  ({ratio(overall['bothx'], overall['total']):.6f})\n")
            f.write("\n")
            f.write(f"oracle_k={oracle_k}\n")
            f.write(f"oracle_in_topK={overall['oracle_in_topK']}  ({ratio(overall['oracle_in_topK'], overall['total']):.6f})\n")
            f.write("\n")
            g_err_total = overall["fix"] + overall["bothx"]
            f.write(f"greedy_error_total={g_err_total}\n")
            f.write(f"repair_rate = fix / (fix+bothx) = {ratio(overall['fix'], g_err_total):.6f}\n")

        print(f"[INFO] wrote: {overall_txt}")

        rows = []
        min_cnt = int(args.min_char_count)

        for ch, dct in counts.items():
            total = int(dct["total"])
            if total < min_cnt:
                continue

            fix = int(dct["fix"])
            bothx = int(dct["bothx"])

            rows.append({
                "char_gt": ch,
                "count": total,
                "ok": int(dct["ok"]),
                "fix": fix,
                "worse": int(dct["worse"]),
                "bothx": bothx,
                "greedy_err": int(dct["greedy_err"]),
                "lm_err": int(dct["lm_err"]),
                "oracle_in_topK": int(dct["oracle_in_topK"]),
                "greedy_miss_rate": ratio(int(dct["greedy_err"]), total),
                "lm_miss_rate": ratio(int(dct["lm_err"]), total),
                "oracle_rate": ratio(int(dct["oracle_in_topK"]), total),
                "fix_rate_total": ratio(fix, total),
                "worse_rate_total": ratio(int(dct["worse"]), total),
                "bothx_rate_total": ratio(bothx, total),
                "repair_rate_char": ratio(fix, fix + bothx),
            })

        df_stats = pd.DataFrame(rows)
        if len(df_stats) > 0:
            df_stats = df_stats.sort_values(["count", "greedy_miss_rate"], ascending=[False, False])

        out_csv = out_dir / "char_stats.csv"
        df_stats.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote: {out_csv}  (rows={len(df_stats)})")

        def dump_confusion(counter: Counter, name: str):
            rows2 = []
            for (gt, pred), c in counter.most_common():
                if c < int(args.confusion_min_count):
                    break
                rows2.append({
                    "gt": gt,
                    "pred": pred,
                    "count": c,
                    "is_correct": int(gt == pred),
                })
                if len(rows2) >= int(args.confusion_topn):
                    break

            dfc = pd.DataFrame(rows2)
            outp = out_dir / f"confusion_{name}_top{int(args.confusion_topn)}.csv"
            dfc.to_csv(outp, index=False, encoding="utf-8-sig")
            print(f"[INFO] wrote: {outp}  (rows={len(dfc)})")

        dump_confusion(conf_g, "greedy")
        dump_confusion(conf_lm, "lm")

    if demo_result is not None:
        page, gt, g, lm_out = demo_result

        def chunk(xs, n=80):
            s = "".join(xs)
            return "\n".join(s[i:i + n] for i in range(0, len(s), n))

        print("\n===== DEMO PAGE =====")
        print(f"[page] {page}")
        print("GT        :")
        print(chunk(gt))
        print("\nGreedy    :")
        print(chunk(g))
        print("\nGreedy+LM :")
        print(chunk(lm_out))


if __name__ == "__main__":
    main()

"""

python eval_kenlm_q.py \
  --coord-csv /home/ihpc/Documents/saito/KODAI/full/200018243/200018243_coordinate.csv \
  --images-dir /home/ihpc/Documents/saito/KODAI/full/200018243/images \
  --gt-dir /home/ihpc/Documents/saito/KODAI2/gt_pages/200018243 \
  --arpa /home/ihpc/Documents/saito/KODAI2/handchar_kenlmo3.arpa \
  --ckpt ./runs/kodai_effb0_pretrained/best.pth \
  --classes ./runs/kodai_effb0_pretrained/classes.txt \
  --model efficientnet_b0 \
  --img-size 224 \
  --lambda 1.6 \
  --topk 5 \
  --beam-size 5 \
  --annotate-all \
  --stats \
  --oracle-k 5 \
  --confusion-topn 200 \
  --confusion-min-count 5 \
  --min-char-count 10 \
  --font-path /home/ihpc/Documents/saito/KODAI2/fonts/NotoSerifCJKjp-Regular.otf \
  --box-width 4 \
  --out-dir ./outputs/picture/200018243/1.6ken_qqq
  --text-y-offset -3

"""