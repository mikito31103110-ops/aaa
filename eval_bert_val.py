#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations


import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

try:
    import timm
except Exception as e:
    raise SystemExit(
        "timm が必要です。環境に無いなら: pip install timm\n"
        f"ImportError: {e}"
    )

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception as e:
    raise SystemExit(
        "transformers が必要です。環境に無いなら: pip install transformers\n"
        f"ImportError: {e}"
    )


# ----------------------------
# Shared-like transform pieces
# ----------------------------
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
        left = (self.size - nw) // 2
        top = (self.size - nh) // 2
        canvas.paste(img_r, (left, top))
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
    col_gap = int(np.median(widths) * 0.8)
    col_gap = max(col_gap, 5)

    order_x = np.argsort(cx)[::-1]  # 右→左
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


# ----------------------------
# BERT scoring
# ----------------------------
def encode_no_special(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@torch.no_grad()
def score_char_with_mask(
    tokenizer,
    model,
    device: torch.device,
    full_text: str,
    char_pos: int,
    target_char: str,
) -> float:
    """
    full_text の char_pos だけ [MASK] に置き換えて、
    target_char がそこに入る log-prob 近似を返す。

    注意:
    - char model でも 1文字=1token と決め打ちしない
    - multi-token の場合は mask を複数並べて各位置の log-prob を和
    - これは厳密な joint ではなく局所条件付きスコア
    """
    if not (0 <= char_pos < len(full_text)):
        return -1e9

    prefix = full_text[:char_pos]
    suffix = full_text[char_pos + 1:]

    prefix_ids = encode_no_special(tokenizer, prefix)
    target_ids = encode_no_special(tokenizer, target_char)
    suffix_ids = encode_no_special(tokenizer, suffix)

    if len(target_ids) == 0:
        return -1e9

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("tokenizer.mask_token_id is None")

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("tokenizer cls/sep token id is None")

    input_ids = [cls_id] + prefix_ids + [mask_id] * len(target_ids) + suffix_ids + [sep_id]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    logits = model(input_ids=x).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)

    start = 1 + len(prefix_ids)
    score = 0.0
    for i, tid in enumerate(target_ids):
        score += float(log_probs[start + i, tid].item())
    return score


@torch.no_grad()
def bert_iterative_rerank(
    items: List[Item],
    classes_u: List[str],
    tokenizer,
    bert_model,
    device: torch.device,
    lam: float = 1.0,
    num_passes: int = 1,
) -> Tuple[List[str], List[float]]:
    """
    Greedy を初期列として、各位置を [MASK] にして top-k 候補を BERT 再採点。
    fused_score = image_logp + lam * bert_score

    num_passes > 1 なら、更新後の列でもう一度回す。
    """
    if not items:
        return [], []

    cur_chars = [uplus_to_char(classes_u[it.topk_ids[0]]) for it in items]
    final_scores = [0.0] * len(items)

    for _ in range(max(1, int(num_passes))):
        new_chars = cur_chars[:]
        new_scores = final_scores[:]

        for i, it in enumerate(items):
            best_char = cur_chars[i]
            best_score = -1e18

            for cid, img_lp in zip(it.topk_ids, it.topk_logp):
                cand_ch = uplus_to_char(classes_u[cid])

                tmp_chars = cur_chars[:]
                tmp_chars[i] = cand_ch
                full_text = "".join(tmp_chars)

                bert_lp = score_char_with_mask(
                    tokenizer=tokenizer,
                    model=bert_model,
                    device=device,
                    full_text=full_text,
                    char_pos=i,
                    target_char=cand_ch,
                )

                fused = float(img_lp) + float(lam) * float(bert_lp)
                if fused > best_score:
                    best_score = fused
                    best_char = cand_ch

            new_chars[i] = best_char
            new_scores[i] = best_score

        cur_chars = new_chars
        final_scores = new_scores

    return cur_chars, final_scores


# ----------------------------
# Overlay
# ----------------------------
def _load_font(font_path: str, font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"[WARN] failed to load font: {font_path} ({e}) -> fallback default font")
    return ImageFont.load_default()


def draw_overlay_one_page(
    page_img: Image.Image,
    items: List[Item],
    gt_chars: List[str],
    greedy_chars: List[str],
    bert_chars: List[str],
    out_path: Path,
    font_path: str,
    font_size: int,
    box_width: int,
    max_labels: int,
    draw_ok: bool,
) -> None:
    """
    3文字: Greedy / GT / BERT
    色:
    - FIX   : Greedy誤り -> BERT正解 = 緑
    - WORSE : Greedy正解 -> BERT誤り = 赤
    - BOTHx : 両方誤り = 黄
    - OK    : draw_ok=True のとき薄灰
    """
    im = page_img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _load_font(font_path, font_size)

    n = min(len(items), len(gt_chars), len(greedy_chars), len(bert_chars), max_labels)

    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    OKC = (180, 180, 180)

    shown = 0
    for i in range(n):
        x1, y1, x2, y2 = items[i].bbox
        gt = gt_chars[i]
        g = greedy_chars[i]
        b = bert_chars[i]

        g_ok = (g == gt)
        b_ok = (b == gt)

        if g_ok and b_ok:
            if not draw_ok:
                continue
            color = OKC
        elif (not g_ok) and b_ok:
            color = GREEN
        elif g_ok and (not b_ok):
            color = RED
        else:
            color = YELLOW

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        label = f"{g}{gt}{b}"

        tx, ty = x1, y1 - (font_size + 6)
        if ty < 0:
            ty = y1 + 2

        try:
            bb = draw.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = draw.textsize(label, font=font)

        pad = 2
        draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2], fill=(0, 0, 0))
        draw.text((tx + pad, ty + pad), label, fill=(255, 255, 255), font=font)

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


def format_lambda_dirname(lam: float) -> str:
    return f"{lam:.1f}"


def build_lambda_list(start: float = 0.2, stop: float = 2.0, step: float = 0.2) -> List[float]:
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


def write_stats_and_confusions(
    out_dir: Path,
    overall: defaultdict,
    counts: defaultdict,
    conf_g: Counter,
    conf_b: Counter,
    oracle_k: int,
    min_char_count: int,
    confusion_topn: int,
    confusion_min_count: int,
) -> None:
    def ratio(x, d):
        return (x / d) if d > 0 else 0.0

    ok_r = ratio(overall["ok"], overall["total"])
    fix_r = ratio(overall["fix"], overall["total"])
    worse_r = ratio(overall["worse"], overall["total"])
    bothx_r = ratio(overall["bothx"], overall["total"])
    oracle_r = ratio(overall["oracle_in_topK"], overall["total"])

    g_err_total = overall["fix"] + overall["bothx"]
    repair_r = ratio(overall["fix"], g_err_total)

    overall_txt = out_dir / "overall_stats.txt"
    with overall_txt.open("w", encoding="utf-8") as f:
        f.write("=== OVERALL CHAR STATS ===\n")
        f.write(f"total={overall['total']}\n")
        f.write(f"ok={overall['ok']}  ({ok_r:.6f})\n")
        f.write(f"fix={overall['fix']}  ({fix_r:.6f})\n")
        f.write(f"worse={overall['worse']}  ({worse_r:.6f})\n")
        f.write(f"bothx={overall['bothx']}  ({bothx_r:.6f})\n\n")
        f.write(f"oracle_k={oracle_k}\n")
        f.write(f"oracle_in_topK={overall['oracle_in_topK']}  ({oracle_r:.6f})\n\n")
        f.write(f"greedy_error_total={g_err_total}\n")
        f.write(f"repair_rate = fix / (fix+bothx) = {repair_r:.6f}\n")
    print(f"[INFO] wrote: {overall_txt}")

    rows = []
    for ch, dct in counts.items():
        total = int(dct["total"])
        if total < int(min_char_count):
            continue

        ok = int(dct["ok"])
        fix = int(dct["fix"])
        worse = int(dct["worse"])
        bothx = int(dct["bothx"])
        g_err = int(dct["greedy_err"])
        b_err = int(dct["bert_err"])
        orc = int(dct["oracle_in_topK"])

        rows.append({
            "char_gt": ch,
            "count": total,
            "ok": ok,
            "fix": fix,
            "worse": worse,
            "bothx": bothx,
            "greedy_err": g_err,
            "bert_err": b_err,
            "oracle_in_topK": orc,
            "greedy_miss_rate": g_err / total if total > 0 else 0.0,
            "bert_miss_rate": b_err / total if total > 0 else 0.0,
            "oracle_rate": orc / total if total > 0 else 0.0,
            "fix_rate_total": fix / total if total > 0 else 0.0,
            "worse_rate_total": worse / total if total > 0 else 0.0,
            "bothx_rate_total": bothx / total if total > 0 else 0.0,
            "repair_rate_char": fix / (fix + bothx) if (fix + bothx) > 0 else 0.0,
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
            if c < int(confusion_min_count):
                break
            rows2.append({
                "gt": gt,
                "pred": pred,
                "count": c,
                "is_correct": int(gt == pred),
            })
            if len(rows2) >= int(confusion_topn):
                break
        dfc = pd.DataFrame(rows2)
        outp = out_dir / f"confusion_{name}_top{int(confusion_topn)}.csv"
        dfc.to_csv(outp, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote: {outp}  (rows={len(dfc)})")

    dump_confusion(conf_g, "greedy")
    dump_confusion(conf_b, "bert")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--coord-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--gt-dir", required=True)

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)

    # BERT
    ap.add_argument("--bert-model", required=True,
                    help="HF model name or local path. ex) tohoku-nlp/bert-base-japanese-char-v3 or runs/train_bert_kodai/final")
    ap.add_argument("--bert-lambda", type=float, default=None,
                    help="単一lambdaだけ回したいとき用。未指定なら 0.2..2.0 を 0.2 刻みで sweep")
    ap.add_argument("--bert-lambda-start", type=float, default=0.2)
    ap.add_argument("--bert-lambda-stop", type=float, default=2.0)
    ap.add_argument("--bert-lambda-step", type=float, default=0.2)
    ap.add_argument("--bert-passes", type=int, default=1,
                    help="iterative rerank passes")
    ap.add_argument("--topk", type=int, default=5)

    ap.add_argument("--pages", default="all", help="'all' or comma-separated list")
    ap.add_argument("--out-dir", default="./outputs/val/bert")
    ap.add_argument("--demo-page", default="", help="show one page output")

    # 可視化
    ap.add_argument("--annotate-all", action="store_true")
    ap.add_argument("--font-path", default="")
    ap.add_argument("--font-size", type=int, default=18)
    ap.add_argument("--box-width", type=int, default=4)
    ap.add_argument("--max-labels", type=int, default=2000)
    ap.add_argument("--draw-ok", action="store_true")

    # 統計
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--min-char-count", type=int, default=5)
    ap.add_argument("--oracle-k", type=int, default=5)
    ap.add_argument("--confusion-topn", type=int, default=200)
    ap.add_argument("--confusion-min-count", type=int, default=5)

    args = ap.parse_args()

    coord_csv = Path(args.coord_csv)
    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not coord_csv.exists():
        raise FileNotFoundError(coord_csv)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)

    classes_u = read_classes_txt(Path(args.classes))
    K = len(classes_u)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] device: {device}")

    # image model
    model = timm.create_model(args.model, pretrained=True, num_classes=K)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    tfm, mean, std = build_eval_transform(model, img_size=int(args.img_size))
    print(f"[INFO] eval tfm: LetterboxSquare({int(args.img_size)}) -> ToTensor -> Normalize")
    print(f"[INFO] normalize mean/std: mean={mean} std={std}")

    # BERT
    print("[INFO] loading BERT...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModelForMaskedLM.from_pretrained(
        args.bert_model,
    )
    bert_model.to(device)
    bert_model.eval()
    print(f"[INFO] bert_model: {args.bert_model}")

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

    topk = max(1, int(args.topk))
    bert_passes = max(1, int(args.bert_passes))

    # ----------------------------
    # まず画像モデル推論を 1 回だけやってキャッシュ
    # ----------------------------
    page_cache = {}
    pages_skipped_len = 0
    pages_skipped_img = 0
    demo_page = args.demo_page.strip() or (use_pages[0] if use_pages else "")

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
            bbox_list = []
            for i in range(len(d)):
                r = d.iloc[i]
                x = float(r["X"]); y = float(r["Y"])
                w = float(r["Width"]); h = float(r["Height"])
                xx1, yy1, xx2, yy2 = clamp_bbox(x, y, x + w, y + h, W, H)
                crop = im.crop((xx1, yy1, xx2, yy2))
                crops.append(tfm(crop))
                bbox_list.append((xx1, yy1, xx2, yy2))

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
                    top_ids = topi.detach().cpu().numpy().tolist()
                    top_lp = topv.detach().cpu().numpy().tolist()

                    uni_true = str(d.iloc[s + bi]["Unicode"])
                    items.append(
                        Item(
                            page=page,
                            uni_true=uni_true,
                            bbox=tuple(map(int, bbox_list[s + bi])),
                            topk_ids=top_ids,
                            topk_logp=top_lp,
                        )
                    )

            greedy_chars = [uplus_to_char(classes_u[it.topk_ids[0]]) for it in items]

            page_cache[page] = {
                "items": items,
                "gt_chars": gt_chars,
                "greedy_chars": greedy_chars,
                "image": im,
            }

    print("\n[INFO] precompute done")
    print(f"[INFO] cached pages                 : {len(page_cache)}")
    print(f"[INFO] pages skipped (len mismatch): {pages_skipped_len}")
    print(f"[INFO] pages skipped (no image)    : {pages_skipped_img}")

    if len(page_cache) == 0:
        print("[ERROR] no valid pages cached.")
        return

    # lambda リスト決定
    if args.bert_lambda is not None:
        lambda_list = [float(args.bert_lambda)]
    else:
        lambda_list = build_lambda_list(
            start=float(args.bert_lambda_start),
            stop=float(args.bert_lambda_stop),
            step=float(args.bert_lambda_step),
        )

    print(f"[INFO] bert lambda sweep: {lambda_list}")

    summary_rows = []

    for bert_lambda in lambda_list:
        out_dir = out_root / format_lambda_dirname(bert_lambda)
        out_dir.mkdir(parents=True, exist_ok=True)

        overlay_dir = out_dir / "overlays"
        if args.annotate_all:
            overlay_dir.mkdir(parents=True, exist_ok=True)

        total_chars = 0
        correct_greedy = 0
        correct_bert = 0
        pages_done = 0
        demo_result = None

        counts = defaultdict(lambda: defaultdict(int))
        overall = defaultdict(int)

        conf_g = Counter()
        conf_b = Counter()

        oracle_k = max(1, int(args.oracle_k))

        for page in page_cache.keys():
            items = page_cache[page]["items"]
            gt_chars = page_cache[page]["gt_chars"]
            greedy_chars = page_cache[page]["greedy_chars"]
            im = page_cache[page]["image"]

            bert_chars, bert_scores = bert_iterative_rerank(
                items=items,
                classes_u=classes_u,
                tokenizer=tokenizer,
                bert_model=bert_model,
                device=device,
                lam=bert_lambda,
                num_passes=bert_passes,
            )

            g_ok = sum(1 for a, b in zip(greedy_chars, gt_chars) if a == b)
            b_ok = sum(1 for a, b in zip(bert_chars, gt_chars) if a == b)

            total_chars += len(gt_chars)
            correct_greedy += g_ok
            correct_bert += b_ok
            pages_done += 1

            if args.stats:
                for idx, (gt_ch, g_ch, b_ch) in enumerate(zip(gt_chars, greedy_chars, bert_chars)):
                    overall["total"] += 1
                    counts[gt_ch]["total"] += 1

                    k_orc = min(oracle_k, len(items[idx].topk_ids))
                    top_chars_orc = [uplus_to_char(classes_u[cid]) for cid in items[idx].topk_ids[:k_orc]]
                    in_orc = (gt_ch in top_chars_orc)
                    if in_orc:
                        overall["oracle_in_topK"] += 1
                        counts[gt_ch]["oracle_in_topK"] += 1

                    g_ok1 = (g_ch == gt_ch)
                    b_ok1 = (b_ch == gt_ch)

                    conf_g[(gt_ch, g_ch)] += 1
                    conf_b[(gt_ch, b_ch)] += 1

                    if g_ok1 and b_ok1:
                        overall["ok"] += 1
                        counts[gt_ch]["ok"] += 1
                    elif (not g_ok1) and b_ok1:
                        overall["fix"] += 1
                        counts[gt_ch]["fix"] += 1
                        counts[gt_ch]["greedy_err"] += 1
                    elif g_ok1 and (not b_ok1):
                        overall["worse"] += 1
                        counts[gt_ch]["worse"] += 1
                        counts[gt_ch]["bert_err"] += 1
                    else:
                        overall["bothx"] += 1
                        counts[gt_ch]["bothx"] += 1
                        counts[gt_ch]["greedy_err"] += 1
                        counts[gt_ch]["bert_err"] += 1

            if args.annotate_all:
                out_img = overlay_dir / f"overlay_{page}.png"
                draw_overlay_one_page(
                    page_img=im,
                    items=items,
                    gt_chars=gt_chars,
                    greedy_chars=greedy_chars,
                    bert_chars=bert_chars,
                    out_path=out_img,
                    font_path=args.font_path,
                    font_size=int(args.font_size),
                    box_width=int(args.box_width),
                    max_labels=int(args.max_labels),
                    draw_ok=bool(args.draw_ok),
                )

            if page == demo_page:
                demo_result = (page, gt_chars, greedy_chars, bert_chars)

        print("\n===== OVERALL =====")
        print(f"[bert_lambda] {bert_lambda:.1f}")
        print(f"pages evaluated : {pages_done}")

        if total_chars == 0:
            print("[ERROR] evaluated chars = 0. GT長不一致 / 画像なし / gt不足 の可能性が高いです。")
            continue

        greedy_acc = correct_greedy / total_chars
        bert_acc = correct_bert / total_chars

        print(f"Greedy acc   : {greedy_acc:.4f}")
        print(f"Greedy+BERT  : {bert_acc:.4f}")

        summary_rows.append({
            "bert_lambda": bert_lambda,
            "pages_evaluated": pages_done,
            "total_chars": total_chars,
            "greedy_acc": greedy_acc,
            "bert_acc": bert_acc,
        })

        summary_csv = out_dir / "summary.csv"
        pd.DataFrame([summary_rows[-1]]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote: {summary_csv}")

        if args.stats:
            write_stats_and_confusions(
                out_dir=out_dir,
                overall=overall,
                counts=counts,
                conf_g=conf_g,
                conf_b=conf_b,
                oracle_k=oracle_k,
                min_char_count=int(args.min_char_count),
                confusion_topn=int(args.confusion_topn),
                confusion_min_count=int(args.confusion_min_count),
            )

        if demo_result is not None:
            page, gt, g, b = demo_result

            def chunk(xs, n=80):
                s = "".join(xs)
                return "\n".join(s[i:i+n] for i in range(0, len(s), n))

            print("\n===== DEMO PAGE =====")
            print(f"[bert_lambda] {bert_lambda:.1f}")
            print(f"[page] {page}")
            print("GT      :")
            print(chunk(gt))
            print("\nGreedy  :")
            print(chunk(g))
            print("\nBERT    :")
            print(chunk(b))

    if summary_rows:
        summary_all = pd.DataFrame(summary_rows)
        summary_all.to_csv(out_root / "lambda_sweep_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n[INFO] wrote: {out_root / 'lambda_sweep_summary.csv'}")


if __name__ == "__main__":
    main()





"""
PYTHONNOUSERSITE=1 python eval_bert_val.py \
  --coord-csv /home/ihpc/Documents/saito/KODAI/full/100249537/100249537_coordinate.csv \
  --images-dir /home/ihpc/Documents/saito/KODAI/full/100249537/images \
  --gt-dir /home/ihpc/Documents/saito/KODAI2/gt_pages/100249537 \
  --ckpt ./runs/kodai_effb0_pretrained/best.pth \
  --classes ./runs/kodai_effb0_pretrained/classes.txt \
  --model efficientnet_b0 \
  --img-size 224 \
  --topk 5 \
  --bert-model /home/ihpc/Documents/saito/KODAI2/runs/train_bert_kodai_epoch50/final \
  --bert-passes 1 \
  --annotate-all \
  --stats \
  --oracle-k 5 \
  --confusion-topn 200 \
  --confusion-min-count 5 \
  --min-char-count 10 \
  --font-path /home/ihpc/Documents/saito/KODAI/fonts/IPAexGothic.ttf \
  --box-width 4 \
  --out-dir ./outputs/val/bert/50


PYTHONNOUSERSITE=1 python eval_bert_val.py \
  --coord-csv /home/ihpc/Documents/saito/KODAI/full/100249537/100249537_coordinate.csv \
  --images-dir /home/ihpc/Documents/saito/KODAI/full/100249537/images \
  --gt-dir /home/ihpc/Documents/saito/KODAI2/gt_pages/100249537 \
  --ckpt ./runs/kodai_effb0_pretrained/best.pth \
  --classes ./runs/kodai_effb0_pretrained/classes.txt \
  --model efficientnet_b0 \
  --img-size 224 \
  --topk 5 \
  --bert-model /home/ihpc/Documents/saito/KODAI2/runs/train_bert_kodai_epoch3/final \
  --bert-passes 1 \
  --annotate-all \
  --stats \
  --oracle-k 5 \
  --confusion-topn 200 \
  --confusion-min-count 5 \
  --min-char-count 10 \
  --font-path /home/ihpc/Documents/saito/KODAI/fonts/IPAexGothic.ttf \
  --box-width 4 \
  --out-dir ./outputs/val/bert/3

  

  PYTHONNOUSERSITE=1 python eval_bert_val.py \
  --coord-csv /home/ihpc/Documents/saito/KODAI/full/100249537/100249537_coordinate.csv \
  --images-dir /home/ihpc/Documents/saito/KODAI/full/100249537/images \
  --gt-dir /home/ihpc/Documents/saito/KODAI2/gt_pages/100249537 \
  --ckpt ./runs/kodai_effb0_pretrained/best.pth \
  --classes ./runs/kodai_effb0_pretrained/classes.txt \
  --model efficientnet_b0 \
  --img-size 224 \
  --topk 5 \
  --bert-model tohoku-nlp/bert-base-japanese-char-v3 \
  --bert-passes 1 \
  --annotate-all \
  --stats \
  --oracle-k 5 \
  --confusion-topn 200 \
  --confusion-min-count 5 \
  --min-char-count 10 \
  --font-path /home/ihpc/Documents/saito/KODAI/fonts/IPAexGothic.ttf \
  --box-width 4 \
  --out-dir ./outputs/val/bert/no
"""