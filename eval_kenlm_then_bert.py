#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
eval_kenlm_then_bert.py

目的:
1) coord_csv + images_dir から各 bbox crop を作り、timm model で topK 候補を出す
2) 同じページ内の bbox を「右列→左列、各列は上→下」で並べ替える
3) gt_pages/{page}.txt を GT として読み込み、同じ順で比較
4) Stage-1: KenLM(arpa) を使って beam search で系列復元する
5) Stage-2: KenLM出力列を初期値として BERT(char MLM) で iterative masked rerank する
6) ページ単位 / 全体の精度を出す
7) overlay / char_stats / confusion を出す

重要:
- これは image+kenlm+bert の同時融合ではない
- 直列:
    OCR topK -> KenLM beam -> BERT rerank
- BERT段では KenLMスコアは使わない
- BERT段の fused は:
    fused = image_logp + bert_lambda * bert_score
  ただし初期列は greedy ではなく KenLM 出力
"""

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
    import kenlm  # type: ignore
except Exception as e:
    raise SystemExit(
        "python kenlm が必要です: pip install kenlm\n"
        "※カレントに kenlm.py があると壊れます（import shadow）。\n"
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
# Shared transform pieces
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
        top  = (self.size - nh) // 2
        canvas.paste(img_r, (left, top))
        return canvas


def build_eval_transform(model, img_size: int):
    default_cfg = getattr(model, "default_cfg", {}) or {}
    mean = tuple(default_cfg.get("mean", (0.485, 0.456, 0.406)))
    std  = tuple(default_cfg.get("std",  (0.229, 0.224, 0.225)))

    tfm = transforms.Compose([
        LetterboxSquare(img_size, fill=255, interp=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm, mean, std


# ----------------------------
# Utility
# ----------------------------
def read_classes_txt(p: Path) -> List[str]:
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
# KenLM stage
# ----------------------------
def beam_search_char_lm(
    items: List[Item],
    lm: kenlm.Model,
    classes_u: List[str],
    beam_size: int = 20,
    lam: float = 0.3,
) -> List[str]:
    """
    Stage-1:
      OCR topK candidates -> KenLM beam search -> best char sequence

    注意:
    - image側は ln(softmax)
    - kenlm BaseScore は log10
    よって lam はスケール合わせ用の実験パラメータ
    """
    state0 = kenlm.State()
    lm.BeginSentenceWrite(state0)

    beams = [([], 0.0, state0)]  # (seq_chars, score, state)

    for it in items:
        new_beams = []
        for seq, score, st in beams:
            for cid, lp in zip(it.topk_ids, it.topk_logp):
                ch_u = classes_u[cid]
                ch = uplus_to_char(ch_u)
                out_st = kenlm.State()
                lm_score = lm.BaseScore(st, ch, out_st)  # log10
                s = score + float(lp) + lam * float(lm_score)
                new_beams.append((seq + [ch], s, out_st))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    return beams[0][0]


# ----------------------------
# BERT stage
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
    - 1文字=1token と決め打ちしない
    - multi-token の場合は mask を複数並べて各位置の log-prob を和
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


def build_item_logp_map(it: Item, classes_u: List[str]) -> dict[str, float]:
    """
    各位置の候補 char -> image_logp の辞書を作る。
    同じ文字が複数候補に出るケースは、より高い logp を採用。
    """
    mp: dict[str, float] = {}
    for cid, lp in zip(it.topk_ids, it.topk_logp):
        ch = uplus_to_char(classes_u[cid])
        lp = float(lp)
        if (ch not in mp) or (lp > mp[ch]):
            mp[ch] = lp
    return mp


@torch.no_grad()
def bert_rerank_from_initial_sequence(
    items: List[Item],
    classes_u: List[str],
    tokenizer,
    bert_model,
    device: torch.device,
    init_chars: List[str],
    lam: float = 1.0,
    num_passes: int = 1,
    freeze_if_not_in_topk: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Stage-2:
      KenLM出力列を初期値として iterative masked rerank を行う。

    fused_score = image_logp + lam * bert_score

    ポイント:
    - 候補集合は各位置の OCR topK に制限する
    - ただし freeze_if_not_in_topk=True のとき、
      KenLM初期文字が topK に無ければその位置は凍結する
      （無理に別候補へ寄せる暴走を避けるため）
    """
    if not items:
        return [], []

    if len(init_chars) != len(items):
        raise ValueError(f"len(init_chars)={len(init_chars)} != len(items)={len(items)}")

    cur_chars = init_chars[:]
    final_scores = [0.0] * len(items)

    item_char2logp = [build_item_logp_map(it, classes_u) for it in items]

    for _ in range(max(1, int(num_passes))):
        new_chars = cur_chars[:]
        new_scores = final_scores[:]

        for i, it in enumerate(items):
            cand_map = item_char2logp[i]
            cand_chars = list(cand_map.keys())

            # KenLM初期文字がtopK候補に無い場合の扱い
            if cur_chars[i] not in cand_map and freeze_if_not_in_topk:
                new_chars[i] = cur_chars[i]
                new_scores[i] = -1e9
                continue

            # BERT段は OCR topK 候補のみで再評価
            best_char = cur_chars[i]
            best_score = -1e18

            for cand_ch in cand_chars:
                img_lp = float(cand_map[cand_ch])

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
    kenlm_chars: List[str],
    final_chars: List[str],
    out_path: Path,
    font_path: str,
    font_size: int,
    box_width: int,
    max_labels: int,
    draw_ok: bool,
) -> None:
    """
    4文字: Greedy / GT / KenLM / Final(KenLM->BERT)
    色:
    - FIX   : Greedy誤り -> Final正解 = 緑
    - WORSE : Greedy正解 -> Final誤り = 赤
    - BOTHx : 両方誤り = 黄
    - OK    : draw_ok=True のとき薄灰
    """
    im = page_img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _load_font(font_path, font_size)

    n = min(len(items), len(gt_chars), len(greedy_chars), len(kenlm_chars), len(final_chars), max_labels)

    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    OKC = (180, 180, 180)

    shown = 0
    for i in range(n):
        x1, y1, x2, y2 = items[i].bbox
        gt = gt_chars[i]
        g = greedy_chars[i]
        k = kenlm_chars[i]
        f = final_chars[i]

        g_ok = (g == gt)
        f_ok = (f == gt)

        if g_ok and f_ok:
            if not draw_ok:
                continue
            color = OKC
        elif (not g_ok) and f_ok:
            color = GREEN
        elif g_ok and (not f_ok):
            color = RED
        else:
            color = YELLOW

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        label = f"{g}{gt}{k}{f}"

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


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--coord-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--gt-dir", required=True)

    ap.add_argument("--arpa", required=True)

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=5)

    # KenLM stage
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--kenlm-lambda", type=float, default=2.0)

    # BERT stage
    ap.add_argument("--bert-model", required=True,
                    help="HF model name or local path")
    ap.add_argument("--bert-lambda", type=float, default=2.0,
                    help="fused_score = image_logp + bert_lambda * bert_score")
    ap.add_argument("--bert-passes", type=int, default=1,
                    help="iterative rerank passes after KenLM output")
    ap.add_argument("--freeze-if-not-in-topk", action="store_true",
                    help="KenLM出力文字がOCR topKに無い位置をBERT段で凍結する")

    ap.add_argument("--pages", default="all", help="'all' or comma-separated list")
    ap.add_argument("--out-dir", default="./eval_out_kenlm_then_bert")
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
        use_safetensors=True,
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
    beam_size = max(1, int(args.beam_size))
    kenlm_lambda = float(args.kenlm_lambda)
    bert_lambda = float(args.bert_lambda)
    bert_passes = max(1, int(args.bert_passes))

    total_chars = 0
    correct_greedy = 0
    correct_kenlm = 0
    correct_final = 0
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
    conf_k = Counter()
    conf_f = Counter()

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
                x = float(r["X"]); y = float(r["Y"])
                w = float(r["Width"]); h = float(r["Height"])
                xx1, yy1, xx2, yy2 = clamp_bbox(x, y, x + w, y + h, W, H)
                crop = im.crop((xx1, yy1, xx2, yy2))
                crops.append(tfm(crop))

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
                            bbox=tuple(map(int, boxes[s + bi].tolist())),
                            topk_ids=top_ids,
                            topk_logp=top_lp,
                        )
                    )

            greedy_chars = [uplus_to_char(classes_u[it.topk_ids[0]]) for it in items]

            # Stage-1: KenLM
            kenlm_chars = beam_search_char_lm(
                items=items,
                lm=lm,
                classes_u=classes_u,
                beam_size=beam_size,
                lam=kenlm_lambda,
            )

            # Stage-2: BERT from KenLM output
            final_chars, final_scores = bert_rerank_from_initial_sequence(
                items=items,
                classes_u=classes_u,
                tokenizer=tokenizer,
                bert_model=bert_model,
                device=device,
                init_chars=kenlm_chars,
                lam=bert_lambda,
                num_passes=bert_passes,
                freeze_if_not_in_topk=bool(args.freeze_if_not_in_topk),
            )

            g_ok = sum(1 for a, b in zip(greedy_chars, gt_chars) if a == b)
            k_ok = sum(1 for a, b in zip(kenlm_chars, gt_chars) if a == b)
            f_ok = sum(1 for a, b in zip(final_chars, gt_chars) if a == b)

            total_chars += len(gt_chars)
            correct_greedy += g_ok
            correct_kenlm += k_ok
            correct_final += f_ok
            pages_done += 1

            if args.stats:
                for idx, (gt_ch, g_ch, k_ch, f_ch) in enumerate(zip(gt_chars, greedy_chars, kenlm_chars, final_chars)):
                    overall["total"] += 1
                    counts[gt_ch]["total"] += 1

                    k_orc = min(oracle_k, len(items[idx].topk_ids))
                    top_chars_orc = [uplus_to_char(classes_u[cid]) for cid in items[idx].topk_ids[:k_orc]]
                    in_orc = (gt_ch in top_chars_orc)
                    if in_orc:
                        overall["oracle_in_topK"] += 1
                        counts[gt_ch]["oracle_in_topK"] += 1

                    g_ok1 = (g_ch == gt_ch)
                    k_ok1 = (k_ch == gt_ch)
                    f_ok1 = (f_ch == gt_ch)

                    conf_g[(gt_ch, g_ch)] += 1
                    conf_k[(gt_ch, k_ch)] += 1
                    conf_f[(gt_ch, f_ch)] += 1

                    # Greedy -> Final の改善/悪化を見る
                    if g_ok1 and f_ok1:
                        overall["ok"] += 1
                        counts[gt_ch]["ok"] += 1
                    elif (not g_ok1) and f_ok1:
                        overall["fix"] += 1
                        counts[gt_ch]["fix"] += 1
                        counts[gt_ch]["greedy_err"] += 1
                    elif g_ok1 and (not f_ok1):
                        overall["worse"] += 1
                        counts[gt_ch]["worse"] += 1
                        counts[gt_ch]["final_err"] += 1
                    else:
                        overall["bothx"] += 1
                        counts[gt_ch]["bothx"] += 1
                        counts[gt_ch]["greedy_err"] += 1
                        counts[gt_ch]["final_err"] += 1

                    # 参考として KenLM / Final 単独誤り数も保持
                    if not k_ok1:
                        counts[gt_ch]["kenlm_err"] += 1
                    if not f_ok1:
                        counts[gt_ch]["final_err_total"] += 1

            if args.annotate_all:
                out_img = overlay_dir / f"overlay_{page}.png"
                draw_overlay_one_page(
                    page_img=im,
                    items=items,
                    gt_chars=gt_chars,
                    greedy_chars=greedy_chars,
                    kenlm_chars=kenlm_chars,
                    final_chars=final_chars,
                    out_path=out_img,
                    font_path=args.font_path,
                    font_size=int(args.font_size),
                    box_width=int(args.box_width),
                    max_labels=int(args.max_labels),
                    draw_ok=bool(args.draw_ok),
                )

            if page == demo_page:
                demo_result = (page, gt_chars, greedy_chars, kenlm_chars, final_chars)

    print("\n===== OVERALL =====")
    print(f"pages evaluated : {pages_done}")
    print(f"pages skipped (len mismatch) : {pages_skipped_len}")
    print(f"pages skipped (no image)     : {pages_skipped_img}")

    if total_chars == 0:
        print("[ERROR] evaluated chars = 0. GT長不一致 / 画像なし / gt不足 の可能性が高いです。")
        return

    print(f"Greedy acc        : {correct_greedy / total_chars:.4f}")
    print(f"Greedy+KenLM acc  : {correct_kenlm / total_chars:.4f}")
    print(f"KenLM->BERT acc   : {correct_final / total_chars:.4f}")

    if args.stats:
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
        min_cnt = int(args.min_char_count)
        for ch, dct in counts.items():
            total = int(dct["total"])
            if total < min_cnt:
                continue

            ok = int(dct["ok"])
            fix = int(dct["fix"])
            worse = int(dct["worse"])
            bothx = int(dct["bothx"])
            g_err = int(dct["greedy_err"])
            k_err = int(dct["kenlm_err"])
            f_err = int(dct["final_err_total"])
            orc = int(dct["oracle_in_topK"])

            rows.append({
                "char_gt": ch,
                "count": total,
                "ok": ok,
                "fix": fix,
                "worse": worse,
                "bothx": bothx,
                "greedy_err": g_err,
                "kenlm_err": k_err,
                "final_err": f_err,
                "oracle_in_topK": orc,
                "greedy_miss_rate": g_err / total if total > 0 else 0.0,
                "kenlm_miss_rate": k_err / total if total > 0 else 0.0,
                "final_miss_rate": f_err / total if total > 0 else 0.0,
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

        topn = max(1, int(args.confusion_topn))
        min_pair = max(1, int(args.confusion_min_count))

        def dump_confusion(counter: Counter, name: str):
            rows2 = []
            for (gt, pred), c in counter.most_common():
                if c < min_pair:
                    break
                rows2.append({
                    "gt": gt,
                    "pred": pred,
                    "count": c,
                    "is_correct": int(gt == pred),
                })
                if len(rows2) >= topn:
                    break
            dfc = pd.DataFrame(rows2)
            outp = out_dir / f"confusion_{name}_top{topn}.csv"
            dfc.to_csv(outp, index=False, encoding="utf-8-sig")
            print(f"[INFO] wrote: {outp}  (rows={len(dfc)})")

        dump_confusion(conf_g, "greedy")
        dump_confusion(conf_k, "kenlm")
        dump_confusion(conf_f, "final")

    if demo_result is not None:
        page, gt, g, k, f = demo_result

        def chunk(xs, n=80):
            s = "".join(xs)
            return "\n".join(s[i:i+n] for i in range(0, len(s), n))

        print("\n===== DEMO PAGE =====")
        print(f"[page] {page}")
        print("GT          :")
        print(chunk(gt))
        print("\nGreedy      :")
        print(chunk(g))
        print("\nGreedy+KenLM:")
        print(chunk(k))
        print("\nKenLM->BERT :")
        print(chunk(f))


if __name__ == "__main__":
    main()



"""
PYTHONNOUSERSITE=1 python eval_kenlm_then_bert.py \
  --coord-csv /home/ihpc/Documents/saito/KODAI/full/200017458/200017458_coordinate.csv \
  --images-dir /home/ihpc/Documents/saito/KODAI/full/200017458/images \
  --gt-dir /home/ihpc/Documents/saito/KODAI2/gt_pages/200017458 \
  --arpa /home/ihpc/Documents/saito/KODAI2/handchar_kenlmo3.arpa \
  --ckpt ./runs/kodai_effb0_pretrained/best.pth \
  --classes ./runs/kodai_effb0_pretrained/classes.txt \
  --model efficientnet_b0 \
  --img-size 224 \
  --topk 5 \
  --beam-size 5 \
  --kenlm-lambda 1.6 \
  --bert-model /home/ihpc/Documents/saito/KODAI2/runs/train_bert_kodai_epoch50/final \
  --bert-lambda 0.8 \
  --bert-passes 1 \
  --freeze-if-not-in-topk \
  --annotate-all \
  --stats \
  --oracle-k 5 \
  --confusion-topn 200 \
  --confusion-min-count 5 \
  --min-char-count 10 \
  --font-path /home/ihpc/Documents/saito/KODAI/fonts/IPAexGothic.ttf \
  --box-width 4 \
  --out-dir ./outputs/kenlm_then_bert/200017458_kenlm1.6
"""