#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations



import argparse
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

try:
    import timm
except Exception as e:
    raise SystemExit(
        "timm が必要です。環境に無いなら: pip install timm\n"
        f"ImportError: {e}"
    )


# ----------------------------
# Shared-like transform pieces (match training val_tf style)
# ----------------------------
class LetterboxSquare:
    """
    Keep aspect ratio, pad to square with white.
    Input: PIL.Image
    Output: PIL.Image (size, size)
    """
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
    """
    “XYcut風”の簡易列分け:
    - bbox中心 cx を右→左に並べ
    - cx のギャップで列IDを増やす
    - 列ID(右から0,1,2,...) → 各列は cy の昇順（上→下）
    """
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
# nanoGPT LM wrapper
# ----------------------------
class NanoGPTLM:
    """
    Minimal scorer for nanoGPT (karpathy/nanoGPT style model.py).

    We only need:
      - forward(idx) -> logits over vocab for each position
      - take last position logits to get next-token distribution

    This is NOT kv-cache optimized, but it is beam-efficient:
      per timestep we do forward only for each beam prefix once,
      then score all candidate next tokens by indexing the last logprob vector.
    """
    def __init__(self, ckpt_path: Path, meta_path: Path, device: torch.device, nanogpt_root: Optional[Path] = None):
        self.ckpt_path = ckpt_path
        self.meta_path = meta_path
        self.device = device

        if nanogpt_root is not None:
            # allow importing nanoGPT/model.py from arbitrary location
            sys.path.insert(0, str(nanogpt_root.resolve()))

        # Import GPT/GPTConfig from nanoGPT repo
        try:
            from model import GPT, GPTConfig  # type: ignore
        except Exception as e:
            raise SystemExit(
                "Failed to import nanoGPT model.py (GPT/GPTConfig).\n"
                "Fix: run this script from nanoGPT repo root, or pass --nanogpt-root /path/to/nanoGPT.\n"
                f"ImportError: {e}"
            )

        # Load meta.pkl (stoi/itos, block_size, unk_char, dtype)
        meta = pickle.load(open(meta_path, "rb"))
        self.stoi: Dict[str, int] = meta["stoi"]
        self.itos: Dict[int, str] = meta["itos"]
        self.vocab_size: int = int(meta["vocab_size"])
        self.block_size: int = int(meta.get("block_size", 256))
        self.unk_char: str = meta.get("unk_char", "\uFFFD")
        if len(self.unk_char) != 1:
            self.unk_char = "\uFFFD"
        self.unk_id: int = int(self.stoi.get(self.unk_char, 0))

        # Load checkpoint
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        # nanoGPT standard: ckpt contains 'model_args' and 'model' or 'state_dict'
        model_args = ckpt.get("model_args", None)
        state_dict = ckpt.get("model", None) or ckpt.get("state_dict", None) or ckpt

        if model_args is None:
            # best-effort fallback: infer minimal config
            # (this can fail if your ckpt doesn't follow nanoGPT layout)
            raise SystemExit(
                "Checkpoint does not contain 'model_args'.\n"
                "Your ckpt format may differ. Save ckpt in nanoGPT standard format "
                "(with model_args + model/state_dict)."
            )

        # Ensure vocab_size/block_size are consistent with meta
        model_args = dict(model_args)
        model_args["vocab_size"] = self.vocab_size
        model_args["block_size"] = self.block_size

        cfg = GPTConfig(**model_args)
        self.model = GPT(cfg)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

    def char_to_id(self, ch: str) -> int:
        return int(self.stoi.get(ch, self.unk_id))

    @torch.no_grad()
    def next_logprobs(self, prefix_ids: List[int]) -> torch.Tensor:
        """
        Returns logprobs (ln) over vocab for the next token given prefix.
        If prefix is empty, we feed a single unk (or newline if present) to avoid T=0.
        """
        if len(prefix_ids) == 0:
            # choose a stable seed token; prefer newline if present
            seed = self.stoi.get("\n", self.unk_id)
            idx = torch.tensor([[seed]], dtype=torch.long, device=self.device)
        else:
            # truncate to block_size
            ctx = prefix_ids[-self.block_size:]
            idx = torch.tensor([ctx], dtype=torch.long, device=self.device)

        logits, _ = self.model(idx)  # (B,T,V), loss
        last = logits[0, -1, :]      # (V,)
        logp = torch.log_softmax(last, dim=-1)
        return logp


def beam_search_char_nanogpt(
    items: List[Item],
    lm: NanoGPTLM,
    classes_u: List[str],
    beam_size: int = 20,
    lam: float = 0.3,
) -> List[str]:
    """
    Beam search with image logp + lam * LM logp (both ln).
    Efficient part: per timestep, forward only per beam prefix once,
    then score topK candidates by indexing the logprob vector.
    """
    # beam: (seq_chars, score, prefix_token_ids)
    beams: List[Tuple[List[str], float, List[int]]] = [([], 0.0, [])]

    for it in items:
        new_beams: List[Tuple[List[str], float, List[int]]] = []

        # For each current beam, compute LM distribution once
        for seq_chars, score, prefix_ids in beams:
            lm_logp_vec = lm.next_logprobs(prefix_ids)  # (V,)

            # Expand by topK vision candidates
            for cid, img_lp in zip(it.topk_ids, it.topk_logp):
                ch_u = classes_u[cid]
                ch = uplus_to_char(ch_u)

                tid = lm.char_to_id(ch)
                lm_lp = float(lm_logp_vec[tid].item())

                s = score + float(img_lp) + lam * lm_lp
                new_beams.append((seq_chars + [ch], s, prefix_ids + [tid]))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    return beams[0][0]


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
    lm_chars: List[str],
    out_path: Path,
    font_path: str,
    font_size: int,
    box_width: int,
    max_labels: int,
    draw_ok: bool,
) -> None:
    """
    - bbox枠（太さ box_width）
    - 3文字: Greedy/GT/LM（ラベル文字は付けない）
    色：
    - FIX   (g!=gt and lm==gt) : 緑
    - WORSE (g==gt and lm!=gt) : 赤
    - BOTHx (g!=gt and lm!=gt) : 黄
    - OK    (g==gt and lm==gt) : draw_ok=Trueなら薄灰
    """
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
        lmch = lm_chars[i]

        g_ok = (g == gt)
        lm_ok = (lmch == gt)

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

        label = f"{g}{gt}{lmch}"

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
    """
    評価transform（学習val_tf相当）:
      LetterboxSquare(img_size) -> ToTensor -> Normalize(mean/std)
    mean/std は model.default_cfg を優先。
    """
    default_cfg = getattr(model, "default_cfg", {}) or {}
    mean = tuple(default_cfg.get("mean", (0.485, 0.456, 0.406)))
    std  = tuple(default_cfg.get("std",  (0.229, 0.224, 0.225)))

    tfm = transforms.Compose([
        LetterboxSquare(img_size, fill=255, interp=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm, mean, std


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--coord-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--gt-dir", required=True)

    # vision model
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)

    # transform
    ap.add_argument("--img-size", type=int, default=224,
                    help="input size for eval transform (should match training img_size)")

    # nanoGPT LM
    ap.add_argument("--nano-ckpt", required=True, help="nanoGPT checkpoint (must include model_args)")
    ap.add_argument("--nano-meta", required=True, help="meta.pkl produced by prepare.py (contains stoi/itos)")
    ap.add_argument("--nanogpt-root", default="", help="path to nanoGPT repo root (where model.py exists)")

    # decoding
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam-size", type=int, default=20)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.3)

    ap.add_argument("--pages", default="all", help="'all' or comma-separated list")
    ap.add_argument("--out-dir", default="./eval_out_xycut_nanogpt")
    ap.add_argument("--demo-page", default="", help="show one page output (e.g. 100249371_00010_1)")

    # overlay
    ap.add_argument("--annotate-all", action="store_true",
                    help="評価した全ページについて overlay画像を出力する")
    ap.add_argument("--font-path", default="",
                    help="描画用フォント(TTF/OTF)。例: /home/ihpc/Documents/saito/KODAI/fonts/IPAexGothic.ttf")
    ap.add_argument("--font-size", type=int, default=18, help="描画フォントサイズ")
    ap.add_argument("--box-width", type=int, default=4, help="bbox枠の太さ")
    ap.add_argument("--max-labels", type=int, default=2000,
                    help="各ページの描画bbox上限（多すぎると見づらい）")
    ap.add_argument("--draw-ok", action="store_true",
                    help="OK（GreedyもLMも正解）も描画する（デフォルトは誤りのみ描画）")

    # stats
    ap.add_argument("--stats", action="store_true",
                    help="GT文字ごとの誤り統計（FIX/WORSE等）を集計してCSV出力する")
    ap.add_argument("--min-char-count", type=int, default=5,
                    help="出現回数が少ない文字を統計から除外する閾値（ノイズ対策）")

    # Oracle@K / confusion
    ap.add_argument("--oracle-k", type=int, default=5,
                    help="Oracle@K（GTがTopKに入っている率）のK")
    ap.add_argument("--confusion-topn", type=int, default=200,
                    help="混同行列（GT→予測）の上位ペアを何件出すか")
    ap.add_argument("--confusion-min-count", type=int, default=5,
                    help="混同行列のペア頻度がこの値未満は出力しない（ノイズ対策）")

    args = ap.parse_args()

    coord_csv = Path(args.coord_csv)
    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # ----------------------------
    # Load vision model (EfficientNet etc.)
    # ----------------------------
    model = timm.create_model(args.model, pretrained=True, num_classes=K)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    tfm, mean, std = build_eval_transform(model, img_size=int(args.img_size))
    print(f"[INFO] eval tfm: LetterboxSquare({int(args.img_size)}) -> ToTensor -> Normalize")
    print(f"[INFO] normalize mean/std: mean={mean} std={std}")

    # ----------------------------
    # Load nanoGPT LM
    # ----------------------------
    nano_ckpt = Path(args.nano_ckpt)
    nano_meta = Path(args.nano_meta)
    if not nano_ckpt.exists():
        raise FileNotFoundError(nano_ckpt)
    if not nano_meta.exists():
        raise FileNotFoundError(nano_meta)

    nanogpt_root = Path(args.nanogpt_root).resolve() if args.nanogpt_root.strip() else None
    if nanogpt_root is not None and not (nanogpt_root / "model.py").exists():
        raise FileNotFoundError(f"--nanogpt-root has no model.py: {nanogpt_root}")

    print("[INFO] loading nanoGPT LM...")
    lm = NanoGPTLM(
        ckpt_path=nano_ckpt,
        meta_path=nano_meta,
        device=device,
        nanogpt_root=nanogpt_root
    )
    print(f"[INFO] nanoGPT vocab_size={lm.vocab_size} block_size={lm.block_size} unk_char={repr(lm.unk_char)}")

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

    # GTがあるページだけ
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

    # ---- 統計（GT文字ごと）----
    counts = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)  # total/ok/fix/worse/bothx/oracle_in_topK

    # ★混同行列（ペア頻度）
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

            # crops
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

            # inference
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

            lm_chars = beam_search_char_nanogpt(
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

            # ---- 統計・Oracle・混同行列 ----
            if args.stats:
                for idx, (gt_ch, g_ch, lm_ch) in enumerate(zip(gt_chars, greedy_chars, lm_chars)):
                    overall["total"] += 1
                    counts[gt_ch]["total"] += 1

                    # Oracle@K: GTがTopK候補に入っているか
                    k_orc = min(oracle_k, len(items[idx].topk_ids))
                    top_chars_orc = [uplus_to_char(classes_u[cid]) for cid in items[idx].topk_ids[:k_orc]]
                    in_orc = (gt_ch in top_chars_orc)
                    if in_orc:
                        overall["oracle_in_topK"] += 1
                        counts[gt_ch]["oracle_in_topK"] += 1

                    g_ok1 = (g_ch == gt_ch)
                    lm_ok1 = (lm_ch == gt_ch)

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

            # ---- overlay ----
            if args.annotate_all:
                out_img = overlay_dir / f"overlay_{page}.png"
                draw_overlay_one_page(
                    page_img=im,
                    items=items,
                    gt_chars=gt_chars,
                    greedy_chars=greedy_chars,
                    lm_chars=lm_chars,
                    out_path=out_img,
                    font_path=args.font_path,
                    font_size=int(args.font_size),
                    box_width=int(args.box_width),
                    max_labels=int(args.max_labels),
                    draw_ok=bool(args.draw_ok),
                )

            if page == demo_page:
                demo_result = (page, gt_chars, greedy_chars, lm_chars)

    print("\n===== OVERALL =====")
    print(f"pages evaluated : {pages_done}")
    print(f"pages skipped (len mismatch) : {pages_skipped_len}")
    print(f"pages skipped (no image)     : {pages_skipped_img}")

    if total_chars == 0:
        print("[ERROR] evaluated chars = 0. ほぼ確実に GT長さ不一致 / 画像なし / gt_pages不足 です。")
        return

    print(f"Greedy acc    : {correct_greedy / total_chars:.4f}")
    print(f"Greedy+LM acc : {correct_lm / total_chars:.4f}")

    # ---- 統計の書き出し ----
    if args.stats:
        def ratio(x, d):
            return (x / d) if d > 0 else 0.0

        ok_r = ratio(overall["ok"], overall["total"])
        fix_r = ratio(overall["fix"], overall["total"])
        worse_r = ratio(overall["worse"], overall["total"])
        bothx_r = ratio(overall["bothx"], overall["total"])
        oracle_r = ratio(overall["oracle_in_topK"], overall["total"])

        g_err_total = overall["fix"] + overall["bothx"]  # Greedyが間違った総数
        repair_r = ratio(overall["fix"], g_err_total)    # Greedy誤りのうち修復できた割合

        overall_txt = out_dir / "overall_stats.txt"
        with overall_txt.open("w", encoding="utf-8") as f:
            f.write("=== OVERALL CHAR STATS ===\n")
            f.write(f"total={overall['total']}\n")
            f.write(f"ok={overall['ok']}  ({ok_r:.6f})\n")
            f.write(f"fix={overall['fix']}  ({fix_r:.6f})\n")
            f.write(f"worse={overall['worse']}  ({worse_r:.6f})\n")
            f.write(f"bothx={overall['bothx']}  ({bothx_r:.6f})\n")
            f.write("\n")
            f.write(f"oracle_k={oracle_k}\n")
            f.write(f"oracle_in_topK={overall['oracle_in_topK']}  ({oracle_r:.6f})\n")
            f.write("\n")
            f.write(f"greedy_error_total={g_err_total}\n")
            f.write(f"repair_rate = fix / (fix+bothx) = {repair_r:.6f}\n")

        print(f"[INFO] wrote: {overall_txt}")

        # ---- 文字ごと統計CSV（Oracle追加）----
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
            lm_err = int(dct["lm_err"])
            orc = int(dct["oracle_in_topK"])

            greedy_miss = ratio(g_err, total)
            lm_miss = ratio(lm_err, total)

            fix_rate_total = ratio(fix, total)
            worse_rate_total = ratio(worse, total)
            bothx_rate_total = ratio(bothx, total)

            repair_rate_char = ratio(fix, fix + bothx)
            oracle_rate_char = ratio(orc, total)

            rows.append({
                "char_gt": ch,
                "count": total,
                "ok": ok,
                "fix": fix,
                "worse": worse,
                "bothx": bothx,
                "greedy_err": g_err,
                "lm_err": lm_err,
                "oracle_in_topK": orc,
                "greedy_miss_rate": greedy_miss,
                "lm_miss_rate": lm_miss,
                "oracle_rate": oracle_rate_char,
                "fix_rate_total": fix_rate_total,
                "worse_rate_total": worse_rate_total,
                "bothx_rate_total": bothx_rate_total,
                "repair_rate_char": repair_rate_char,
            })

        df_stats = pd.DataFrame(rows)
        if len(df_stats) > 0:
            df_stats = df_stats.sort_values(["count", "greedy_miss_rate"], ascending=[False, False])

        out_csv = out_dir / "char_stats.csv"
        df_stats.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote: {out_csv}  (rows={len(df_stats)})")

        # ---- 混同行列（上位ペア）----
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
        dump_confusion(conf_lm, "lm")

    if demo_result is not None:
        page, gt, g, lm_out = demo_result

        def chunk(xs, n=80):
            s = "".join(xs)
            return "\n".join(s[i:i+n] for i in range(0, len(s), n))

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
python eval_nanogpt.py \
  --coord-csv /home/ihpc/Documents/saito/KODAI/full/200018243/200018243_coordinate.csv \
  --images-dir /home/ihpc/Documents/saito/KODAI/full/200018243/images \
  --gt-dir /home/ihpc/Documents/saito/KODAI2/gt_pages/200018243 \
  --nano-ckpt /home/ihpc/Documents/saito/KODAI2/nanoGPT/out_kodai_char_main/ckpt.pt \
  --nano-meta /home/ihpc/Documents/saito/KODAI2/nanoGPT/data/kodai_char/meta.pkl \
  --nanogpt-root /home/ihpc/Documents/saito/KODAI2/nanoGPT \
  --ckpt ./runs/kodai_effb0_pretrained/best.pth \
  --classes ./runs/kodai_effb0_pretrained/classes.txt \
  --model efficientnet_b0 \
  --img-size 224 \
  --lambda 0.6 \
  --topk 5 \
  --beam-size 5 \
  --annotate-all \
  --stats \
  --oracle-k 5 \
  --confusion-topn 200 \
  --confusion-min-count 5 \
  --min-char-count 10 \
  --font-path /home/ihpc/Documents/saito/KODAI/fonts/IPAexGothic.ttf \
  --box-width 4 \
  --out-dir ./outputs/effib0_200018243_nanogpt_0.6
"""