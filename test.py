#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations


import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

try:
    import kenlm  # type: ignore
except Exception as e:
    raise SystemExit(
        "python kenlm が必要です: pip install kenlm\n"
        "※カレントに kenlm.py があると import shadow します。\n"
        f"ImportError: {e}"
    )

try:
    import timm
except Exception:
    timm = None


# ----------------------------
# Utils
# ----------------------------
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
    右列→左列、列内は上→下の簡易順序付け。
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


def _load_font(font_path: str, font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"[WARN] failed to load font: {font_path} ({e}) -> fallback default font")
    return ImageFont.load_default()


# ----------------------------
# Letterbox (stack落ち対策の本体)
# ----------------------------
class LetterboxSquare:
    """
    PIL.Image -> PIL.Image (size,size)
    アスペクトを保ってリサイズし、余白を白でパディング。
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


def build_crop_transform(img_size: int):
    """
    cropを必ず同一サイズTensorにする（stack落ち回避）。
    """
    return transforms.Compose([
        LetterboxSquare(img_size, fill=255, interp=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ----------------------------
# PreActResNet-18 (ckpt互換: projection名 = shortcut)
# ----------------------------
class PreActBasicBlock(nn.Module):
    """
    Pre-activation BasicBlock (He et al. identity mapping spirit)

    residual:  BN(in)->ReLU->Conv3x3(stride) -> BN->ReLU->Conv3x3
    shortcut:  形が同じなら identity (x)
               形が違う/stride!=1 なら 1x1 projection を使う
               projection は "pre-activated tensor (BN+ReLU後)" に掛ける

    **ckpt互換のポイント**
    - projection conv の名前を self.shortcut にする
      (あなたのckptが layerX.Y.shortcut.weight で保存されているため)
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))

        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int, in_ch: int = 3, stem: str = "cifar"):
        super().__init__()
        assert stem in ("cifar", "imagenet")

        self.in_planes = 64

        if stem == "cifar":
            # 3x3 stride1 (CIFAR-like)
            self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = None
        else:
            # ImageNet-like 7x7 stride2 + maxpool
            self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.bn = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        if self.maxpool is not None:
            out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def preact_resnet18(num_classes: int, stem: str = "cifar"):
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_ch=3, stem=stem)


# ----------------------------
# Beam search with KenLM
# ----------------------------
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
# Overlay
# ----------------------------
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
    bbox枠 + 3文字（Greedy/GT/LM）を描画
    色:
      FIX   : Greedy誤り→LM正解 = 緑
      WORSE : Greedy正解→LM誤り = 赤
      BOTHx : Greedy誤り かつ LM誤り = 黄
      OK    : draw_ok=True のとき薄灰
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
        lm = lm_chars[i]

        g_ok = (g == gt)
        lm_ok = (lm == gt)

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

        label = f"{g}{gt}{lm}"
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
# Model loader
# ----------------------------
def load_model(args, num_classes: int, device: torch.device) -> nn.Module:
    model_name = args.model.lower()

    if model_name == "preactresnet18":
        model = preact_resnet18(num_classes=num_classes, stem=args.stem).to(device)
    else:
        if timm is None:
            raise SystemExit("timm が import できません。timmモデルを使うなら pip install timm")
        model = timm.create_model(args.model, pretrained=args.timm_pretrained, num_classes=num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # DataParallel 対策
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # strictはデフォルトTrue（壊れたまま進むのを防ぐ）
    try:
        model.load_state_dict(sd, strict=args.strict)
        print(f"[INFO] load_state_dict OK (strict={args.strict})")
    except RuntimeError as e:
        print("[ERROR] load_state_dict failed.")
        print("  -> ckptとモデル定義が一致していません（stem/shortcut名/層構造/num_classes等）")
        raise

    model.eval()
    return model


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--coord-csv", type=str, required=True)
    ap.add_argument("--images-dir", type=str, required=True)
    ap.add_argument("--gt-dir", type=str, required=True)
    ap.add_argument("--arpa", type=str, required=True)

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--classes", type=str, required=True)

    ap.add_argument("--model", type=str, default="preactresnet18",
                    help="preactresnet18 または timm のモデル名（efficientnet_b0等）")
    ap.add_argument("--stem", type=str, default="cifar", choices=["cifar", "imagenet"],
                    help="preactresnet18 の stem。学習時と一致させること。")
    ap.add_argument("--timm-pretrained", action="store_true",
                    help="timmモデルを使う場合に pretrained=True にする（ckptロード前の初期化用）")

    ap.add_argument("--img-size", type=int, default=224, help="cropの固定サイズ（stack落ち回避）")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam-size", type=int, default=20)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.3)

    ap.add_argument("--pages", default="all", help="'all' or comma-separated list")
    ap.add_argument("--out-dir", default="./eval_out_xycut")
    ap.add_argument("--demo-page", default="", help="show one page output (e.g. 100249371_00010_1)")

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

    # Oracle / 混同行列
    ap.add_argument("--oracle-k", type=int, default=5)
    ap.add_argument("--confusion-topn", type=int, default=200)
    ap.add_argument("--confusion-min-count", type=int, default=5)

    # ckptロード
    ap.add_argument("--strict", action="store_true", default=True,
                    help="デフォルトTrue。壊れたロードを防ぐ。Falseにしたいなら --no-strict を使う。")
    ap.add_argument("--no-strict", dest="strict", action="store_false")

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

    # LM
    print("[INFO] loading kenlm...")
    lm = kenlm.Model(str(arpa))
    print(f"[INFO] kenlm order = {lm.order}")

    # classes
    classes_u = read_classes_txt(Path(args.classes))
    K = len(classes_u)
    print(f"[INFO] num_classes = {K}")

    # device
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    device = torch.device(dev)
    print(f"[INFO] device: {device}")

    # model
    model = load_model(args, num_classes=K, device=device)

    # crop transform (固定サイズ化!)
    tfm = build_crop_transform(args.img_size)

    # coord csv
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

    # 統計
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
            bbox_list: List[Tuple[int, int, int, int]] = []
            for i in range(len(d)):
                r = d.iloc[i]
                x = float(r["X"]); y = float(r["Y"])
                w = float(r["Width"]); h = float(r["Height"])
                xx1, yy1, xx2, yy2 = clamp_bbox(x, y, x + w, y + h, W, H)
                crop = im.crop((xx1, yy1, xx2, yy2))
                crops.append(tfm(crop))  # <- ここで固定サイズ化
                bbox_list.append((int(xx1), int(yy1), int(xx2), int(yy2)))

            if not crops:
                continue

            # 推論
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
                            bbox=bbox_list[s + bi],
                            topk_ids=top_ids,
                            topk_logp=top_lp,
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

            # 統計 / 混同行列 / Oracle
            if args.stats:
                for idx, (gt_ch, g_ch, lm_ch) in enumerate(zip(gt_chars, greedy_chars, lm_chars)):
                    overall["total"] += 1
                    counts[gt_ch]["total"] += 1

                    k_orc = min(oracle_k, len(items[idx].topk_ids))
                    top_chars_orc = [uplus_to_char(classes_u[cid]) for cid in items[idx].topk_ids[:k_orc]]
                    in_orc = (gt_ch in top_chars_orc)
                    if in_orc:
                        overall["oracle_in_topK"] += 1
                        counts[gt_ch]["oracle_in_topK"] += 1

                    conf_g[(gt_ch, g_ch)] += 1
                    conf_lm[(gt_ch, lm_ch)] += 1

                    g_ok1 = (g_ch == gt_ch)
                    lm_ok1 = (lm_ch == gt_ch)

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

            # overlay
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
        print("[ERROR] evaluated chars = 0. GT長さ不一致/画像無し/gt_pages不足の可能性が高いです。")
        return

    print(f"Greedy acc    : {correct_greedy / total_chars:.4f}")
    print(f"Greedy+LM acc : {correct_lm / total_chars:.4f}")

    # 統計の書き出し
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
            f.write(f"bothx={overall['bothx']}  ({bothx_r:.6f})\n")
            f.write("\n")
            f.write(f"oracle_k={oracle_k}\n")
            f.write(f"oracle_in_topK={overall['oracle_in_topK']}  ({oracle_r:.6f})\n")
            f.write("\n")
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
            lm_err = int(dct["lm_err"])
            orc = int(dct["oracle_in_topK"])

            greedy_miss = ratio(g_err, total)
            lm_miss = ratio(lm_err, total)
            oracle_rate_char = ratio(orc, total)

            fix_rate_total = ratio(fix, total)
            worse_rate_total = ratio(worse, total)
            bothx_rate_total = ratio(bothx, total)
            repair_rate_char = ratio(fix, fix + bothx)

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

        # 混同行列
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

    # demo
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
