#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, math, os, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision import transforms
from tqdm import tqdm

from PIL import Image


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTENSIONS

def format_mmss(seconds: float) -> str:
    if seconds is None:
        return "??:??"
    try:
        seconds = float(seconds)
    except Exception:
        return "??:??"
    if seconds <= 0 or not math.isfinite(seconds):
        return "??:??"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ----------------------------
# Dataset indexing
# ----------------------------
@dataclass
class Sample:
    path: str
    y: int
    doc: str

def list_docs(full_root: Path) -> List[Path]:
    docs = [p for p in full_root.iterdir() if p.is_dir()]
    docs.sort(key=lambda x: x.name)
    return docs

def collect_classes(full_root: Path) -> List[str]:
    classes = set()
    for doc_dir in list_docs(full_root):
        char_dir = doc_dir / "characters"
        if not char_dir.is_dir():
            continue
        for udir in char_dir.iterdir():
            if udir.is_dir():
                classes.add(udir.name)
    classes = sorted(classes)
    if not classes:
        raise RuntimeError(f"no classes found under {full_root}/*/characters/*")
    return classes

def collect_samples_docwise(full_root: Path, class_to_idx: Dict[str, int], followlinks: bool=True) -> List[Sample]:
    out: List[Sample] = []
    for doc_dir in list_docs(full_root):
        doc_id = doc_dir.name
        char_dir = doc_dir / "characters"
        if not char_dir.is_dir():
            continue
        for udir in char_dir.iterdir():
            if not udir.is_dir():
                continue
            u = udir.name
            if u not in class_to_idx:
                continue
            y = class_to_idx[u]
            for r, _, fnames in os.walk(str(udir), followlinks=followlinks):
                for fn in fnames:
                    p = Path(r) / fn
                    if p.is_file() and is_image_file(p):
                        out.append(Sample(path=str(p), y=y, doc=doc_id))
    if not out:
        raise RuntimeError("collected samples = 0")
    return out

def docwise_split(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    docs = sorted(list({s.doc for s in samples}))
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_val = max(1, int(round(len(docs) * val_ratio)))
    val_docs = set(docs[:n_val])
    train_s = [s for s in samples if s.doc not in val_docs]
    val_s   = [s for s in samples if s.doc in val_docs]
    if len(train_s) == 0 or len(val_s) == 0:
        raise RuntimeError(f"bad split train={len(train_s)} val={len(val_s)} docs={len(docs)}")
    return train_s, val_s

class FullCharDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform
        self.loader = default_loader
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self.loader(s.path)
        if self.transform is not None:
            img = self.transform(img)
        return img, s.y


# ----------------------------
# Letterbox 
# ----------------------------
class LetterboxSquare:
    """
    アスペクト比を維持して正方形へ。余白は白で埋める。
    - 入力: PIL.Image
    - 出力: PIL.Image (size, size)
    """
    def __init__(self, size: int, fill: int = 255, interp=Image.BICUBIC):
        self.size = int(size)
        self.fill = int(fill)
        self.interp = interp

    def __call__(self, img: Image.Image) -> Image.Image:
        # 念のためRGB化（default_loaderは通常RGBだが、壊れファイル対策）
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))

        # scale to fit
        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        img_r = img.resize((nw, nh), resample=self.interp)

        # white canvas
        canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        left = (self.size - nw) // 2
        top  = (self.size - nh) // 2
        canvas.paste(img_r, (left, top))
        return canvas


# ----------------------------
# Transforms 
# ----------------------------
def build_transforms(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([

        LetterboxSquare(img_size, fill=255, interp=Image.BICUBIC),

        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),

        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([

        LetterboxSquare(img_size, fill=255, interp=Image.BICUBIC),

        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ----------------------------
# PreActResNet-18 (minimal)
# ----------------------------
class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)  # pre-activation shortcut
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_ch=3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, n, stride):
        strides = [stride] + [1]*(n-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mixup_layer: int = -1, x2=None, lam: float = 1.0):
        out = self.conv1(x)
        if mixup_layer == 0 and x2 is not None:
            out = lam*out + (1-lam)*x2

        out = self.layer1(out)
        if mixup_layer == 1 and x2 is not None:
            out = lam*out + (1-lam)*x2

        out = self.layer2(out)
        if mixup_layer == 2 and x2 is not None:
            out = lam*out + (1-lam)*x2

        out = self.layer3(out)
        if mixup_layer == 3 and x2 is not None:
            out = lam*out + (1-lam)*x2

        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.fc(out)
        return logits

def preact_resnet18(num_classes: int):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, in_ch=3)


# ----------------------------
# Mixup (Input / Manifold)
# ----------------------------
def sample_lam(alpha: float, device):
    if alpha <= 0:
        return 1.0
    dist = torch.distributions.Beta(alpha, alpha)
    return float(dist.sample().to(device).item())

def mixup_input(x, y, alpha: float):
    if alpha <= 0 or x.size(0) < 2:
        return x, y, None, 1.0
    device = x.device
    lam = sample_lam(alpha, device)
    perm = torch.randperm(x.size(0), device=device)
    x2 = x[perm]
    y2 = y[perm]
    xm = lam*x + (1-lam)*x2
    return xm, y, y2, lam

def mixup_criterion(logits, y1, y2, lam):
    if y2 is None:
        return F.cross_entropy(logits, y1)
    return lam*F.cross_entropy(logits, y1) + (1-lam)*F.cross_entropy(logits, y2)

def balanced_accuracy(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    accs = []
    for c in range(num_classes):
        m = (gt == c)
        if m.sum() == 0:
            continue
        accs.append((pred[m] == gt[m]).mean())
    return float(np.mean(accs)) if accs else 0.0


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, opt, device, amp: bool, mixup_alpha: float, manifold: bool, n_manifold_layers: int):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type=="cuda"))
    total_loss, total_n = 0.0, 0

    pbar = tqdm(loader, desc="train", leave=False, ncols=160)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if not manifold:
            x_m, y1, y2, lam = mixup_input(x, y, mixup_alpha)
            with torch.amp.autocast("cuda", enabled=(amp and device.type=="cuda")):
                logits = model(x_m)
                loss = mixup_criterion(logits, y1, y2, lam)
        else:
            if mixup_alpha > 0 and x.size(0) >= 2:
                lam = sample_lam(mixup_alpha, device)
                perm = torch.randperm(x.size(0), device=device)
                x2 = x[perm]
                y2 = y[perm]
                mix_layer = random.randint(0, max(0, n_manifold_layers-1))
            else:
                lam, x2, y2, mix_layer = 1.0, None, None, -1

            with torch.amp.autocast("cuda", enabled=(amp and device.type=="cuda")):
                if mix_layer == 0:
                    mix_layer = 1

                feat2 = model.conv1(x2)
                if mix_layer >= 1:
                    feat2 = model.layer1(feat2)
                if mix_layer >= 2:
                    feat2 = model.layer2(feat2)
                if mix_layer >= 3:
                    feat2 = model.layer3(feat2)

                logits = model(x, mixup_layer=mix_layer, x2=feat2, lam=lam)
                loss = lam*F.cross_entropy(logits, y) + (1-lam)*F.cross_entropy(logits, y2)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz
        pbar.set_postfix(loss=f"{total_loss/max(total_n,1):.4f}", eta=format_mmss(pbar.format_dict.get("remaining",0.0)))

    return total_loss / max(total_n, 1)

@torch.no_grad()
def evaluate(model, loader, device, amp: bool, num_classes: int):
    model.eval()
    total_loss, total_n = 0.0, 0
    preds, gts = [], []

    pbar = tqdm(loader, desc="val", leave=False, ncols=160)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(amp and device.type=="cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        p = torch.argmax(logits, dim=1)
        preds.append(p.detach().cpu().numpy())
        gts.append(y.detach().cpu().numpy())

        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz

    pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    gt   = np.concatenate(gts)   if gts else np.array([], dtype=np.int64)
    top1 = float((pred == gt).mean()) if gt.size else 0.0
    bacc = balanced_accuracy(pred, gt, num_classes=num_classes) if gt.size else 0.0
    return total_loss / max(total_n, 1), top1, bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.06)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--out-dir", type=str, default="./runs/kodai_preact18_letterbox")
    ap.add_argument("--use-weighted-sampler", action="store_true")

    ap.add_argument("--mixup-alpha", type=float, default=0.0, help="0=off, e.g. 0.2")
    ap.add_argument("--manifold-mixup", action="store_true", help="use manifold mixup instead of input mixup")
    ap.add_argument("--manifold-layers", type=int, default=3, help="1..3 (mix at layer1..layer3)")

    args = ap.parse_args()
    set_seed(args.seed)

    full_root = Path(args.full_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    classes = collect_classes(full_root)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    num_classes = len(classes)

    samples = collect_samples_docwise(full_root, class_to_idx)
    train_s, val_s = docwise_split(samples, args.val_ratio, args.seed)
    print(f"[INFO] classes={num_classes} samples={len(samples)} train={len(train_s)} val={len(val_s)}")

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = FullCharDataset(train_s, transform=train_tf)
    val_ds   = FullCharDataset(val_s,   transform=val_tf)

    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        counts = [0]*num_classes
        for s in train_s:
            counts[s.y] += 1
        counts = [c if c>0 else 1 for c in counts]
        w_class = [1.0/c for c in counts]
        weights = [w_class[s.y] for s in train_s]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
        print("[INFO] WeightedRandomSampler ON")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers>0)
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers>0)
    )

    model = preact_resnet18(num_classes=num_classes).to(device)

    opt = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    save_json(out_dir/"classes.json", classes)
    save_json(out_dir/"meta.json", vars(args))

    best_bacc = -1.0
    hist = []
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, opt, device, args.amp,
            mixup_alpha=args.mixup_alpha,
            manifold=args.manifold_mixup,
            n_manifold_layers=args.manifold_layers
        )
        va_loss, va_top1, va_bacc = evaluate(model, val_loader, device, args.amp, num_classes=num_classes)
        scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | top1={va_top1:.4f} | bacc={va_bacc:.4f} | {dt:.1f}s")

        row = dict(
            epoch=epoch, tr_loss=float(tr_loss), va_loss=float(va_loss),
            top1=float(va_top1), bacc=float(va_bacc), sec=float(dt),
            lr=float(opt.param_groups[0]["lr"])
        )
        hist.append(row)
        save_json(out_dir/"history.json", hist)

        if va_bacc > best_bacc:
            best_bacc = va_bacc
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best_bacc": best_bacc},
                out_dir/"best.pth"
            )
            print(f"[SAVE] best.pth (best_bacc={best_bacc:.4f})")

        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best_bacc": best_bacc},
            out_dir/"last.pth"
        )

    print("[DONE] best_bacc=", best_bacc)
    print("[OUT] ", out_dir.resolve())

if __name__ == "__main__":
    main()



"""
python re_train.py \
  --full-root /home/ihpc/Documents/saito/KODAI/full \
  --out-dir ./runs/kodai_preact18_letterbox \
  --epochs 100 \
  --batch-size 64 \
  --img-size 224 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --momentum 0.9 \
  --val-ratio 0.06 \
  --workers 4 \
  --amp \
  --use-weighted-sampler
  
  
  
python re_train.py \
  --full-root /home/ihpc/Documents/saito/KODAI/full \
  --out-dir ./runs/kodai_preact18_manifoldmix \
  --epochs 50 \
  --batch-size 64 \
  --img-size 224 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --val-ratio 0.06 \
  --amp \
  --use-weighted-sampler \
  --mixup-alpha 0.2 \
  --manifold-mixup \
  --manifold-layers 3

  
"""