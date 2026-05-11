#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import argparse, json, math, os, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import timm


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


class EarlyStopper:
    """
    Early stopping on a monitored metric.
    - mode="max": stop when metric doesn't improve for `patience` epochs.
    - mode="min": stop when metric doesn't improve (decrease) for `patience` epochs.
    - min_delta: required improvement amount (for max: metric > best + min_delta)
    """
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0):
        assert patience >= 0
        assert mode in ("max", "min")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, metric: float, epoch: int) -> bool:
        improved = False
        if self.mode == "max":
            if metric > self.best + self.min_delta:
                improved = True
        else:
            if metric < self.best - self.min_delta:
                improved = True

        if improved:
            self.best = float(metric)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            if self.patience == 0:
                return True
            return self.bad_epochs > self.patience


# ----------------------------
# Dataset (split-root version)
# ----------------------------
@dataclass
class Sample:
    path: str
    y: int
    cls: str

def list_class_dirs(split_dir: Path) -> List[Path]:
    if not split_dir.is_dir():
        return []
    dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda x: x.name)
    return dirs

def collect_classes_from_splits(train_root: Path, val_root: Path, test_root: Optional[Path]) -> List[str]:
    """
    Expected structure:
      split_root/{train,val,test}/U+XXXX/**/*.png

    Class name is the directory name under each split (e.g., 'U+4E00').
    """
    split_roots = [train_root, val_root]
    if test_root is not None:
        split_roots.append(test_root)

    classes = set()
    for root in split_roots:
        if root is None or (not root.is_dir()):
            continue
        for udir in root.iterdir():
            if not udir.is_dir():
                continue
            found = False
            for r, _, fnames in os.walk(str(udir), followlinks=True):
                for fn in fnames:
                    p = Path(r) / fn
                    if p.is_file() and is_image_file(p):
                        found = True
                        break
                if found:
                    break
            if found:
                classes.add(udir.name)

    classes = sorted(classes)
    if not classes:
        raise RuntimeError(f"no classes found under splits: {train_root}, {val_root}, {test_root}")
    return classes

def collect_samples_split(split_dir: Path, class_to_idx: Dict[str, int], followlinks: bool=True) -> List[Sample]:
    """
    Collect images under:
      split_dir/U+XXXX/**/*.png
    """
    out: List[Sample] = []
    for cls_dir in list_class_dirs(split_dir):
        cls = cls_dir.name
        if cls not in class_to_idx:
            continue
        y = class_to_idx[cls]
        for r, _, fnames in os.walk(str(cls_dir), followlinks=followlinks):
            for fn in fnames:
                p = Path(r) / fn
                if p.is_file() and is_image_file(p):
                    out.append(Sample(path=str(p), y=y, cls=cls))
    return out

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


# ----------------------------
# Transforms (EfficientNet default mean/std if available)
# ----------------------------
def build_transforms(img_size: int, mean, std):
    train_tf = transforms.Compose([
        LetterboxSquare(img_size, fill=255, interp=Image.BICUBIC),
        transforms.RandomGrayscale(p=0.2),
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
# Mixup (Input) - OFF by default (mixup_alpha=0.0)
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
def train_one_epoch(model, loader, opt, device, amp: bool,
                    scaler: torch.amp.GradScaler,
                    mixup_alpha: float,
                    manifold: bool):
    if manifold:
        raise NotImplementedError(
            "manifold mixup is NOT implemented for EfficientNet in this script. "
            "Use input mixup (mixup_alpha) or implement manifold mixup with forward hooks safely."
        )

    model.train()
    total_loss, total_n = 0.0, 0

    pbar = tqdm(loader, desc="train", leave=False, ncols=160)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        x_m, y1, y2, lam = mixup_input(x, y, mixup_alpha)
        with torch.amp.autocast("cuda", enabled=(amp and device.type=="cuda")):
            logits = model(x_m)
            loss = mixup_criterion(logits, y1, y2, lam)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz
        pbar.set_postfix(
            loss=f"{total_loss/max(total_n,1):.4f}",
            eta=format_mmss(pbar.format_dict.get("remaining", 0.0))
        )

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


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", type=str, required=True,
                    help="root containing train/ val/ test/ directories")
    ap.add_argument("--out-dir", type=str, default="./runs/kodai_effb0")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)

    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--use-weighted-sampler", action="store_true")

    # EfficientNet config
    ap.add_argument("--model", type=str, default="efficientnet_b0")
    ap.add_argument("--pretrained", action="store_true",
                    help="use ImageNet pretrained weights (timm). OFF by default for fair from-scratch.")

    # Mixup (OFF by default)
    ap.add_argument("--mixup-alpha", type=float, default=0.0, help="0=off, e.g. 0.2")

    # Keep args for compatibility, but NOT supported here
    ap.add_argument("--manifold-mixup", action="store_true", help="NOT supported for EfficientNet in this script")
    ap.add_argument("--manifold-layers", type=int, default=3, help="(ignored)")

    # Early stopping
    ap.add_argument("--earlystop", action="store_true", help="enable early stopping")
    ap.add_argument("--es-patience", type=int, default=10, help="patience epochs (no improvement)")
    ap.add_argument("--es-min-delta", type=float, default=0.0, help="minimum improvement to reset patience")
    ap.add_argument("--es-monitor", type=str, default="bacc", choices=["bacc", "top1", "val_loss"],
                    help="metric to monitor for early stopping")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.manifold_mixup:
        raise SystemExit(
            "[ERROR] --manifold-mixup is not implemented for EfficientNet in this script.\n"
            "        Use input mixup (--mixup-alpha) or implement manifold mixup with forward hooks."
        )

    split_root = Path(args.split_root)
    train_root = split_root / "train"
    val_root   = split_root / "val"
    test_root  = split_root / "test"
    has_test = test_root.is_dir()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print(f"[INFO] split_root={split_root} (test={'YES' if has_test else 'NO'})")

    # Collect classes and samples based on split structure
    classes = collect_classes_from_splits(train_root, val_root, test_root if has_test else None)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    train_samples = collect_samples_split(train_root, class_to_idx)
    val_samples   = collect_samples_split(val_root, class_to_idx)
    test_samples  = collect_samples_split(test_root, class_to_idx) if has_test else []

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError(f"empty split: train={len(train_samples)} val={len(val_samples)}")

    print(f"[INFO] classes={num_classes} train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    # Build model (timm)
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=num_classes)
    model = model.to(device)

    # Use model default mean/std if present; else fallback to ImageNet defaults
    default_cfg = getattr(model, "default_cfg", {}) or {}
    mean = tuple(default_cfg.get("mean", (0.485, 0.456, 0.406)))
    std  = tuple(default_cfg.get("std",  (0.229, 0.224, 0.225)))
    print("[INFO] model:", args.model, "pretrained:", bool(args.pretrained))
    print("[INFO] normalize mean/std:", mean, std)

    train_tf, val_tf = build_transforms(args.img_size, mean=mean, std=std)
    train_ds = FullCharDataset(train_samples, transform=train_tf)
    val_ds   = FullCharDataset(val_samples,   transform=val_tf)
    test_ds  = FullCharDataset(test_samples,  transform=val_tf) if has_test else None

    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        counts = [0] * num_classes
        for s in train_samples:
            counts[s.y] += 1
        counts = [c if c > 0 else 1 for c in counts]
        w_class = [1.0 / c for c in counts]
        weights = [w_class[s.y] for s in train_samples]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
        print("[INFO] WeightedRandomSampler ON")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0)
    )
    test_loader = None
    if has_test and test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0)
        )

    opt = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    save_json(out_dir / "classes.json", classes)
    save_json(out_dir / "meta.json", vars(args))

    # AMP scaler (keep across epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type=="cuda"))

    # Early stopping setup
    es = None
    if args.earlystop:
        es_mode = "min" if args.es_monitor == "val_loss" else "max"
        es = EarlyStopper(patience=args.es_patience, mode=es_mode, min_delta=args.es_min_delta)
        print(f"[INFO] EarlyStop ON monitor={args.es_monitor} mode={es_mode} patience={args.es_patience} min_delta={args.es_min_delta}")

    best_metric = 1e18 if args.es_monitor == "val_loss" else -1e18
    best_bacc = -1.0
    hist = []

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, opt, device, args.amp, scaler,
            mixup_alpha=args.mixup_alpha,
            manifold=args.manifold_mixup
        )
        va_loss, va_top1, va_bacc = evaluate(model, val_loader, device, args.amp, num_classes=num_classes)
        scheduler.step()
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | top1={va_top1:.4f} | bacc={va_bacc:.4f} | {dt:.1f}s")

        row = dict(
            epoch=int(epoch), tr_loss=float(tr_loss), va_loss=float(va_loss),
            top1=float(va_top1), bacc=float(va_bacc), sec=float(dt),
            lr=float(opt.param_groups[0]["lr"])
        )
        hist.append(row)
        save_json(out_dir / "history.json", hist)

        # Choose monitored metric
        if args.es_monitor == "bacc":
            metric = float(va_bacc)
            improved = metric > best_metric
        elif args.es_monitor == "top1":
            metric = float(va_top1)
            improved = metric > best_metric
        else:
            metric = float(va_loss)
            improved = metric < best_metric

        # Save best by monitor
        if improved:
            best_metric = metric
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                 "best_metric": best_metric, "monitor": args.es_monitor, "model_name": args.model},
                out_dir / "best_by_monitor.pth"
            )
            print(f"[SAVE] best_by_monitor.pth ({args.es_monitor}={best_metric:.6f})")

        # Save best by bacc too
        if va_bacc > best_bacc:
            best_bacc = float(va_bacc)
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                 "best_bacc": best_bacc, "model_name": args.model},
                out_dir / "best.pth"
            )
            print(f"[SAVE] best.pth (best_bacc={best_bacc:.4f})")

        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
             "best_metric": best_metric, "best_bacc": best_bacc, "model_name": args.model},
            out_dir / "last.pth"
        )

        # Early stop decision
        if es is not None:
            if es.step(metric, epoch):
                print(f"[EARLYSTOP] stop at epoch={epoch} best_epoch={es.best_epoch} best={es.best:.6f} monitor={args.es_monitor}")
                break

    print("[DONE] best_bacc=", best_bacc)
    print("[OUT] ", out_dir.resolve())

    if test_loader is not None:
        te_loss, te_top1, te_bacc = evaluate(model, test_loader, device, args.amp, num_classes=num_classes)
        print(f"[TEST] loss={te_loss:.4f} top1={te_top1:.4f} bacc={te_bacc:.4f}")


if __name__ == "__main__":
    main()


"""
python effitrain.py \
  --split-root /home/ihpc_double/Documents/saito/KODAI/fullsprit \
  --out-dir ./runs/kodai_effb0_pretrained \
  --epochs 100 \
  --batch-size 128 \
  --img-size 224 \
  --lr 0.02 \
  --weight-decay 1e-4 \
  --momentum 0.9 \
  --workers 4 \
  --amp \
  --use-weighted-sampler \
  --earlystop --es-monitor bacc --es-patience 10 --es-min-delta 0.0005 \
  --pretrained

"""
"""
screen -S
ctrl+a d
"""



"""
scp -P 20005 -r /home/ihpc_double/Documents/saito/KODAI2/runs/kodai_effb0_pretrained ihpc@172.24.160.42:/home/ihpc/Documents/saito/KODAI2/runs
"""