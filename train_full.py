#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full.py
- KODAI/full/各文献/characters/UNICODE/ にある 1文字画像を全件学習する分類器
- 文献（doc）単位で train/val split（リークを避ける）
- classes は full 全体の Unicode フォルダ名の集合で固定
- tqdm 進捗バー（train/val）+ 残り時間(ETA)を「分:秒」で明示表示
- AMP は torch.amp の新API
- 勾配蓄積 (gradient accumulation) 対応: --accum-steps
- 長尾対策: --use-weighted-sampler（inverse freq）
- 指標は train_loss と val_top1 のみ（必要最小限）
- 追加: epoch単位のETA（全学習があと何分で終わるか）を表示

python train_full.py \
  --full-root /home/ihpc/Documents/saito/KODAI/full \
  --model efficientnet_b0 \
  --epochs 100 \
  --batch-size 64 \
  --img-size 224 \
  --lr 1e-3 \
  --out-dir ./runs/full_effb0 \
  --amp \
  --use-weighted-sampler
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision import transforms

import timm
from tqdm import tqdm


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTENSIONS


def format_minutes(seconds: float) -> str:
    """
    seconds -> "mm:ss" 相当（ここでは "mm m ss s" を "mm m ss s" ではなく "mm m ss s" 表記にせず
    既存仕様に合わせて "MMmSSs"）
    """
    if seconds is None:
        return "??m??s"
    try:
        seconds = float(seconds)
    except Exception:
        return "??m??s"
    if seconds <= 0 or not math.isfinite(seconds):
        return "??m??s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}m{s:02d}s"


# ----------------------------
# Dataset indexing
# ----------------------------
@dataclass
class Sample:
    path: str
    y: int
    doc: str  # document id (folder name under full/)


def list_docs(full_root: Path) -> List[Path]:
    # full_root/100249371 みたいな文献フォルダを列挙
    docs = [p for p in full_root.iterdir() if p.is_dir()]
    docs.sort(key=lambda x: x.name)
    return docs


def collect_classes(full_root: Path) -> List[str]:
    """
    classes = 全 doc の characters/ 以下に存在する Unicode フォルダ名の union
    """
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
        raise RuntimeError(f"[ERROR] no classes found under {full_root}/*/characters/*")
    return classes


def collect_samples_docwise(
    full_root: Path,
    class_to_idx: Dict[str, int],
    followlinks: bool = True,
) -> List[Sample]:
    """
    full_root/doc/characters/UNICODE/*.png を全部拾って Sample 化
    doc-wise split するため doc id も持つ
    """
    samples: List[Sample] = []

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
                    if is_image_file(p) and p.is_file():
                        samples.append(Sample(path=str(p), y=y, doc=doc_id))

    if not samples:
        raise RuntimeError("[ERROR] collected samples = 0. Directory structure or permissions wrong?")
    return samples


def docwise_split(
    samples: List[Sample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    """
    doc（文献）単位で分割する。リークを避ける。
    """
    docs = sorted(list({s.doc for s in samples}))
    rng = random.Random(seed)
    rng.shuffle(docs)

    n_val = max(1, int(round(len(docs) * val_ratio)))
    val_docs = set(docs[:n_val])

    train_s = [s for s in samples if s.doc not in val_docs]
    val_s = [s for s in samples if s.doc in val_docs]

    if len(train_s) == 0 or len(val_s) == 0:
        raise RuntimeError(f"[ERROR] bad split: train={len(train_s)} val={len(val_s)} docs={len(docs)}")
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
# Transforms
# ----------------------------
def build_transforms(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.6, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(
            int(math.floor(img_size / 0.875)),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    accum_steps: int,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0

    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if use_cuda else None

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        loader,
        desc="train",
        leave=False,
        ncols=200,
        dynamic_ncols=False,
        smoothing=0.05,
    )

    last_step = -1
    for step, (imgs, targets) in enumerate(pbar):
        last_step = step
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_cuda:
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(imgs)
                loss = loss_fn(logits, targets) / max(accum_steps, 1)
            assert scaler is not None
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(imgs)
            loss = loss_fn(logits, targets) / max(accum_steps, 1)
            loss.backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        bsz = targets.size(0)
        total_loss += float(loss.item()) * bsz * max(accum_steps, 1)
        total_n += bsz

        eta = pbar.format_dict.get("remaining", 0.0)
        pbar.set_postfix(
            loss=f"{total_loss / max(total_n, 1):.4f}",
            eta=format_minutes(eta),
        )

    # 端数stepが残ったら一回更新（accum_stepsで割り切れない場合）
    if last_step >= 0 and ((last_step + 1) % accum_steps != 0):
        if use_cuda:
            assert scaler is not None
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    use_cuda = (device.type == "cuda")

    pbar = tqdm(
        loader,
        desc="val",
        leave=False,
        ncols=200,
        dynamic_ncols=False,
        smoothing=0.05,
    )

    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_cuda:
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(imgs)
                loss = loss_fn(logits, targets)
        else:
            logits = model(imgs)
            loss = loss_fn(logits, targets)

        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == targets).sum().item())

        bsz = targets.size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz

        eta = pbar.format_dict.get("remaining", 0.0)
        pbar.set_postfix(
            val_loss=f"{total_loss / max(total_n, 1):.4f}",
            val_top1=f"{total_correct / max(total_n, 1):.4f}",
            eta=format_minutes(eta),
        )

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-root", type=str, required=True, help="KODAI/full")
    ap.add_argument("--model", type=str, default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--weight-decay", type=float, default=2e-5)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val-ratio", type=float, default=0.06, help="doc-wise split ratio")
    ap.add_argument("--use-weighted-sampler", action="store_true")
    ap.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation steps")

    ap.add_argument("--out-dir", type=str, default="./runs/train_full")
    ap.add_argument("--resume", type=str, default="", help="path to .pth (optional)")

    ap.add_argument("--followlinks", action="store_true", help="follow symlinks when scanning")
    args = ap.parse_args()

    set_seed(args.seed)

    full_root = Path(args.full_root)
    if not full_root.is_dir():
        raise FileNotFoundError(full_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # classes
    classes = collect_classes(full_root)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # samples
    samples = collect_samples_docwise(full_root, class_to_idx, followlinks=args.followlinks)
    train_samples, val_samples = docwise_split(samples, val_ratio=args.val_ratio, seed=args.seed)

    print(f"[INFO] num_classes(full): {num_classes}")
    print(f"[INFO] samples total : {len(samples)}")
    print(f"[INFO] train samples : {len(train_samples)} (doc-wise)")
    print(f"[INFO] val samples   : {len(val_samples)} (doc-wise)")
    print(f"[INFO] out_dir      : {out_dir.resolve()}")

    # transforms
    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = FullCharDataset(train_samples, transform=train_tf)
    val_ds = FullCharDataset(val_samples, transform=val_tf)

    # sampler (optional)
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
        print("[INFO] using WeightedRandomSampler (inverse freq)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0),
    )

    # model
    model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)
    model.to(device)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    start_epoch = 0
    best_top1 = -1.0

    # resume (optional)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(sd, strict=True)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_top1 = float(ckpt.get("best_top1", best_top1))
        print(f"[RESUME] {args.resume} start_epoch={start_epoch} best_top1={best_top1}")

    # save meta
    save_json(out_dir / "classes.json", classes)
    save_json(out_dir / "class_to_idx.json", class_to_idx)
    meta = {
        "full_root": str(full_root.resolve()),
        "num_classes": num_classes,
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "lr": args.lr,
        "amp": bool(args.amp),
        "val_ratio": args.val_ratio,
        "use_weighted_sampler": bool(args.use_weighted_sampler),
        "followlinks": bool(args.followlinks),
    }
    save_json(out_dir / "meta.json", meta)

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            amp=args.amp, accum_steps=max(1, args.accum_steps)
        )
        val_loss, val_top1 = evaluate(model, val_loader, device, amp=args.amp)

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_top1": float(val_top1),
            "time_sec": float(dt),
        }
        history.append(row)
        save_json(out_dir / "history.json", history)

        # ---- NEW: epoch-level ETA (total remaining) ----
        epochs_done = len(history)
        avg_epoch_time = sum(h["time_sec"] for h in history) / max(epochs_done, 1)
        remaining_epochs = args.epochs - epoch - 1
        eta_total_sec = avg_epoch_time * remaining_epochs

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_top1={val_top1:.4f} | "
            f"time={dt:.1f}s | "
            f"ETA_total={format_minutes(eta_total_sec)}"
        )

        # save best
        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_top1": best_top1,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best.pth")
            print(f"[SAVE] best.pth (val_top1={best_top1:.4f})")

        # save last
        ckpt_last = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_top1": best_top1,
            "args": vars(args),
        }
        torch.save(ckpt_last, out_dir / "last.pth")

    print(f"[DONE] best_top1={best_top1:.4f}")
    print(f"[OUT]  {out_dir.resolve()}")


if __name__ == "__main__":
    main()
