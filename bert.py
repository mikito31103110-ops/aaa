#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import math
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)



ENCODING = "utf-8"

# gt_pages のルート
DEFAULT_GT_ROOT = "/home/ihpc/Documents/saito/KODAI2/gt_pages"

# 学習対象の文献ID
DEFAULT_TRAIN_BOOK_IDS = [
    "100241706",
    "100249376",
    "100249476",
    "100249537",
    "200003076",
    "200004148",
    "200006663",
    "200006665",
    "200008316",
    "200014685",
    "200015779",
    "200021644",
    "200021712",
    "200021802",
    "200021925",
    "brsk00000",
    "hnsd00000",
    "umgy00000",
]

# 出力先
DEFAULT_OUTPUT_DIR = "/home/ihpc/Documents/saito/KODAI2/runs/train_bert_kodai_epoch300"

# ベースモデル
DEFAULT_MODEL_NAME = "tohoku-nlp/bert-base-japanese-char-v3"


# =========================================================
# 文字列正規化
# KenLM用の「空白区切り」はここでは使わない
# BERT-char 用なので、基本は空白除去
# =========================================================
def normalize_text_for_bert(
    s: str,
    remove_spaces: bool = True,
    remove_markers: bool = True,
) -> str:
    if s is None:
        return ""

    # 改行類を除去
    s = s.replace("\r", "")
    s = s.replace("\n", "")

    # 全角空白を半角に寄せる
    s = s.replace("\u3000", " ")

    # 前後空白除去
    s = s.strip()

    # BERT用では文字間スペースは邪魔になりやすい
    if remove_spaces:
        s = s.replace(" ", "")

    return s


def maybe_reverse_text(s: str, reverse_train: bool) -> str:
    if not reverse_train:
        return s
    return s[::-1]


# =========================================================
# gt_pages から学習対象ページを集める
# =========================================================
def collect_gt_page_files(gt_root: Path, train_book_ids: List[str]) -> List[Path]:
    pages: List[Path] = []
    missing_dirs: List[str] = []
    missing_books: List[str] = []

    for bid in train_book_ids:
        d = gt_root / bid
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
        print(f"[ERROR] no txt files found under {gt_root} for TRAIN_BOOK_IDS")
        sys.exit(1)

    return pages


# =========================================================
# gt_pages -> 学習用文字列リスト
# 1ページ = 1サンプル
# =========================================================
def build_training_texts_from_gtpages(
    gt_root: Path,
    train_book_ids: List[str],
    encoding: str = ENCODING,
    reverse_train: bool = False,
    remove_spaces: bool = True,
    remove_markers: bool = True,
    min_chars: int = 1,
) -> List[str]:
    pages = collect_gt_page_files(gt_root, train_book_ids)

    print(f"[INFO] found {len(pages)} txt files from {len(train_book_ids)} TRAIN books")
    print(f"[INFO] REVERSE_TRAIN: {reverse_train}")
    print(f"[INFO] remove_spaces: {remove_spaces}")
    print(f"[INFO] remove_markers: {remove_markers}")

    texts: List[str] = []
    skipped_empty = 0
    skipped_too_short = 0

    for p in pages:
        s = p.read_text(encoding=encoding, errors="ignore")
        s = normalize_text_for_bert(
            s,
            remove_spaces=remove_spaces,
            remove_markers=remove_markers,
        )
        s = maybe_reverse_text(s, reverse_train)

        if not s:
            skipped_empty += 1
            continue

        if len(s) < min_chars:
            skipped_too_short += 1
            continue

        texts.append(s)

    print(f"[INFO] texts kept={len(texts)}")
    print(f"[INFO] empty skipped={skipped_empty}")
    print(f"[INFO] too_short skipped={skipped_too_short}")

    if not texts:
        print("[ERROR] no valid training texts after normalization")
        sys.exit(1)

    return texts


# =========================================================
# 学習用コーパス保存（確認・再利用用）
# =========================================================
def save_corpus_preview_and_full(texts: List[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "train_corpus_bert.txt"
    preview_path = output_dir / "train_corpus_preview.txt"
    meta_path = output_dir / "corpus_meta.json"

    with corpus_path.open("w", encoding="utf-8") as f:
        for s in texts:
            f.write(s + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        n = min(20, len(texts))
        for i in range(n):
            f.write(f"[{i}] {texts[i]}\n")

    lengths = [len(x) for x in texts]
    meta = {
        "num_texts": len(texts),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": sum(lengths) / len(lengths),
        "median_len_approx": sorted(lengths)[len(lengths) // 2],
        "corpus_path": str(corpus_path),
        "preview_path": str(preview_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] saved corpus:   {corpus_path}")
    print(f"[INFO] saved preview:  {preview_path}")
    print(f"[INFO] saved meta:     {meta_path}")

    return corpus_path


# =========================================================
# torch Dataset
# texts を token 化して持つ
# =========================================================
class GTPagesMLMDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples: List[Dict[str, List[int]]] = []
        self._build()

    def _build(self):
        kept = 0
        dropped = 0

        for s in self.texts:
            enc = self.tokenizer(
                s,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True,
            )

            # [CLS] [SEP] だけになるような壊れた例は捨てる
            if len(enc["input_ids"]) <= 2:
                dropped += 1
                continue

            self.examples.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }
            )
            kept += 1

        print(f"[INFO] dataset examples kept={kept}, dropped={dropped}")

        if not self.examples:
            raise RuntimeError("No valid tokenized examples were created.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


# =========================================================
# メイン
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    # データ設定
    parser.add_argument("--gt_root", type=str, default=DEFAULT_GT_ROOT)
    parser.add_argument(
        "--train_book_ids",
        type=str,
        nargs="*",
        default=DEFAULT_TRAIN_BOOK_IDS,
        help="train に使う book_id 群。未指定ならコード内 DEFAULT_TRAIN_BOOK_IDS を使う",
    )
    parser.add_argument("--reverse_train", action="store_true")
    parser.add_argument("--keep_spaces", action="store_true", help="BERT用では通常使わない")
    parser.add_argument("--keep_markers", action="store_true", help="▲ などを残す")
    parser.add_argument("--min_chars", type=int, default=1)

    # モデル設定
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=256)

    # 学習設定
    parser.add_argument("--num_train_epochs", type=float, default=300.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # mixed precision
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # dataloader
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        print("[ERROR] --fp16 and --bf16 cannot be enabled at the same time")
        sys.exit(1)

    if not args.train_book_ids:
        print("[ERROR] train_book_ids is empty")
        print("        set DEFAULT_TRAIN_BOOK_IDS in the script or pass --train_book_ids ...")
        sys.exit(1)

    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("=== settings ===")
    print("gt_root                      :", gt_root)
    print("train_book_ids               :", args.train_book_ids)
    print("reverse_train                :", args.reverse_train)
    print("keep_spaces                  :", args.keep_spaces)
    print("keep_markers                 :", args.keep_markers)
    print("min_chars                    :", args.min_chars)
    print("model_name                   :", args.model_name)
    print("output_dir                   :", output_dir)
    print("max_length                   :", args.max_length)
    print("num_train_epochs             :", args.num_train_epochs)
    print("learning_rate                :", args.learning_rate)
    print("weight_decay                 :", args.weight_decay)
    print("mlm_probability              :", args.mlm_probability)
    print("per_device_train_batch_size  :", args.per_device_train_batch_size)
    print("gradient_accumulation_steps  :", args.gradient_accumulation_steps)
    print("logging_steps                :", args.logging_steps)
    print("save_steps                   :", args.save_steps)
    print("save_total_limit             :", args.save_total_limit)
    print("seed                         :", args.seed)
    print("fp16                         :", args.fp16)
    print("bf16                         :", args.bf16)
    print()

    # -----------------------------------------------------
    # 1) gt_pages から学習文字列を作る
    # -----------------------------------------------------
    texts = build_training_texts_from_gtpages(
        gt_root=gt_root,
        train_book_ids=args.train_book_ids,
        encoding=ENCODING,
        reverse_train=args.reverse_train,
        remove_spaces=not args.keep_spaces,
        remove_markers=not args.keep_markers,
        min_chars=args.min_chars,
    )

    save_corpus_preview_and_full(texts, output_dir)

    # -----------------------------------------------------
    # 2) tokenizer / model
    # -----------------------------------------------------
    print("\n=== loading tokenizer/model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        use_safetensors=True,
    )

    # -----------------------------------------------------
    # 3) Dataset
    # -----------------------------------------------------
    print("\n=== building torch dataset ===")
    train_dataset = GTPagesMLMDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # -----------------------------------------------------
    # 4) Data collator
    # 動的に [MASK] を入れる
    # -----------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # -----------------------------------------------------
    # 5) TrainingArguments
    # -----------------------------------------------------
    print("\n=== preparing training args ===")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    # -----------------------------------------------------
    # 6) Trainer
    # -----------------------------------------------------
    print("\n=== start training ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()

    # -----------------------------------------------------
    # 7) 保存
    # -----------------------------------------------------
    print("\n=== saving model ===")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # train metrics
    metrics = train_result.metrics
    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] saved final model   : {final_dir}")
    print(f"[INFO] saved train metrics : {metrics_path}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()



"""
PYTHONNOUSERSITE=1 python bert.py \
  --fp16
"""