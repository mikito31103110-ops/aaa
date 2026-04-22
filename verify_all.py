#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_all.py
- A) クロップ座標がページ画像と一致してるか（縮尺/回転/反転/座標系ズレ）
- B) 推論前処理の一致（RGB化、正規化の前段の画像統計など)
- C) 評価サンプル不足（macro系が暴れるのは当然か）を定量化
- ついでに "ページ画像にBBoxを描画した可視化" と "切り出しパッチ" を大量に吐く

依存: pandas, pillow
（torch/torchvision無くても A/C は見れる）
"""

from __future__ import annotations
import argparse, random, math, json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"]


def find_page_image(images_dir: Path, image_key: str) -> Path | None:
    """image_key: '100249371_00002_1' みたいな拡張子なしを想定。"""
    for ext in IMG_EXTS:
        p = images_dir / f"{image_key}{ext}"
        if p.exists():
            return p
    # たまに大文字拡張子とかあるので総当たり（重いので最後）
    cand = list(images_dir.glob(f"{image_key}.*"))
    for p in cand:
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def clamp_bbox(x, y, w, h, W, H):
    x0 = max(0, min(int(x), W - 1))
    y0 = max(0, min(int(y), H - 1))
    x1 = max(0, min(int(x + w), W))
    y1 = max(0, min(int(y + h), H))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def crop_variants(img: Image.Image, bbox):
    """
    座標系バグの典型を一気に可視化するために、
    同じbboxを以下の変換で試して patch を返す。

    - normal
    - flipX / flipY
    - rot90/180/270（回転後に bbox を貼るのは本当は別計算が必要だが、
      「座標系がそもそも回転してる」ケースを雑に炙り出す用途で、
      回転画像に対して "同じbbox" を切る方式にしている）
    """
    W, H = img.size
    x0, y0, x1, y1 = bbox
    out = {}

    def safe_crop(im, tag):
        try:
            patch = im.crop((x0, y0, x1, y1))
            out[tag] = patch
        except Exception:
            pass

    safe_crop(img, "normal")
    safe_crop(img.transpose(Image.FLIP_LEFT_RIGHT), "flipX")
    safe_crop(img.transpose(Image.FLIP_TOP_BOTTOM), "flipY")
    safe_crop(img.transpose(Image.ROTATE_90), "rot90")
    safe_crop(img.transpose(Image.ROTATE_180), "rot180")
    safe_crop(img.transpose(Image.ROTATE_270), "rot270")
    return out


def patch_score(p: Image.Image) -> float:
    """
    「切り出しがそれっぽいか」を雑にスコア化（0〜1）。
    - ほぼ真っ白/真っ黒なら低スコア
    - 画素の分散がある程度あれば高スコア
    くずし字は背景紙と墨のコントラストがあるので、分散が出やすい想定。
    """
    g = p.convert("L")
    # downsample for speed
    g = g.resize((max(8, g.size[0] // 4), max(8, g.size[1] // 4)))
    px = list(g.getdata())
    n = len(px)
    if n == 0:
        return 0.0
    mean = sum(px) / n
    var = sum((v - mean) ** 2 for v in px) / n
    # var の経験的な正規化（雑）
    # 真っ白: var ~ 0, それなりの文字: var 数百〜数千
    score = 1.0 - math.exp(-var / 800.0)
    return float(max(0.0, min(1.0, score)))


def draw_bbox_on_page(img: Image.Image, bbox, label: str | None = None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox
    d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
    if label:
        # PIL標準フォントで十分
        d.text((x0 + 2, y0 + 2), label, fill=(255, 0, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodai-root", required=True, type=str)
    ap.add_argument("--book-id", required=True, type=str)
    ap.add_argument("--coord-csv", default="", type=str)
    ap.add_argument("--pages", type=int, default=17, help="評価に使うページ枚数（末尾からNページ）")
    ap.add_argument("--samples", type=int, default=50, help="可視化する行数（bbox/crop）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="./verify_out")
    args = ap.parse_args()

    random.seed(args.seed)

    kodai_root = Path(args.kodai_root)
    book_id = args.book_id

    coord_csv = Path(args.coord_csv) if args.coord_csv else kodai_root / "full" / book_id / f"{book_id}_coordinate.csv"
    images_dir = kodai_root / "full" / book_id / "images"

    out_dir = Path(args.out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    (out_dir / "report").mkdir(parents=True, exist_ok=True)

    print("========== PATHS ==========")
    print("[INFO] coord_csv :", coord_csv)
    print("[INFO] images_dir:", images_dir)
    print("[INFO] out_dir   :", out_dir.resolve())

    if not coord_csv.exists():
        raise FileNotFoundError(f"coord_csv not found: {coord_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    df = pd.read_csv(coord_csv)
    need_cols = ["Unicode", "Image", "X", "Y", "Width", "Height"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"coord_csv missing column: {c} (have {list(df.columns)})")

    df["Image"] = df["Image"].astype(str)
    df["Unicode"] = df["Unicode"].astype(str)

    # ページ一覧（Image列が page_key 前提）
    pages = sorted(df["Image"].unique().tolist())
    print("[INFO] coord rows   :", len(df))
    print("[INFO] unique pages :", len(pages))
    if len(pages) == 0:
        raise RuntimeError("No pages in coord_csv")

    # 末尾Nページを test と見なす（あなたの step.py と合わせる）
    n_test = min(args.pages, len(pages))
    test_pages = pages[-n_test:]
    test_df = df[df["Image"].isin(test_pages)].copy()
    print("[INFO] using test pages:", n_test, "(tail of sorted list)")
    print("[INFO] test rows        :", len(test_df))

    # ページ画像の存在率
    missing_pages = []
    page_path = {}
    for pk in test_pages:
        p = find_page_image(images_dir, pk)
        if p is None:
            missing_pages.append(pk)
        else:
            page_path[pk] = p
    print("[INFO] page files found :", len(page_path), "/", len(test_pages))
    if missing_pages:
        print("[WARN] missing page files (first 10):", missing_pages[:10])

    # C) サンプル不足指標：クラス頻度
    true_counts = Counter(test_df["Unicode"].tolist())
    uniq_true = len(true_counts)
    total = len(test_df)
    print("========== C) SAMPLE STATS ==========")
    print("[INFO] test unique true classes:", uniq_true)
    if total > 0:
        # 1回しか出ないクラスの割合
        singletons = sum(1 for k, v in true_counts.items() if v == 1)
        print("[INFO] singleton classes:", singletons, f"({singletons/uniq_true:.3f} of classes)")
        # macro系が暴れる度合い（1サンプル以下のクラスが多いほど不安定）
        print("[INFO] avg samples per class:", total / max(1, uniq_true))
    else:
        print("[WARN] test_df empty -> coord_csv/page split issue")

    # A) クロップ整合の検証：可視化サンプルを抽出
    # 画像ファイルが存在する行だけ
    test_df["page_path"] = test_df["Image"].map(lambda x: str(page_path.get(x, "")))
    usable = test_df[test_df["page_path"] != ""].copy()
    print("========== A) CROP VALIDATION ==========")
    print("[INFO] rows with page file:", len(usable), "/", len(test_df))

    if len(usable) == 0:
        print("[FATAL] page画像が見つからないのでクロップ検証できません。images_dir と Image列の対応を見直してください。")
        return

    # ランダムにN行サンプル
    n = min(args.samples, len(usable))
    sample_rows = usable.sample(n=n, random_state=args.seed).reset_index(drop=True)

    # 各サンプルについて、normal/flip/rot のスコアを取り、最良変換を集計
    best_transform_counts = Counter()
    score_records = []
    saved = 0

    for i, row in sample_rows.iterrows():
        pk = row["Image"]
        uni = row["Unicode"]
        x, y, w, h = row["X"], row["Y"], row["Width"], row["Height"]
        pimg_path = Path(row["page_path"])

        try:
            page_img = Image.open(pimg_path).convert("RGB")
        except Exception as e:
            continue

        W, H = page_img.size
        bbox = clamp_bbox(x, y, w, h, W, H)
        if bbox is None:
            score_records.append({"i": i, "page": pk, "unicode": uni, "status": "invalid_bbox"})
            continue

        variants = crop_variants(page_img, bbox)
        if not variants:
            score_records.append({"i": i, "page": pk, "unicode": uni, "status": "no_variants"})
            continue

        scores = {k: patch_score(v) for k, v in variants.items()}
        best_t = max(scores.items(), key=lambda kv: kv[1])[0]
        best_s = scores[best_t]
        best_transform_counts[best_t] += 1

        score_records.append({
            "i": i,
            "page": pk,
            "unicode": uni,
            "page_w": W, "page_h": H,
            "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "bbox": bbox,
            "best_transform": best_t,
            "best_score": float(best_s),
            **{f"score_{k}": float(v) for k, v in scores.items()}
        })

        # 保存：ページにbbox描画＆patch保存（bestのやつ）
        try:
            # bbox付きページ（重いので間引き）
            if saved < 30:
                page_vis = draw_bbox_on_page(page_img, bbox, label=f"{uni} #{i}")
                page_vis.save(out_dir / "pages" / f"{pk}__{i}__{uni}.jpg", quality=90)

            # best patch と normal patch は保存（比較用）
            variants[best_t].save(out_dir / "crops" / f"{pk}__{i}__{uni}__BEST_{best_t}.jpg", quality=95)
            variants["normal"].save(out_dir / "crops" / f"{pk}__{i}__{uni}__normal.jpg", quality=95)
            saved += 1
        except Exception:
            pass

    # レポート出力
    rec_df = pd.DataFrame(score_records)
    rec_df.to_csv(out_dir / "report" / "crop_scores.csv", index=False)

    print("[INFO] saved crops/pages:", saved)
    print("========== A) DIAG SUMMARY ==========")
    if len(rec_df) == 0:
        print("[FATAL] サンプルからcropスコアが取れませんでした（画像読み込み/座標列の型などを確認）。")
        return

    # best_score 分布
    valid_scores = rec_df[rec_df["best_score"].notna()]["best_score"].tolist()
    valid_scores.sort()
    def q(p):
        if not valid_scores:
            return None
        idx = int(round((len(valid_scores)-1) * p))
        return valid_scores[idx]

    print("[INFO] best_score quantiles:",
          "q10=", f"{q(0.10):.3f}" if q(0.10) is not None else "NA",
          "q50=", f"{q(0.50):.3f}" if q(0.50) is not None else "NA",
          "q90=", f"{q(0.90):.3f}" if q(0.90) is not None else "NA")

    print("[INFO] best_transform counts:", dict(best_transform_counts))

    # 判定ロジック（雑だけど“原因特定”には効く）
    # - normalが圧倒的＆スコアもそこそこ -> 座標は概ね合ってる
    # - flip/rotが勝つ割合が高い -> 座標系の取り扱いが怪しい
    # - best_score自体が全体に低い -> そもそもbboxが文字を切れてない可能性が高い
    total_s = sum(best_transform_counts.values()) or 1
    normal_rate = best_transform_counts["normal"] / total_s
    flip_rate = (best_transform_counts["flipX"] + best_transform_counts["flipY"]) / total_s
    rot_rate = (best_transform_counts["rot90"] + best_transform_counts["rot180"] + best_transform_counts["rot270"]) / total_s

    diagnosis = []
    if normal_rate >= 0.75 and q(0.50) is not None and q(0.50) >= 0.25:
        diagnosis.append("座標→ページ画像は概ね整合してそう（少なくとも致命的ズレの確率は低い）。")
    else:
        diagnosis.append("座標→ページ画像の整合が怪しい（可視化画像で '赤枠が文字を囲っているか' を最優先で目視確認）。")

    if flip_rate >= 0.25:
        diagnosis.append("flip が頻繁に最良 → 左右/上下反転の座標系ミスの疑い。")
    if rot_rate >= 0.25:
        diagnosis.append("rot が頻繁に最良 → ページ画像の回転・縦横扱いミスの疑い。")
    if q(0.50) is not None and q(0.50) < 0.15:
        diagnosis.append("best_score中央値が低い → bboxが文字を切れていない（縮尺違い・単位違い・別画像参照）可能性が高い。")

    # B) 前処理の前段（RGB化や階調）を軽くチェック：crop( normal )の画素統計
    # (torch無しでできる範囲)
    print("========== B) PREPROCESS SANITY ==========")
    # 正常なら背景(紙)が明るく、墨が暗いのでL平均はそこそこ高め＆分散もあるはず
    l_means, l_vars = [], []
    for _, r in rec_df.dropna(subset=["bbox"]).head(30).iterrows():
        pk = r["page"]
        p = page_path.get(pk)
        if not p:
            continue
        img = Image.open(p).convert("RGB")
        x0, y0, x1, y1 = eval(str(r["bbox"])) if isinstance(r["bbox"], str) else r["bbox"]
        patch = img.crop((x0, y0, x1, y1)).convert("L").resize((32, 32))
        px = list(patch.getdata())
        npx = len(px)
        m = sum(px) / npx
        v = sum((t - m) ** 2 for t in px) / npx
        l_means.append(m); l_vars.append(v)

    if l_means:
        mean_mean = sum(l_means) / len(l_means)
        mean_var = sum(l_vars) / len(l_vars)
        print(f"[INFO] patch L-mean(avg over {len(l_means)}): {mean_mean:.1f}  (0=black,255=white)")
        print(f"[INFO] patch L-var (avg): {mean_var:.1f}")
        if mean_var < 50:
            print("[WARN] 分散がかなり低い → ほぼ白/ほぼ黒のcropが多い可能性。座標ズレ or 余白過多を疑う。")
    else:
        print("[WARN] patch統計が取れませんでした。")

    # レポートまとめ
    summary = {
        "coord_csv": str(coord_csv),
        "images_dir": str(images_dir),
        "pages_total": len(pages),
        "test_pages_used": n_test,
        "test_rows": len(test_df),
        "page_files_found": len(page_path),
        "unique_true_classes": uniq_true,
        "avg_samples_per_class": (total / max(1, uniq_true)) if total else 0.0,
        "singleton_class_ratio": (sum(1 for k, v in true_counts.items() if v == 1) / max(1, uniq_true)) if uniq_true else 0.0,
        "crop_samples": int(n),
        "best_score_q10": q(0.10),
        "best_score_q50": q(0.50),
        "best_score_q90": q(0.90),
        "best_transform_counts": dict(best_transform_counts),
        "normal_rate": normal_rate,
        "flip_rate": flip_rate,
        "rot_rate": rot_rate,
        "diagnosis": diagnosis,
    }
    (out_dir / "report" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== VERDICT ==========")
    for s in diagnosis:
        print("[DIAG]", s)
    print("[OUT] crops :", (out_dir / "crops").resolve())
    print("[OUT] pages :", (out_dir / "pages").resolve())
    print("[OUT] report:", (out_dir / "report").resolve())


if __name__ == "__main__":
    main()
