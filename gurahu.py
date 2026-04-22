#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties


def configure_japanese_font(font_path: str) -> FontProperties:
    font_file = Path(font_path)
    if not font_file.exists():
        raise FileNotFoundError(
            f"Font file not found: {font_file}\n"
            "Please set FONT_PATH to your downloaded Japanese font file."
        )

    fm.fontManager.addfont(str(font_file))
    font_prop = FontProperties(fname=str(font_file))

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    return font_prop


def add_value_labels(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.1f}",
    y_offset: float = 1.2,
) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    # =========================================================
    # 1) FONT PATH
    # =========================================================
    FONT_PATH = "/home/ihpc/Documents/saito/KODAI/fonts/IPAexGothic.ttf"
    font_prop = configure_japanese_font(FONT_PATH)

    # =========================================================
    # 2) Data
    # =========================================================
    conditions = [
        "   ",
        "   ",
        "   ",
        "   ",
    ]

    baseline = [
        36.32,
        88.00,
        80.37,
        67.98,
    ]

    kenlm = [
        62.33,
        94.67,
        97.06,
        69.46,
    ]

    bert = [
        78.03,
        1.33,
        99.02,
        69.95,
    ]

    # =========================================================
    # 3) Plot settings
    # =========================================================
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.titlesize"] = 17
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 12

    x = np.arange(len(conditions))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))

    # alphaを小さくすると棒の色が薄くなる
    # 0.45〜0.60くらいが論文図では無難
    bar_alpha = 0.50

    # MATLABっぽい青系3色
    bars_baseline = ax.bar(
        x - width,
        baseline,
        width=width,
        label="Baseline",
        color="#0072BD",   # MATLAB blue
        alpha=0.6,
        edgecolor="black",
        linewidth=0.6,
    )

    bars_kenlm = ax.bar(
        x,
        kenlm,
        width=width,
        label="KenLM",
        color="#4DBEEE",   # MATLAB light blue
        alpha=0.6,
        edgecolor="black",
        linewidth=0.6,
    )

    bars_bert = ax.bar(
        x + width,
        bert,
        width=width,
        label="BERT",
        color="#A6CEE3",   # MATLAB green寄り。青系に寄せるなら下に変更
        alpha=0.6,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontproperties=font_prop)

    ax.set_ylabel("Final accuracy (%)", fontproperties=font_prop)
    ax.set_xlabel("Representative conditions", fontproperties=font_prop)
    ax.set_title(
        "Representative character-level outcomes under different conditions",
        fontproperties=font_prop,
    )

    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", prop=font_prop)

    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    add_value_labels(ax, bars_baseline, fmt="{:.1f}")
    add_value_labels(ax, bars_kenlm, fmt="{:.1f}")
    add_value_labels(ax, bars_bert, fmt="{:.1f}")

    fig.tight_layout()

    # =========================================================
    # 4) Save
    # =========================================================
    output_dir = Path("discussion_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "discussion_5_2_grouped_conditions.png"
    pdf_path = output_dir / "discussion_5_2_grouped_conditions.pdf"
    svg_path = output_dir / "discussion_5_2_grouped_conditions.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print("Saved files:")
    print(f"  PNG: {png_path.resolve()}")
    print(f"  PDF: {pdf_path.resolve()}")
    print(f"  SVG: {svg_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()