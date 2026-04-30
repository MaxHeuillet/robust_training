"""
plot_benchmark_table.py
=======================
Generates a publication-ready PDF/PNG figure characterising the
RobustGenBench benchmark datasets.

Usage:
    python plot_benchmark_table.py --dest ~/data/processed
    python plot_benchmark_table.py --dest ~/data/processed --out figure1.pdf
"""

import argparse
import csv
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import zstandard as zstd
except ImportError:
    raise SystemExit("Install zstandard:  pip install zstandard")

# ── dataset metadata ──────────────────────────────────────────────────────────

DATASETS = [
    dict(key="uc-merced-land-use-dataset", name="UC Merced",      task="coarse", domain="satellite"),
    dict(key="caltech101",                 name="Caltech-101",     task="coarse", domain="natural"),
    dict(key="flowers-102",                name="Flowers-102",     task="fine",   domain="natural"),
    dict(key="oxford-iiit-pet",            name="Oxford-IIIT Pet", task="fine",   domain="natural"),
    dict(key="stanford_cars",              name="Stanford Cars",   task="fine",   domain="natural"),
    dict(key="fgvc-aircraft-2013b",        name="FGVC Aircraft",   task="fine",   domain="natural"),
]

TASK_COLORS   = {"coarse": "#2C2C2A", "fine": "#2C2C2A"}
BAR_COLOR     = "#7F77DD"
DIVIDER_COLOR = "#d8d6ce"
TEXT_PRIMARY  = "#2C2C2A"
TEXT_MUTED    = "#888780"

# ── archive reading ───────────────────────────────────────────────────────────

def read_test_labels(archive_path: Path):
    labels, n_classes = [], 0
    dctx = zstd.ZstdDecompressor()
    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|*") as tf:
                for member in tf:
                    if member.name == "metadata.json":
                        meta = json.loads(tf.extractfile(member).read())
                        n_classes = meta.get("N", 0)
                    elif member.name == "test/labels.csv":
                        content = tf.extractfile(member).read().decode()
                        labels = [int(row["label"])
                                  for row in csv.DictReader(io.StringIO(content))]
    return labels, n_classes

# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(dest: Path, out: Path):
    n = len(DATASETS)

    # ── NeurIPS half-column: 3.25 in wide ────────────────────────────────────
    # Slightly increased font sizes for readability
    FS       = 5.5          # font size for all text  (was 4.5)
    FS_TICK  = 4.5          # annotation inside bar   (was 3.5)

    col_widths = [0.72, 0.35, 0.37, 0.37, 0.38, 0.94]
    row_height = 0.26       # taller rows (was 0.22)
    header_pad = 0.22       # more header room (was 0.18)
    fig_w = sum(col_widths)                     # ~3.25 in
    fig_h = row_height * n + header_pad         # ~1.78 in

    fig = plt.figure(figsize=(fig_w, fig_h))

    top_frac    = 1.0 - header_pad / fig_h
    bottom_frac = 0.0

    gs = gridspec.GridSpec(
        nrows=n, ncols=6,
        figure=fig,
        left=0.0, right=1.0,
        top=top_frac, bottom=bottom_frac,
        hspace=0.0, wspace=0.0,
        width_ratios=col_widths,
        height_ratios=[1] * n,
    )

    sep_axes = []

    for ri, ds in enumerate(DATASETS):
        archive = dest / f"{ds['key']}_processed.tar.zst"
        labels, n_classes = [], 0
        if archive.exists():
            labels, n_classes = read_test_labels(archive)
        n_test = len(labels)
        counts = Counter(labels)
        sorted_counts = sorted(counts.values(), reverse=True)

        is_first = ri == 0
        is_last  = ri == n - 1

        # ── text columns ──────────────────────────────────────────────────────
        text_cols = [
            (ds["name"],                          True,  "left",   0.04),
            (ds["task"],  TASK_COLORS[ds["task"]], False, "left",   0.04),
            (ds["domain"],                         False, "left",   0.06),
            (str(n_classes),                       False, "center", 0.5 ),
            (f"{n_test:,}",                        False, "center", 0.5 ),
        ]
        for ci, col_def in enumerate(text_cols):
            if len(col_def) == 4:
                txt, bold, ha, xoff = col_def
                color = TEXT_PRIMARY
            else:
                txt, color, bold, ha, xoff = col_def

            ax = fig.add_subplot(gs[ri, ci])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(xoff, 0.5, txt,
                    transform=ax.transAxes, fontsize=FS,
                    fontweight="bold" if bold else "normal",
                    color=color, va="center", ha=ha,
                    clip_on=True)

            if is_first:
                ax.plot([0, 1], [1, 1], color=DIVIDER_COLOR, lw=0.7,
                        transform=ax.transAxes, clip_on=False)
            ax.plot([0, 1], [0, 0], color=DIVIDER_COLOR,
                    lw=0.7 if is_last else 0.35,
                    ls="-"  if is_last else "--",
                    transform=ax.transAxes, clip_on=False)

            if is_first:
                sep_axes.append((ci, ax))

        # ── bar chart ─────────────────────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[ri, 5])
        if is_first:
            sep_axes.append((5, ax_bar))

        if sorted_counts:
            x = np.arange(len(sorted_counts))
            ax_bar.bar(x, sorted_counts, width=0.9,
                       color=BAR_COLOR, alpha=0.78, linewidth=0)
            ax_bar.set_xlim(-0.5, len(sorted_counts) - 0.5)
            ax_bar.set_ylim(0, max(sorted_counts) * 1.12)
            ax_bar.set_yticks([])
            ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
            # max-count annotation top-left
            ax_bar.text(0.02, 0.95, str(max(sorted_counts)),
                        transform=ax_bar.transAxes, fontsize=FS_TICK,
                        color=TEXT_MUTED, va="top", ha="left")
        else:
            ax_bar.text(0.5, 0.5, "archive not found",
                        transform=ax_bar.transAxes, fontsize=FS_TICK,
                        color=TEXT_MUTED, ha="center", va="center")

        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.spines["bottom"].set_visible(True)
        ax_bar.spines["bottom"].set_color(DIVIDER_COLOR)
        ax_bar.spines["bottom"].set_linewidth(0.35 if not is_last else 0.7)
        ax_bar.set_facecolor("none")
        if is_first:
            ax_bar.plot([0, 1], [1, 1], color=DIVIDER_COLOR, lw=0.7,
                        transform=ax_bar.transAxes, clip_on=False)
        ax_bar.plot([0, 1], [0, 0], color=DIVIDER_COLOR,
                    lw=0.7 if is_last else 0.35,
                    ls="-"  if is_last else "--",
                    transform=ax_bar.transAxes, clip_on=False)

    # ── column headers (placed in figure coords above the grid) ───────────────
    headers = ["Dataset", "Task", "Domain", "Classes", "Size",
               "Class distrib. (test)"]
    has     = ["left", "left", "left", "center", "center", "left"]
    x_offs  = [0.04,   0.04,   0.06,   0.5,      0.5,      0.02]
    for ci, (hdr, ha, xoff) in enumerate(zip(headers, has, x_offs)):
        ax_ref = fig.add_subplot(gs[0, ci])
        ax_ref.axis("off")
        ax_ref.text(xoff, 1.22, hdr,
                    transform=ax_ref.transAxes,
                    fontsize=FS, fontweight="bold", color=TEXT_PRIMARY,
                    va="bottom", ha=ha)
        ax_ref.set_zorder(10)

    # ── vertical separators ───────────────────────────────────────────────────
    fig.canvas.draw()
    for ci, ax in sep_axes:
        bbox = ax.get_position()
        fig.add_artist(plt.Line2D(
            [bbox.x0, bbox.x0], [bottom_frac, top_frac + header_pad / fig_h * 0.75],
            transform=fig.transFigure,
            color=DIVIDER_COLOR, linewidth=0.5, zorder=5,
        ))
    bbox_last = fig.axes[-1].get_position()
    fig.add_artist(plt.Line2D(
        [bbox_last.x1, bbox_last.x1],
        [bottom_frac, top_frac + header_pad / fig_h * 0.75],
        transform=fig.transFigure,
        color=DIVIDER_COLOR, linewidth=0.5, zorder=5,
    ))

    plt.savefig(out, bbox_inches="tight", pad_inches=0, dpi=300,
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved → {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dest", default="~/data/processed")
    parser.add_argument("--out",  default=None)
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    out  = "./results_analysis_neurips2026/benchmark_table.pdf"
    print(f"Reading archives from {dest} …")
    make_figure(dest, out)

if __name__ == "__main__":
    main()