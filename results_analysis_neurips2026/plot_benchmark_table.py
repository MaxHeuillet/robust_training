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
from matplotlib.ticker import MaxNLocator
import numpy as np

try:
    import zstandard as zstd
except ImportError:
    raise SystemExit("Install zstandard:  pip install zstandard")

# ── dataset metadata ──────────────────────────────────────────────────────────

DATASETS = [
    dict(key="uc-merced-land-use-dataset", name="UC Merced",      task="coarse-grained", domain="satellite"),
    dict(key="caltech101",                 name="Caltech-101",     task="coarse-grained", domain="natural"),
    dict(key="flowers-102",                name="Flowers-102",     task="fine-grained",   domain="natural"),
    dict(key="oxford-iiit-pet",            name="Oxford-IIIT Pet", task="fine-grained",   domain="natural"),
    dict(key="stanford_cars",              name="Stanford Cars",   task="fine-grained",   domain="natural"),
    dict(key="fgvc-aircraft-2013b",        name="FGVC Aircraft",   task="fine-grained",   domain="natural"),
]

TASK_COLORS   = {"coarse-grained": "#888780", "fine-grained": "#7F77DD"}
BAR_COLOR     = "#7F77DD"   # unified for all distributions
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

    # cols: dataset | task | domain | classes | test obs | distribution
    col_widths = [0.9, 0.8, 0.55, 0.42, 0.42, 3.6]
    row_height = 0.38
    fig_w      = sum(col_widths) + 0.15
    fig_h      = row_height * n + 0.38

    fig = plt.figure(figsize=(fig_w, fig_h))

    top_frac    = 1.0 - 0.05 / fig_h
    bottom_frac = 0.04 / fig_h

    gs = gridspec.GridSpec(
        nrows=n, ncols=6,
        figure=fig,
        left=0.0, right=1.0,
        top=top_frac, bottom=bottom_frac,
        hspace=0.0, wspace=0.0,
        width_ratios=col_widths,
        height_ratios=[1] * n,
    )

    # vertical separator x-positions in figure coords — computed after first axes placed
    # we draw them via fig.lines after the loop using the stored axes bbox info
    sep_axes = []   # (col_idx, ax) for first row only

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
            (ds["name"],                         True,  "left",   0.05),
            (ds["task"],  TASK_COLORS[ds["task"]], False, "left",  0.06),
            (ds["domain"],                        False, "left",   0.06),
            (str(n_classes),                      False, "center", 0.5 ),
            (f"{n_test:,}",                       False, "center", 0.5 ),
        ]
        for ci, col_def in enumerate(text_cols):
            if len(col_def) == 4:
                txt, bold, ha, xoff = col_def
                color = TEXT_PRIMARY
            else:
                txt, color, bold, ha, xoff = col_def

            ax = fig.add_subplot(gs[ri, ci])
            ax.axis("off")
            ax.text(xoff, 0.5, txt,
                    transform=ax.transAxes, fontsize=6.5,
                    fontweight="bold" if bold else "normal",
                    color=TEXT_PRIMARY, va="center", ha=ha)

            # horizontal rules
            if is_first:
                ax.plot([0, 1], [1, 1], color=DIVIDER_COLOR, lw=0.8,
                        transform=ax.transAxes, clip_on=False)
            ax.plot([0, 1], [0, 0], color=DIVIDER_COLOR,
                    lw=0.8 if is_last else 0.4,
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
            ax_bar.bar(x, sorted_counts, width=0.85,
                       color=BAR_COLOR, alpha=0.78, linewidth=0)
            ax_bar.set_xlim(-1, len(sorted_counts))
            ax_bar.set_ylim(0, max(sorted_counts) * 1.55)
            ax_bar.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2, prune="both"))
            ax_bar.tick_params(axis="y", labelsize=5, length=2,
                               pad=1, color=DIVIDER_COLOR, direction="in")
            ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
            for lbl in ax_bar.get_yticklabels():
                lbl.set_color(TEXT_PRIMARY)
                lbl.set_horizontalalignment("left")
            ax_bar.yaxis.set_tick_params(which="both", left=False)
            ax_bar.set_yticks(ax_bar.get_yticks())
            ax_bar.yaxis.set_label_position("right")
            ax_bar.yaxis.set_tick_params(pad=0)

            mean_obs = n_test / n_classes if n_classes else 0
            cv = np.std(sorted_counts) / mean_obs if mean_obs > 0 else 0
            # if cv < 0.05:
            #     ann = f"balanced ({int(round(mean_obs))}/class)"
            # elif cv < 0.3:
            #     ann = f"mildly imbalanced  (CV={cv:.2f})"
            # else:
            #     ann = f"imbalanced  (CV={cv:.2f})"
            # ax_bar.text(0.5, 0.97, ann, transform=ax_bar.transAxes,
            #             fontsize=5, color=TEXT_MUTED, ha="center", va="top")
        else:
            ax_bar.text(0.5, 0.5, "archive not found",
                        transform=ax_bar.transAxes, fontsize=6,
                        color=TEXT_MUTED, ha="center", va="center")

        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.spines["bottom"].set_visible(True)
        ax_bar.spines["bottom"].set_color(DIVIDER_COLOR)
        ax_bar.spines["bottom"].set_linewidth(0.4 if not is_last else 0.8)
        ax_bar.set_facecolor("none")
        if is_first:
            ax_bar.plot([0, 1], [1, 1], color=DIVIDER_COLOR, lw=0.8,
                        transform=ax_bar.transAxes, clip_on=False)
        ax_bar.plot([0, 1], [0, 0], color=DIVIDER_COLOR,
                    lw=0.8 if is_last else 0.4,
                    ls="-"  if is_last else "--",
                    transform=ax_bar.transAxes, clip_on=False)

    # ── column headers ────────────────────────────────────────────────────────
    headers = ["Dataset", "Task", "Domain", "Classes", "Nb. Obs",
               "Test class distribution  (sorted · 1 bar = 1 class)"]
    x_offs  = [0.05, 0.06, 0.06, 0.5, 0.5, 0.01]
    for ci, (hdr, xoff) in enumerate(zip(headers, x_offs)):
        ax_h = fig.add_subplot(gs[0, ci])
        ax_h.axis("off")
        ha = "center" if xoff == 0.5 else "left"
        ax_h.text(xoff, 1.28, hdr,
                  transform=ax_h.transAxes,
                  fontsize=6.5, fontweight="bold", color=TEXT_PRIMARY,
                  va="bottom", ha=ha)
        ax_h.set_zorder(10)

    # ── vertical separators between all columns ───────────────────────────────
    # draw after layout is committed so bbox is correct
    fig.canvas.draw()
    for ci, ax in sep_axes:
        bbox = ax.get_position()
        # draw at the LEFT edge of this column (right edge of previous)
        x = bbox.x0
        fig.add_artist(plt.Line2D(
            [x, x], [bottom_frac, top_frac],
            transform=fig.transFigure,
            color=DIVIDER_COLOR, linewidth=0.6, zorder=5,
        ))
    # also draw rightmost border
    bbox_last = fig.axes[-1].get_position()
    x_right = bbox_last.x1
    fig.add_artist(plt.Line2D(
        [x_right, x_right], [bottom_frac, top_frac],
        transform=fig.transFigure,
        color=DIVIDER_COLOR, linewidth=0.6, zorder=5,
    ))

    plt.savefig(out, bbox_inches="tight", dpi=300,
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
    out  = "/Users/maximeheuillet/Desktop/robust_training/results_analysis_neurips2026/benchmark_table.pdf" #Path(args.out).expanduser() if args.out else dest / "benchmark_table.pdf"
    print(f"Reading archives from {dest} …")
    make_figure(dest, out)

if __name__ == "__main__":
    main()