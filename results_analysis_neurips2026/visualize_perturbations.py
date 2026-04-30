#!/usr/bin/env python3
"""
visualize_perturbations_latex.py — Load images per dataset per perturbation type,
compute dataset-wide attack intensity statistics, and output:

  1. figures/perturbations/  — folder of saved sample images (JPEG, 224×224)
       naming: {dataset}_{perturbation_key}.jpg
  2. perturbation_grid.tex   — LaTeX figure (portrait, fits NeurIPS \textwidth)
  3. perturbation_stats.tex  — LaTeX table  (portrait, fits NeurIPS \textwidth)

Copy the figures/ folder and .tex files into your manuscript tree and
\input{} them directly.
"""

import csv
import io
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths  (edit these to match your setup)
# ---------------------------------------------------------------------------

DATA_ROOT  = Path(os.path.expanduser("~/data"))
ADV_ROOT   = DATA_ROOT / "adversarial"
CLEAN_ROOT = DATA_ROOT / "processed"

OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/perturbation_output"))
IMG_DIR    = OUTPUT_DIR / "figures" / "perturbations"

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

PERTURBATIONS = [
    ("clean",        None,                                                        "clean"),
    ("L∞ ε=4/255",  "zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard", "linf4"),
    ("L∞ ε=8/255",  "zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard", "linf8"),
    ("L∞ ε=30/255", "zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard","linf30"),
    ("L2 ε=2",      "zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard",   "l2e2"),
    ("L2 ε=8",      "zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard",   "l2e8"),
    ("L1 ε=75",     "zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard",  "l1e75"),
    ("L1 ε=300",    "zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard",  "l1e300"),
    ("Common s=3",  "common/common_severity3",                                     "common3"),
]

DATASET_DISPLAY = {
    "caltech101":                 "Caltech-101",
    "fgvc-aircraft-2013b":        "FGVC Aircraft",
    "flowers-102":                "Flowers-102",
    "oxford-iiit-pet":            "Oxford Pet",
    "stanford_cars":              "Stanford Cars",
    "uc-merced-land-use-dataset": "UC Merced",
}

# Short keys for filenames (no spaces/slashes)
DATASET_KEY = {
    "caltech101":                 "caltech",
    "fgvc-aircraft-2013b":        "aircraft",
    "flowers-102":                "flowers",
    "oxford-iiit-pet":            "pet",
    "stanford_cars":              "cars",
    "uc-merced-land-use-dataset": "merced",
}

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard"); sys.exit(1)


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def load_archive_full(archive_path: Path) -> tuple[list[str], dict[str, bytes]]:
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    raw_by_name, rows = {}, []
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        for member in tar.getmembers():
            if member.name.startswith("test/") and (
                    member.name.endswith(".png") or member.name.endswith(".jpg")):
                raw_by_name[Path(member.name).name] = tar.extractfile(member).read()
        for cand in ["test/labels.csv", "labels.csv"]:
            try:
                f    = tar.extractfile(tar.getmember(cand))
                rows = list(csv.DictReader(io.TextIOWrapper(f)))
                break
            except KeyError:
                continue
    return [r["filename"] for r in rows], raw_by_name


def find_archive(dataset: str, adv_subpath: str) -> Path | None:
    folder = ADV_ROOT / adv_subpath
    if not folder.exists():
        return None
    matches = sorted(folder.glob(f"{dataset}*_processed.tar.zst"))
    return matches[0] if matches else None


def find_attacked_index(clean_fnames, clean_imgs, ref_fnames, ref_imgs) -> int:
    for i, (cf, rf) in enumerate(zip(clean_fnames, ref_fnames)):
        rc = clean_imgs.get(cf)
        rr = ref_imgs.get(rf)
        if rc is None or rr is None:
            continue
        a = np.array(Image.open(io.BytesIO(rc)).convert("RGB"))
        b = np.array(Image.open(io.BytesIO(rr)).convert("RGB"))
        if not np.array_equal(a, b):
            return i
    return 0


# ---------------------------------------------------------------------------
# Per-image & dataset-wide diff stats
# ---------------------------------------------------------------------------

def image_diff_stats(clean: Image.Image, perturbed: Image.Image) -> dict:
    a = np.array(clean.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32)
    b = np.array(perturbed.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32)
    diff = np.abs(a - b)
    return {
        "identical":   bool(np.array_equal(a, b)),
        "max_diff":    float(diff.max()),
        "mean_diff":   float(diff.mean()),
        "pct_changed": float((diff.sum(axis=-1) > 0).mean() * 100),
    }


def compute_dataset_wide_stats(clean_fnames, clean_imgs, adv_fnames, adv_imgs) -> dict:
    mean_diffs, max_diffs, pct_changed = [], [], []
    n_identical, n_compared = 0, 0
    for cf, af in zip(clean_fnames, adv_fnames):
        rc, ra = clean_imgs.get(cf), adv_imgs.get(af)
        if rc is None or ra is None:
            continue
        a = np.array(Image.open(io.BytesIO(rc)).convert("RGB")).astype(np.float32)
        b = np.array(Image.open(io.BytesIO(ra)).convert("RGB")).astype(np.float32)
        diff = np.abs(a - b)
        n_compared += 1
        if np.array_equal(a, b):
            n_identical += 1
            mean_diffs.append(0.0); max_diffs.append(0.0); pct_changed.append(0.0)
        else:
            mean_diffs.append(float(diff.mean()))
            max_diffs.append(float(diff.max()))
            pct_changed.append(float((diff.sum(axis=-1) > 0).mean() * 100))
    if n_compared == 0:
        return None
    return {
        "n_images": n_compared, "n_identical": n_identical,
        "pct_identical": n_identical / n_compared * 100,
        "mean_diff_avg": float(np.mean(mean_diffs)),
        "mean_diff_std": float(np.std(mean_diffs)),
        "max_diff_avg":  float(np.mean(max_diffs)),
        "max_diff_std":  float(np.std(max_diffs)),
        "pct_changed_avg": float(np.mean(pct_changed)),
        "pct_changed_std": float(np.std(pct_changed)),
    }


# ---------------------------------------------------------------------------
# Main loading
# ---------------------------------------------------------------------------

def load_images_for_dataset(dataset: str):
    clean_archive = CLEAN_ROOT / f"{dataset}_processed.tar.zst"
    ref_archive   = find_archive(dataset, PERTURBATIONS[1][1])
    if not clean_archive.exists() or ref_archive is None:
        return None, {}, {}
    clean_fnames, clean_imgs = load_archive_full(clean_archive)
    ref_fnames,   ref_imgs   = load_archive_full(ref_archive)
    idx = find_attacked_index(clean_fnames, clean_imgs, ref_fnames, ref_imgs)
    print(f"    Using index {idx} (successfully attacked)")

    def get_img(fnames, imgs):
        fname = fnames[idx] if idx < len(fnames) else None
        raw   = imgs.get(fname) if fname else None
        return Image.open(io.BytesIO(raw)).convert("RGB") if raw else None

    clean_img    = get_img(clean_fnames, clean_imgs)
    perturb_imgs = {}
    ds_stats     = {}
    for label, subpath, _ in PERTURBATIONS[1:]:
        archive = find_archive(dataset, subpath)
        if archive is None:
            perturb_imgs[label] = None; ds_stats[label] = None; continue
        fnames, imgs = load_archive_full(archive)
        perturb_imgs[label] = get_img(fnames, imgs)
        print(f"    {label:<14} computing dataset-wide stats...")
        ds_stats[label] = compute_dataset_wide_stats(clean_fnames, clean_imgs, fnames, imgs)
    return clean_img, perturb_imgs, ds_stats


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------

print("Loading images & computing dataset-wide statistics...\n")
all_images    = {}   # all_images[dataset][perturb_key] = PIL Image or None
dataset_stats = {}   # dataset_stats[dataset][label] = stats dict or None

for dataset in DATASETS:
    print(f"  {dataset}")
    all_images[dataset]    = {}
    dataset_stats[dataset] = {}

    clean_img, perturb_imgs, ds_stats = load_images_for_dataset(dataset)

    all_images[dataset]["clean"] = clean_img
    dataset_stats[dataset]["clean"] = None

    for label, _, key in PERTURBATIONS[1:]:
        all_images[dataset][label] = perturb_imgs.get(label)
        dataset_stats[dataset][label] = ds_stats.get(label)

        s = dataset_stats[dataset][label]
        img = all_images[dataset][label]
        status = "✓" if img else "N/A"
        if s:
            status += (f"  [N={s['n_images']}, "
                       f"mean_Δ={s['mean_diff_avg']:.2f}±{s['mean_diff_std']:.2f}, "
                       f"max_Δ={s['max_diff_avg']:.1f}±{s['max_diff_std']:.1f}, "
                       f"{s['pct_changed_avg']:.1f}% px changed, "
                       f"{s['pct_identical']:.1f}% identical]")
        print(f"    {label:<14} {status}")


# ---------------------------------------------------------------------------
# 1) Save images to folder
# ---------------------------------------------------------------------------

IMG_DIR.mkdir(parents=True, exist_ok=True)
saved_count = 0

for dataset in DATASETS:
    dk = DATASET_KEY[dataset]
    for label, _, pk in PERTURBATIONS:
        img = all_images[dataset].get(label)
        if img is not None:
            img_resized = img.resize((224, 224), Image.BICUBIC)
            fname = f"{dk}_{pk}.jpg"
            img_resized.save(IMG_DIR / fname, format="JPEG", quality=92)
            saved_count += 1

print(f"\n✓ Saved {saved_count} images → {IMG_DIR}/")


# ---------------------------------------------------------------------------
# 2) Generate LaTeX: image grid figure
# ---------------------------------------------------------------------------

def latex_safe(s: str) -> str:
    """Escape special LaTeX characters."""
    return (s.replace("∞", r"$\infty$")
             .replace("ε=", r"$\varepsilon{=}$")
             .replace("_", r"\_"))


def generate_grid_tex() -> str:
    n_ds = len(DATASETS)
    # Column spec: one label column + one image column per dataset
    # Using @{} to kill inter-column padding, \hspace for tight control
    col_spec = "l" + "c" * n_ds

    lines = []
    lines.append(r"% perturbation_grid.tex — \input{} this in your NeurIPS manuscript")
    lines.append(r"% Requires: \usepackage{graphicx}, images in figures/perturbations/")
    lines.append(r"\begin{figure*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{1.5pt}")
    lines.append(r"  \renewcommand{\arraystretch}{0.2}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")

    # Header row
    hdr_cells = " & ".join(
        r"\textbf{" + latex_safe(DATASET_DISPLAY[ds]) + r"}"
        for ds in DATASETS
    )
    lines.append(r"    & " + hdr_cells + r" \\")
    lines.append(r"    \midrule")

    # Image rows
    img_width = f"{5.5 / (n_ds + 0.6):.2f}"  # inches, to fit \textwidth with label col

    for label, _, pk in PERTURBATIONS:
        cells = []
        for ds in DATASETS:
            dk = DATASET_KEY[ds]
            fname = f"figures/perturbations/{dk}_{pk}.jpg"
            cells.append(
                r"\includegraphics[width=" + img_width + r"in]{" + fname + r"}"
            )
        row_label = latex_safe(label)
        lines.append(r"    \raisebox{0.35in}{\rotatebox{90}{\scriptsize " + row_label + r"}} & "
                      + " & ".join(cells) + r" \\[2pt]")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Representative adversarial perturbations across datasets. "
                 r"Each row shows the same test image under a different attack "
                 r"($L_\infty$, $L_2$, $L_1$ norm-bounded, and common corruptions). "
                 r"Surrogate: CLIP ViT-H/14 (LAION-2B).}")
    lines.append(r"  \label{fig:perturbation-grid}")
    lines.append(r"\end{figure*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3) Generate LaTeX: statistics table
# ---------------------------------------------------------------------------

def generate_stats_tex() -> str:
    n_ds = len(DATASETS)
    perturb_labels = [(label, pk) for label, _, pk in PERTURBATIONS if label != "clean"]

    col_spec = "l" + "c" * n_ds

    lines = []
    lines.append(r"% perturbation_stats.tex — \input{} this in your NeurIPS manuscript")
    lines.append(r"% Requires: \usepackage{booktabs, multirow}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Dataset-wide attack intensity statistics (all test images). "
                 r"Mean/std computed per-image then averaged. "
                 r"Surrogate: CLIP ViT-H/14 (LAION-2B).}")
    lines.append(r"  \label{tab:attack-stats}")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")

    # Header
    hdr_cells = " & ".join(
        r"\textbf{" + latex_safe(DATASET_DISPLAY[ds]) + r"}"
        for ds in DATASETS
    )
    lines.append(r"    \textbf{Perturbation} & " + hdr_cells + r" \\")
    lines.append(r"    \midrule")

    # Data rows
    for label, pk in perturb_labels:
        cells = []
        for ds in DATASETS:
            s = dataset_stats[ds].get(label)
            if s is None:
                cells.append("---")
            else:
                cell = (
                    r"\makecell[c]{"
                    f"$\\bar{{\\Delta}}$={s['mean_diff_avg']:.2f}$\\pm${s['mean_diff_std']:.2f}"
                    r" \\ "
                    f"max={s['max_diff_avg']:.1f}$\\pm${s['max_diff_std']:.1f}"
                    r" \\ "
                    f"px={s['pct_changed_avg']:.1f}\\%"
                    r" \\ "
                    f"id={s['pct_identical']:.0f}\\%"
                    r"}"
                )
                cells.append(cell)

        row_label = latex_safe(label)
        lines.append(r"    " + row_label + " & " + " & ".join(cells) + r" \\")

        # Add a light separator between norm groups
        if label in ("L∞ ε=30/255", "L2 ε=8", "L1 ε=300"):
            lines.append(r"    \addlinespace[3pt]")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Write .tex files
# ---------------------------------------------------------------------------

grid_tex  = generate_grid_tex()
stats_tex = generate_stats_tex()

(OUTPUT_DIR / "perturbation_grid.tex").write_text(grid_tex, encoding="utf-8")
(OUTPUT_DIR / "perturbation_stats.tex").write_text(stats_tex, encoding="utf-8")

print(f"✓ perturbation_grid.tex  → {OUTPUT_DIR / 'perturbation_grid.tex'}")
print(f"✓ perturbation_stats.tex → {OUTPUT_DIR / 'perturbation_stats.tex'}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Output structure:

    {OUTPUT_DIR}/
    ├── figures/
    │   └── perturbations/
    │       ├── caltech_clean.jpg
    │       ├── caltech_linf4.jpg
    │       ├── ...
    │       └── merced_common3.jpg
    ├── perturbation_grid.tex
    └── perturbation_stats.tex

  In your NeurIPS manuscript:

    \\usepackage{{graphicx, booktabs, makecell}}

    \\input{{perturbation_grid}}    % full-width figure
    \\input{{perturbation_stats}}   % full-width table

  Make sure figures/perturbations/ is accessible from your
  LaTeX compile directory.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")