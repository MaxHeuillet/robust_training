#!/usr/bin/env python3
"""
visualize_perturbations.py — Load images per dataset per perturbation type,
compute dataset-wide attack intensity statistics, and render:
  1. A transposed HTML image grid  (perturbations × datasets)
  2. A dataset-wide average Δ statistics table

Output: perturbation_grid.html  (open in browser)
"""

import base64
import csv
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT  = Path(os.path.expanduser("~/data"))
ADV_ROOT   = DATA_ROOT / "adversarial"
CLEAN_ROOT = DATA_ROOT / "processed"
OUTPUT_HTML = Path(os.path.expanduser("~/Desktop/perturbation_grid.html"))

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

PERTURBATIONS = [
    ("clean",        None),
    ("L∞ ε=4/255",  "zeroshot_clip_vith14_laion2b/linf_eps4_autoattack_standard"),
    ("L∞ ε=8/255",  "zeroshot_clip_vith14_laion2b/linf_eps8_autoattack_standard"),
    ("L∞ ε=30/255", "zeroshot_clip_vith14_laion2b/linf_eps30_autoattack_standard"),
    ("L2 ε=2",      "zeroshot_clip_vith14_laion2b/l2_eps2_autoattack_standard"),
    ("L2 ε=8",      "zeroshot_clip_vith14_laion2b/l2_eps8_autoattack_standard"),
    ("L1 ε=75",     "zeroshot_clip_vith14_laion2b/l1_eps75_autoattack_standard"),
    ("L1 ε=300",    "zeroshot_clip_vith14_laion2b/l1_eps300_autoattack_standard"),
    ("Common s=3",  "common/common_severity3"),
]

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard"); sys.exit(1)


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def load_archive_full(archive_path: Path) -> tuple[list[str], dict[str, bytes]]:
    """Load all test/ images into dict + ordered filenames from labels.csv."""
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
    """
    Find the first index where clean and reference (Linf8) differ —
    i.e. the surrogate was correct and AutoAttack successfully perturbed.
    """
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
# Per-image diff stats
# ---------------------------------------------------------------------------

def image_diff_stats(clean: Image.Image, perturbed: Image.Image) -> dict:
    """Compute pixel-level diff stats between clean and perturbed image."""
    a = np.array(clean.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32)
    b = np.array(perturbed.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32)
    diff = np.abs(a - b)
    identical  = bool(np.array_equal(a, b))
    max_diff   = float(diff.max())
    mean_diff  = float(diff.mean())
    pct_changed = float((diff.sum(axis=-1) > 0).mean() * 100)
    return {
        "identical":   identical,
        "max_diff":    max_diff,
        "mean_diff":   mean_diff,
        "pct_changed": pct_changed,
    }


# ---------------------------------------------------------------------------
# Dataset-wide statistics (all test images)
# ---------------------------------------------------------------------------

def compute_dataset_wide_stats(clean_fnames, clean_imgs, adv_fnames, adv_imgs) -> dict:
    """
    Iterate over ALL paired test images and compute aggregate Δ statistics:
      - mean/std of per-image mean absolute diff
      - mean/std of per-image max diff
      - mean/std of per-image % pixels changed
      - fraction of images that are identical (unattacked)
    """
    mean_diffs  = []
    max_diffs   = []
    pct_changed = []
    n_identical = 0
    n_compared  = 0

    for cf, af in zip(clean_fnames, adv_fnames):
        rc = clean_imgs.get(cf)
        ra = adv_imgs.get(af)
        if rc is None or ra is None:
            continue
        a = np.array(Image.open(io.BytesIO(rc)).convert("RGB")).astype(np.float32)
        b = np.array(Image.open(io.BytesIO(ra)).convert("RGB")).astype(np.float32)
        diff = np.abs(a - b)
        n_compared += 1

        if np.array_equal(a, b):
            n_identical += 1
            mean_diffs.append(0.0)
            max_diffs.append(0.0)
            pct_changed.append(0.0)
        else:
            mean_diffs.append(float(diff.mean()))
            max_diffs.append(float(diff.max()))
            pct_changed.append(float((diff.sum(axis=-1) > 0).mean() * 100))

    if n_compared == 0:
        return None

    return {
        "n_images":         n_compared,
        "n_identical":      n_identical,
        "pct_identical":    n_identical / n_compared * 100,
        "mean_diff_avg":    float(np.mean(mean_diffs)),
        "mean_diff_std":    float(np.std(mean_diffs)),
        "max_diff_avg":     float(np.mean(max_diffs)),
        "max_diff_std":     float(np.std(max_diffs)),
        "pct_changed_avg":  float(np.mean(pct_changed)),
        "pct_changed_std":  float(np.std(pct_changed)),
    }


# ---------------------------------------------------------------------------
# Main loading logic
# ---------------------------------------------------------------------------

def load_images_for_dataset(dataset: str):
    """
    Returns:
      - clean_img (PIL) for the representative sample
      - {perturb_label: perturb_img} for the same sample
      - {perturb_label: dataset_wide_stats_dict}
    """
    clean_archive = CLEAN_ROOT / f"{dataset}_processed.tar.zst"
    ref_archive   = find_archive(dataset, PERTURBATIONS[1][1])  # Linf eps4

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

    clean_img = get_img(clean_fnames, clean_imgs)

    perturb_imgs = {}
    dataset_stats = {}

    for label, subpath in PERTURBATIONS[1:]:
        archive = find_archive(dataset, subpath)
        if archive is None:
            perturb_imgs[label]  = None
            dataset_stats[label] = None
            continue

        fnames, imgs = load_archive_full(archive)
        perturb_imgs[label] = get_img(fnames, imgs)

        # Compute dataset-wide stats over all test images
        print(f"    {label:<14} computing dataset-wide stats over all test images...")
        dataset_stats[label] = compute_dataset_wide_stats(
            clean_fnames, clean_imgs, fnames, imgs
        )

    return clean_img, perturb_imgs, dataset_stats


def pil_to_b64(img: Image.Image, size=(224, 224)) -> str:
    img = img.resize(size, Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Build grid data
# ---------------------------------------------------------------------------

print("Loading images & computing dataset-wide statistics...\n")
grid           = {}   # grid[dataset][label] = b64 string
sample_stats   = {}   # per-sample diff stats
dataset_stats  = {}   # dataset-wide aggregate stats

for dataset in DATASETS:
    print(f"  {dataset}")
    grid[dataset]          = {}
    sample_stats[dataset]  = {}
    dataset_stats[dataset] = {}

    clean_img, perturb_imgs, ds_stats = load_images_for_dataset(dataset)

    grid[dataset]["clean"]          = pil_to_b64(clean_img) if clean_img else None
    sample_stats[dataset]["clean"]  = None
    dataset_stats[dataset]["clean"] = None

    for label, _ in PERTURBATIONS[1:]:
        img = perturb_imgs.get(label)
        grid[dataset][label] = pil_to_b64(img) if img else None

        if img is not None and clean_img is not None:
            sample_stats[dataset][label] = image_diff_stats(clean_img, img)
        else:
            sample_stats[dataset][label] = None

        dataset_stats[dataset][label] = ds_stats.get(label)

        # Console output
        s = dataset_stats[dataset][label]
        status = "✓" if img else "N/A"
        if s:
            status += (f"  [N={s['n_images']}, "
                       f"mean_Δ={s['mean_diff_avg']:.2f}±{s['mean_diff_std']:.2f}, "
                       f"max_Δ={s['max_diff_avg']:.1f}±{s['max_diff_std']:.1f}, "
                       f"{s['pct_changed_avg']:.1f}% px changed, "
                       f"{s['pct_identical']:.1f}% identical]")
        print(f"    {label:<14} {status}")


# ---------------------------------------------------------------------------
# Render HTML — TRANSPOSED layout: rows = perturbations, cols = datasets
# ---------------------------------------------------------------------------

DATASET_DISPLAY = {
    "caltech101":                 "Caltech-101",
    "fgvc-aircraft-2013b":        "FGVC Aircraft",
    "flowers-102":                "Flowers-102",
    "oxford-iiit-pet":            "Oxford Pet",
    "stanford_cars":              "Stanford Cars",
    "uc-merced-land-use-dataset": "UC Merced",
}

perturb_labels = [p[0] for p in PERTURBATIONS]


def diff_badge(stats: dict | None, label: str) -> str:
    if label == "clean" or stats is None:
        return ""
    if stats["identical"]:
        return '<div class="badge identical">⚠ identical</div>'
    color = "#b8f050" if stats["max_diff"] > 5 else "#f0a830"
    return (
        f'<div class="badge ok" style="color:{color}">'
        f'Δmax={stats["max_diff"]:.1f} · '
        f'mean={stats["mean_diff"]:.2f} · '
        f'{stats["pct_changed"]:.0f}%'
        f'</div>'
    )


def render_cell(b64, stats, label):
    badge = diff_badge(stats, label)
    if b64 is None:
        return f'<td><div class="missing">N/A</div>{badge}</td>'
    return f'<td><img src="data:image/jpeg;base64,{b64}" loading="lazy">{badge}</td>'


# ---- Transposed image grid: rows = perturbations, columns = datasets ----
header_cells = "".join(
    f"<th>{DATASET_DISPLAY.get(ds, ds)}</th>" for ds in DATASETS
)

rows_html = ""
for label in perturb_labels:
    cells = "".join(
        render_cell(
            grid[ds].get(label),
            sample_stats[ds].get(label),
            label,
        )
        for ds in DATASETS
    )
    rows_html += f'<tr><td class="ds">{label}</td>{cells}</tr>\n'


# ---- Dataset-wide statistics table ----
def fmt(val, precision=2):
    if val is None:
        return "—"
    return f"{val:.{precision}f}"


stats_header = "".join(
    f"<th>{DATASET_DISPLAY.get(ds, ds)}</th>" for ds in DATASETS
)

# We'll show 3 metrics in subtables: mean Δ, max Δ, % px changed
STAT_METRICS = [
    ("Mean Δ (avg ± std)",  "mean_diff_avg",   "mean_diff_std",   2),
    ("Max Δ (avg ± std)",   "max_diff_avg",    "max_diff_std",    1),
    ("% Pixels Changed",    "pct_changed_avg", "pct_changed_std", 1),
    ("% Identical Images",  "pct_identical",   None,              1),
]

stats_rows_html = ""
for label in perturb_labels:
    if label == "clean":
        continue
    # One row per perturbation, each cell shows a compact multi-line stat block
    cells = ""
    for ds in DATASETS:
        s = dataset_stats[ds].get(label)
        if s is None:
            cells += '<td class="stat-cell">—</td>'
        else:
            cells += (
                f'<td class="stat-cell">'
                f'<span class="stat-line">mean Δ <b>{s["mean_diff_avg"]:.2f}</b> ± {s["mean_diff_std"]:.2f}</span>'
                f'<span class="stat-line">max Δ  <b>{s["max_diff_avg"]:.1f}</b> ± {s["max_diff_std"]:.1f}</span>'
                f'<span class="stat-line">px chg <b>{s["pct_changed_avg"]:.1f}%</b> ± {s["pct_changed_std"]:.1f}</span>'
                f'<span class="stat-line ident">{s["pct_identical"]:.1f}% identical ({s["n_identical"]}/{s["n_images"]})</span>'
                f'</td>'
            )
    stats_rows_html += f'<tr><td class="ds">{label}</td>{cells}</tr>\n'


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Perturbation Visualizer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0c0c0c;color:#ddd;font-family:'IBM Plex Sans',sans-serif;padding:40px 28px}}
  h1{{font-family:'IBM Plex Mono',monospace;font-size:18px;color:#b8f050;border-left:3px solid #b8f050;
      padding-left:14px;margin-bottom:6px}}
  h2{{font-family:'IBM Plex Mono',monospace;font-size:15px;color:#7ac030;border-left:3px solid #1e2e10;
      padding-left:14px;margin:40px 0 10px 0}}
  .sub{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#444;padding-left:17px;margin-bottom:28px}}
  .wrap{{overflow-x:auto;margin-bottom:20px}}

  /* -- Shared table styles -- */
  table{{border-collapse:collapse;width:max-content}}
  thead tr{{background:#111;border-bottom:2px solid #1a1a1a}}
  th{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555;text-transform:uppercase;
      letter-spacing:.07em;padding:10px 10px;text-align:center;white-space:nowrap}}
  th:first-child{{text-align:left;min-width:110px;color:#444}}
  tbody tr{{border-bottom:1px solid #141414}}
  tbody tr:hover{{background:#111}}
  td.ds{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;color:#999;
         padding:10px 14px;white-space:nowrap;border-right:1px solid #1a1a1a;vertical-align:middle}}

  /* -- Image grid table -- */
  .grid-table td{{padding:8px 8px;text-align:center;vertical-align:middle;border-right:1px solid #141414}}
  .grid-table td:last-child{{border-right:none}}
  .grid-table td img{{display:block;width:160px;height:160px;object-fit:cover;border-radius:3px;
          border:1px solid #1e1e1e;transition:transform .2s,border-color .2s;cursor:zoom-in}}
  .grid-table td img:hover{{transform:scale(1.5);border-color:#b8f050;z-index:10;position:relative}}
  .grid-table th{{min-width:176px}}
  .missing{{width:160px;height:160px;display:flex;align-items:center;justify-content:center;
            font-family:'IBM Plex Mono',monospace;font-size:10px;color:#2a2a2a;
            background:#0a0a0a;border-radius:3px;border:1px dashed #1e1e1e;margin:auto}}
  .badge{{font-family:'IBM Plex Mono',monospace;font-size:8px;margin-top:4px;
          padding:2px 5px;border-radius:2px;text-align:center;letter-spacing:.03em}}
  .badge.identical{{background:#2a0a0a;color:#f05050;border:1px solid #3a1010}}
  .badge.ok{{background:#0e1a08;border:1px solid #1e2e10}}

  /* -- Stats table -- */
  .stats-table th{{min-width:150px}}
  .stats-table td{{padding:8px 12px;text-align:left;vertical-align:top;border-right:1px solid #141414}}
  .stats-table td:last-child{{border-right:none}}
  .stat-cell{{font-family:'IBM Plex Mono',monospace;font-size:10px;line-height:1.7}}
  .stat-line{{display:block;color:#888}}
  .stat-line b{{color:#ddd}}
  .stat-line.ident{{color:#f0a830;font-size:9px;margin-top:2px}}
</style>
</head>
<body>

<h1>Perturbation Visualizer</h1>
<p class="sub">CLIP ViT-H/14 LAION-2B surrogate · one representative test image per dataset · RobustGenBench</p>

<h2>Sample Image Grid</h2>
<div class="wrap">
<table class="grid-table">
  <thead><tr><th>Perturbation</th>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>

<h2>Dataset-Wide Attack Intensity (all test images)</h2>
<div class="wrap">
<table class="stats-table">
  <thead><tr><th>Perturbation</th>{stats_header}</tr></thead>
  <tbody>{stats_rows_html}</tbody>
</table>
</div>

</body>
</html>"""

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"\n✓ Saved → {OUTPUT_HTML}")
print("  Open in your browser to view the grid.")