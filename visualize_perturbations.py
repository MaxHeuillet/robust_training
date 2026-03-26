#!/usr/bin/env python3
"""
visualize_perturbations.py — Load one image per dataset per perturbation type
and render an HTML grid. Extracts only the first image from each archive
in-memory — no disk space needed.

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
CLEAN_ROOT = Path("/tmp/robustgenbench/data_processed")
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
    ("L∞ ε=8/255",  "zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard"),
    ("L∞ ε=30/255", "zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard"),
    ("L2 ε=2",      "zeroshot_clip_vitb16_laion2b/l2_eps2_autoattack_standard"),
    ("L2 ε=8",      "zeroshot_clip_vitb16_laion2b/l2_eps8_autoattack_standard"),
    ("L1 ε=75",     "zeroshot_clip_vitb16_laion2b/l1_eps75_autoattack_standard"),
    ("L1 ε=300",    "zeroshot_clip_vitb16_laion2b/l1_eps300_autoattack_standard"),
    ("Common s=3",  "common/common_severity3"),
]

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard"); sys.exit(1)


# ---------------------------------------------------------------------------
# Extract ONLY the first image from a .tar.zst — fully in-memory
# ---------------------------------------------------------------------------

def first_image_from_archive(archive_path: Path) -> Image.Image | None:
    """
    Opens the tar.zst archive, reads labels.csv to get the first filename,
    then extracts only that one file — all in memory, nothing written to disk.
    Only considers images under test/ to avoid train/val filename collisions.
    """
    if not archive_path.exists():
        return None
    try:
        with open(archive_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buf = io.BytesIO(reader.read())

        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:") as tar:
            csv_member = tar.getmember("test/labels.csv")
            csv_f      = tar.extractfile(csv_member)
            rows       = list(csv.DictReader(io.TextIOWrapper(csv_f)))
            if not rows:
                return None
            first_filename = rows[0]["filename"]
            # Explicitly use test/ prefix to avoid train/val filename collisions
            img_member = tar.getmember(f"test/{first_filename}")
            img_f      = tar.extractfile(img_member)
            return Image.open(io.BytesIO(img_f.read())).convert("RGB")

    except Exception as e:
        print(f"    ⚠ {archive_path.name}: {e}")
        return None


def find_archive(dataset: str, adv_subpath: str) -> Path | None:
    folder  = ADV_ROOT / adv_subpath
    if not folder.exists():
        return None
    matches = sorted(folder.glob(f"{dataset}*_processed.tar.zst"))
    return matches[0] if matches else None


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


def find_attacked_index(clean_fnames, clean_imgs, ref_fnames, ref_imgs) -> int:
    """
    Find the first index where clean and reference (Linf8) differ —
    i.e. the surrogate was correct and AutoAttack successfully perturbed the image.
    Returns 0 as fallback.
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


def load_images_for_dataset(dataset: str) -> tuple[Image.Image | None, dict[str, Image.Image | None]]:
    """
    Returns (clean_img, {perturb_label: perturb_img}) all from the same image index.
    The index is chosen to be an unattacked image (clean == Linf8 pixel-identical).
    """
    clean_archive = CLEAN_ROOT / f"{dataset}_processed.tar.zst"
    ref_archive   = find_archive(dataset, PERTURBATIONS[1][1])  # Linf8

    if not clean_archive.exists() or ref_archive is None:
        return None, {}

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
    for label, subpath in PERTURBATIONS[1:]:  # skip clean entry
        archive = find_archive(dataset, subpath)
        if archive is None:
            perturb_imgs[label] = None
            continue
        fnames, imgs = load_archive_full(archive)
        perturb_imgs[label] = get_img(fnames, imgs)

    return clean_img, perturb_imgs


def pil_to_b64(img: Image.Image, size=(224, 224)) -> str:
    img = img.resize(size, Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def image_diff_stats(clean: Image.Image, perturbed: Image.Image) -> dict:
    """Compute pixel-level diff stats between clean and perturbed image."""
    a = np.array(clean.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32)
    b = np.array(perturbed.convert("RGB").resize((224, 224), Image.BICUBIC)).astype(np.float32)
    diff      = np.abs(a - b)
    identical = bool(np.array_equal(a, b))
    max_diff  = float(diff.max())
    mean_diff = float(diff.mean())
    pct_changed = float((diff.sum(axis=-1) > 0).mean() * 100)
    return {
        "identical":    identical,
        "max_diff":     max_diff,
        "mean_diff":    mean_diff,
        "pct_changed":  pct_changed,
    }


# ---------------------------------------------------------------------------
# Build grid
# ---------------------------------------------------------------------------

print("Loading images (in-memory extraction, no disk writes)...\n")
grid       = {}
diff_stats = {}

for dataset in DATASETS:
    print(f"  {dataset}")
    grid[dataset]       = {}
    diff_stats[dataset] = {}

    clean_img, perturb_imgs = load_images_for_dataset(dataset)

    # Store clean
    grid[dataset]["clean"]       = pil_to_b64(clean_img) if clean_img else None
    diff_stats[dataset]["clean"] = None

    # Store each perturbation
    for label, _ in PERTURBATIONS[1:]:
        img = perturb_imgs.get(label)
        grid[dataset][label] = pil_to_b64(img) if img else None

        if img is not None and clean_img is not None:
            diff_stats[dataset][label] = image_diff_stats(clean_img, img)
        else:
            diff_stats[dataset][label] = None

        s      = diff_stats[dataset][label]
        status = "✓" if img else "N/A"
        if s:
            flag    = "⚠ IDENTICAL" if s["identical"] else f"Δmax={s['max_diff']:.1f}"
            status += f"  [{flag}]"
        print(f"    {label:<14} {status}")

# ---------------------------------------------------------------------------
# Render HTML
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
        return '<div class="badge identical">⚠ identical to clean</div>'
    color = "#b8f050" if stats["max_diff"] > 5 else "#f0a830"
    return (
        f'<div class="badge ok" style="color:{color}">'
        f'Δmax={stats["max_diff"]:.1f} &nbsp;|&nbsp; '
        f'mean={stats["mean_diff"]:.2f} &nbsp;|&nbsp; '
        f'{stats["pct_changed"]:.1f}% px changed'
        f'</div>'
    )

def render_cell(b64, stats, label):
    badge = diff_badge(stats, label)
    if b64 is None:
        return f'<td><div class="missing">N/A</div>{badge}</td>'
    return f'<td><img src="data:image/jpeg;base64,{b64}" loading="lazy">{badge}</td>'

rows_html = ""
for dataset in DATASETS:
    cells = "".join(
        render_cell(
            grid[dataset].get(lbl),
            diff_stats[dataset].get(lbl),
            lbl,
        )
        for lbl in perturb_labels
    )
    ds_label = DATASET_DISPLAY.get(dataset, dataset)
    rows_html += f'<tr><td class="ds">{ds_label}</td>{cells}</tr>\n'

header_cells = "".join(f"<th>{lbl}</th>" for lbl in perturb_labels)

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
  .sub{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#444;padding-left:17px;margin-bottom:28px}}
  .wrap{{overflow-x:auto}}
  table{{border-collapse:collapse;width:max-content}}
  thead tr{{background:#111;border-bottom:2px solid #1a1a1a}}
  th{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555;text-transform:uppercase;
      letter-spacing:.07em;padding:10px 10px;text-align:center;white-space:nowrap;min-width:242px}}
  th:first-child{{text-align:left;min-width:128px;color:#444}}
  th:nth-child(2){{color:#b8f050}}
  tbody tr{{border-bottom:1px solid #141414}}
  tbody tr:hover{{background:#111}}
  td.ds{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;color:#999;
         padding:10px 14px;white-space:nowrap;border-right:1px solid #1a1a1a;vertical-align:middle}}
  td{{padding:8px 8px;text-align:center;vertical-align:middle;border-right:1px solid #141414}}
  td:last-child{{border-right:none}}
  td img{{display:block;width:224px;height:224px;object-fit:cover;border-radius:3px;
          border:1px solid #1e1e1e;transition:transform .2s,border-color .2s;cursor:zoom-in}}
  td img:hover{{transform:scale(1.06);border-color:#b8f050;z-index:10;position:relative}}
  .missing{{width:224px;height:224px;display:flex;align-items:center;justify-content:center;
            font-family:'IBM Plex Mono',monospace;font-size:10px;color:#2a2a2a;
            background:#0a0a0a;border-radius:3px;border:1px dashed #1e1e1e;margin:auto}}
  .badge{{font-family:'IBM Plex Mono',monospace;font-size:9px;margin-top:5px;
          padding:3px 6px;border-radius:2px;text-align:center;letter-spacing:.03em}}
  .badge.identical{{background:#2a0a0a;color:#f05050;border:1px solid #3a1010}}
  .badge.ok{{background:#0e1a08;border:1px solid #1e2e10}}
</style>
</head>
<body>
<h1>Perturbation Visualizer</h1>
<p class="sub">CLIP ViT-B/16 LAION-2B surrogate · first test image per dataset · RobustGenBench</p>
<div class="wrap">
<table>
  <thead><tr><th>Dataset</th>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
</body>
</html>"""

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"\n✓ Saved → {OUTPUT_HTML}")
print("  Open in your browser to view the grid.")