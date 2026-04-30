"""
reorganize_results.py
─────────────────────
Runs the same file-resolution logic as your original load_result_dataset,
then dumps the winning .pkl for each (backbone, dataset, loss) combination
into two clean, flat directory structures:

    results_clean/
        full_fine_tuning50/
        full_fine_tuning5/
        linear_probing50/

    configs/HPO_results_clean/
        full_fine_tuning50/
        full_fine_tuning5/
        linear_probing50/

Each file is named:  {backbone}__{dataset}__{loss}.pkl
(double-underscore to avoid ambiguity with underscores inside each field)

Non-destructive: original folders are never modified.

Usage
─────
    python reorganize_results.py --base /Users/you/Desktop/robust_training
    python reorganize_results.py --base /Users/you/Desktop/robust_training --dry-run
"""

import argparse
import shutil
from pathlib import Path


# ── experiment groups (shared by both trees) ──────────────────────────────────
# Each tuple: (pn1, pn2, pn3, output_folder_name)
# Resolution priority mirrors load_result_dataset: pn3 first, then pn2, then pn1.
# Use "none" to skip a slot.
GROUPS = [
    (
        "full_fine_tuning_50epochs_edge_paper_final2",
        "full_fine_tuning_50epochs_paper_final2",
        "none",
        "full_fine_tuning50",
    ),
    (
        "full_fine_tuning_5epochs_edge_article1",
        "full_fine_tuning_5epochs_article1",
        "none",
        "full_fine_tuning5",
    ),
    (
        "linearprobe_50epochs_edge_paper_final2",
        "linearprobe_50epochs_paper_final2",
        "none",
        "linear_probing50",
    ),
]

# ── the two source→destination trees to process ───────────────────────────────
# Each tuple: (src subfolder, dst subfolder, file extension)
TREES = [
    ("results",             "results_clean",             ".pkl"),
    ("configs/HPO_results", "configs/HPO_results_clean", ".yaml"),
]

LOSSES = ("TRADES_v2", "CLASSIC_AT")

DATASETS = (
    "uc-merced-land-use-dataset",
    "stanford_cars",
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
)

BACKBONES = (
    "CLIP-convnext_base_w-laion_aesthetic-s13B-b82K",
    "CLIP-convnext_base_w-laion2B-s13B-b82K",
    "deit_small_patch16_224.fb_in1k",
    "robust_resnet50",
    "vit_small_patch16_224.augreg_in21k",
    "convnext_base.fb_in1k",
    "resnet50.a1_in1k",
    "robust_vit_base_patch16_224",
    "vit_base_patch16_224.mae",
    "vit_small_patch16_224.dino",
    "convnext_base.fb_in22k",
    "robust_convnext_base",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_base_patch16_clip_224.laion2b",
    "convnext_tiny.fb_in1k",
    "robust_convnext_tiny",
    "robust_deit_small_patch16_224",
    "vit_small_patch16_224.augreg_in1k",
    "convnext_tiny.fb_in22k",
    "vit_base_patch16_clip_224.laion2b_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "eva02_base_patch14_224.mim_in22k",
    "eva02_tiny_patch14_224.mim_in22k",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "convnext_base.clip_laion2b_augreg_ft_in12k_in1k",
    "convnext_base.fb_in22k_ft_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
    "coatnet_0_rw_224.sw_in1k",
    "coatnet_2_rw_224.sw_in12k_ft_in1k",
    "coatnet_2_rw_224.sw_in12k",
    "regnetx_004.pycls_in1k",
    "efficientnet-b0",
    "deit_tiny_patch16_224.fb_in1k",
    "mobilevit-small",
    "mobilenetv3_large_100.ra_in1k",
    "edgenext_small.usi_in1k",
    "coat_tiny.in1k",
)


def resolve_src(src_root: Path, pn1: str, pn2: str, pn3: str, stem: str, ext: str) -> Path | None:
    """Return the first existing .pkl among pn3 → pn2 → pn1 (mirrors load_result_dataset)."""
    for pn in (pn3, pn2, pn1):
        if pn == "none":
            continue
        candidate = src_root / pn / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def process_tree(src_root: Path, dst_root: Path, ext: str, dry_run: bool) -> tuple[int, int]:
    found = missing = 0
    for pn1, pn2, pn3, out_folder in GROUPS:
        out_dir = dst_root / out_folder
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        for loss in LOSSES:
            for dataset in DATASETS:
                for backbone in BACKBONES:
                    stem = f"{backbone}_{dataset}_{loss}"
                    src_path = resolve_src(src_root, pn1, pn2, pn3, stem, ext)
                    dst_path = out_dir / f"{backbone}__{dataset}__{loss}{ext}"

                    if src_path is None:
                        print(f"  MISSING  [{out_folder}]  {stem}")
                        missing += 1
                        continue

                    rel = src_path.relative_to(src_root)
                    print(f"  OK  {rel}  →  {out_folder}/{dst_path.name}")
                    found += 1

                    if not dry_run:
                        shutil.copy2(src_path, dst_path)

    return found, missing


def main():
    parser = argparse.ArgumentParser(description="Reorganise results into clean flat folders.")
    parser.add_argument("--base",    required=True, help="Root of the project, e.g. /Users/you/Desktop/robust_training")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    args = parser.parse_args()

    base = Path(args.base)
    total_found = total_missing = 0

    for src_rel, dst_rel, ext in TREES:
        src_root = base / src_rel
        dst_root = base / dst_rel
        if not src_root.exists():
            print(f"\n  SKIPPING {src_rel} — folder not found")
            continue
        print(f"\n── {src_rel}  →  {dst_rel} ──")
        f, m = process_tree(src_root, dst_root, ext, args.dry_run)
        total_found += f
        total_missing += m

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}"
          f"{total_found} files copied, {total_missing} missing.")


if __name__ == "__main__":
    main()