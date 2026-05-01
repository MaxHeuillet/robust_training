"""
Generate the MAC (Mean Absolute Correlation) table with naive and stratified estimates.

This script computes:
- Naive MAC: computed over all configurations in a subset J
- Stratified MAC: corrects for confounding by stratifying on the strongest confounder
  (model_size for all categories except size itself, which is stratified by model_type)

Rank flips between naive and stratified estimates are highlighted in the LaTeX output:
  - Green: rank agreement (finding is robust to confounding)
  - Red: rank flip (finding is sensitive to subset composition)

Requirements:
- Input: fft_50_full.csv with columns:
    backbone_name, model_size, model_type, loss_function,
    pre_training_strategy, dataset, clean_acc, Linf_acc, L2_acc, L1_acc, common_acc
- Output: mac_table.tex (LaTeX table)

Usage:
    python generate_mac_table.py --input fft_50_full.csv --output mac_table.tex
"""

import argparse
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core metric
# ─────────────────────────────────────────────────────────────────────────────
ACC_COLS = ['clean_acc', 'Linf_acc', 'L2_acc', 'L1_acc', 'common_acc']


def mean_absolute_correlation(sub: pd.DataFrame, min_rows: int = 3) -> float:
    """
    MAC for a subset: mean absolute off-diagonal Spearman correlation
    across perturbation types.
    """
    if len(sub) < min_rows:
        return np.nan
    corr = sub[ACC_COLS].corr(method='spearman')
    mask = ~np.eye(len(ACC_COLS), dtype=bool)
    return np.abs(corr.values[mask]).mean()


# ─────────────────────────────────────────────────────────────────────────────
# MAC computation (naive and stratified)
# ─────────────────────────────────────────────────────────────────────────────
def compute_naive_mac(gdf: pd.DataFrame, datasets: list) -> dict:
    """Compute MAC per dataset and average, without any stratification."""
    macs = {}
    for ds in datasets:
        sub = gdf[gdf['dataset'] == ds]
        macs[ds] = mean_absolute_correlation(sub)
    macs['avg'] = np.nanmean([macs[ds] for ds in datasets])
    return macs


def compute_stratified_mac(
    gdf: pd.DataFrame,
    stratify_col: str,
    datasets: list,
    min_rows: int = 4
) -> tuple[dict, set]:
    """
    Compute MAC per dataset, stratified by `stratify_col`.

    For each (dataset, stratum) pair, compute MAC independently.
    Then average across strata (equal weight) to get the per-dataset MAC.
    Finally average across datasets.

    Returns:
        macs: dict with per-dataset MACs and 'avg'
        strata_dropped: set of stratum values excluded due to insufficient rows
    """
    macs = {}
    strata_dropped = set()

    for ds in datasets:
        ds_df = gdf[gdf['dataset'] == ds]
        stratum_macs = []
        for stratum, sdf in ds_df.groupby(stratify_col):
            if len(sdf) >= min_rows:
                m = mean_absolute_correlation(sdf)
                if not np.isnan(m):
                    stratum_macs.append(m)
            else:
                strata_dropped.add(stratum)
        macs[ds] = np.mean(stratum_macs) if stratum_macs else np.nan

    macs['avg'] = np.nanmean([macs[ds] for ds in datasets])
    return macs, strata_dropped


# ─────────────────────────────────────────────────────────────────────────────
# Rank flip detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_rank_flips(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    datasets: list,
    min_rows: int = 4
) -> set:
    """Return set of group values whose rank changes between naive and stratified MAC."""
    naive_vals = {}
    strat_vals = {}

    for val, gdf in df.groupby(group_col):
        naive_vals[val] = compute_naive_mac(gdf, datasets)['avg']
        s, _ = compute_stratified_mac(gdf, stratify_col, datasets, min_rows)
        strat_vals[val] = s['avg']

    naive_rank = pd.Series(naive_vals).rank(ascending=False)
    strat_rank = pd.Series(strat_vals).rank(ascending=False)

    return set(naive_rank[naive_rank != strat_rank].index)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX generation
# ─────────────────────────────────────────────────────────────────────────────
SIZE_DISPLAY = {'large': 'Base', 'medium': 'Small', 'small': 'Tiny'}

# Each entry: (display_name, group_col, stratify_col)
CATEGORIES = [
    ('Loss',           'loss_function',          'model_size'),
    ('Size',           'model_size',             'model_type'),
    ('Type',           'model_type',             'model_size'),
    ('Pre-training',   'pre_training_strategy',  'model_size'),
]


def fmt(v: float) -> str:
    if v is None or np.isnan(v):
        return '---'
    return f"{v:.3f}"


def generate_latex(df: pd.DataFrame, min_rows: int = 4) -> str:
    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)

    # Pre-compute rank flips for each category
    flip_sets = {}
    for cat_name, group_col, stratify_col in CATEGORIES:
        flip_sets[cat_name] = detect_rank_flips(
            df, group_col, stratify_col, datasets, min_rows
        )

    lines = []
    lines.append(r"\begin{wraptable}{r}{0.5\textwidth}")
    lines.append(r"\vspace{-1em}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Mean Absolute Correlation (MAC) in FFT-50. "
        r"$\widehat{\texttt{MAC}}$ is computed over all configurations in subset $J$. "
        r"$\widehat{\texttt{MAC}}_s$ corrects for confounding via stratification by "
        r"model size (or by architecture type for the size category). "
        r"\colorbox{green!15}{Green} indicates rank agreement; "
        r"\colorbox{red!15}{red} indicates a rank flip between the two estimates. "
        r"$^\dagger$Some strata excluded due to insufficient observations "
        rf"($<{min_rows}$ per dataset)."
        r"}"
    )
    lines.append(r"\label{tab:spearman_corr}")
    lines.append(r"\vspace{0.3em}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}ll r cc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& & $n$ & $\widehat{\texttt{MAC}}$ & $\widehat{\texttt{MAC}}_s$ \\"
    )
    lines.append(r"\midrule")

    # Global row
    n_global = len(df) // n_datasets
    global_mac = compute_naive_mac(df, datasets)['avg']
    lines.append(
        rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{All {n_global} configs}}}} "
        rf"& {n_global} & {fmt(global_mac)} & --- \\"
    )
    lines.append(r"\midrule")

    # Category rows
    for cat_name, group_col, stratify_col in CATEGORIES:
        sorted_vals = sorted(df[group_col].unique())
        flipped = flip_sets[cat_name]

        for i, val in enumerate(sorted_vals):
            gdf = df[df[group_col] == val]
            n = len(gdf) // n_datasets
            naive_avg = compute_naive_mac(gdf, datasets)['avg']
            strat, dropped = compute_stratified_mac(
                gdf, stratify_col, datasets, min_rows
            )
            strat_avg = strat['avg']

            # Display name
            display_val = SIZE_DISPLAY.get(val, val).replace('_', r'\_')

            # Footnote for excluded strata
            footnote = '$^{\\dagger}$' if dropped else ''

            # Color: red if rank flipped, green otherwise
            color = (r"\cellcolor{red!15}" if val in flipped
                     else r"\cellcolor{green!15}")

            # Category label (rotated, multirow)
            if i == 0:
                cat_cell = (
                    rf"\multirow{{{len(sorted_vals)}}}{{*}}"
                    rf"{{\rotatebox[origin=c]{{90}}{{\small {cat_name}}}}}"
                )
            else:
                cat_cell = ""

            lines.append(
                rf"  {cat_cell} & {display_val} & {n} "
                rf"& {color}{fmt(naive_avg)} "
                rf"& {color}{fmt(strat_avg)}{footnote} \\"
            )

        lines.append(r"\midrule")

    # Replace last \midrule with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{-1em}")
    lines.append(r"\end{wraptable}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate MAC table with naive and stratified estimates."
    )
    parser.add_argument(
        "--input", type=str, default="./results_analysis_neurips2026/fft_50_full.csv",
        help="Path to input CSV (default: fft_50_full.csv)"
    )
    parser.add_argument(
        "--output", type=str, default="./results_analysis_neurips2026/mac_table.tex",
        help="Path to output .tex file (default: mac_table.tex)"
    )
    parser.add_argument(
        "--min-rows", type=int, default=4,
        help="Minimum rows per stratum to include (default: 4)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Validate expected columns
    required = ACC_COLS + [
        'backbone_name', 'model_size', 'model_type',
        'loss_function', 'pre_training_strategy', 'dataset'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    latex = generate_latex(df, min_rows=args.min_rows)

    with open(args.output, 'w') as f:
        f.write(latex)

    print(f"Table written to {args.output}")

    # Print summary
    datasets = sorted(df['dataset'].unique())
    print(f"\nDatasets ({len(datasets)}): {datasets}")
    print(f"Total configs per dataset: {len(df) // len(datasets)}")
    print(f"Min rows per stratum: {args.min_rows}")


if __name__ == "__main__":
    main()