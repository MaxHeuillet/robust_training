#!/usr/bin/env python3
"""
latex_random_vs_transfer.py — Build a LaTeX table comparing accuracy under
random Linf 30/255 noise vs. transfer attacks at the same budget.

Reads batch manifests + predictions.jsonl from ./llm_classification_results/
(same layout as summarize_results.py).

Output: LaTeX source on stdout, plus an optional --out file.

Layout: one sub-table per model_key (gemini, claude, ...), rows = datasets,
columns = {Random, CLIP-B, CLIP-H, MetaCLIP, SigLIP}.
Each cell shows accuracy; lowest accuracy per row is bolded (= most effective attack).
Optionally shows Δ vs. Random baseline in parentheses.
"""

import argparse
import json
from pathlib import Path

BASE = Path("llm_classification_results")

# Column order: Random first (baseline), then surrogates by increasing "strength"
# roughly ordered by model size/capability
ATTACKS = [
    ("random_linf30",                 r"Random"),
    ("adv_linf30",                    r"CLIP-B/16"),
    ("adv_clip_vith14_linf30",        r"CLIP-H/14"),
    ("adv_metaclip_linf_eps30",       r"MetaCLIP-H"),
    ("adv_siglip2_384_linf_eps30",    r"SigLIP-SO400M"),
]

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

DATASET_DISPLAY = {
    "caltech101":                 r"Caltech-101",
    "fgvc-aircraft-2013b":        r"FGVC-Aircraft",
    "flowers-102":                r"Flowers-102",
    "oxford-iiit-pet":            r"Oxford-Pet",
    "stanford_cars":              r"Stanford Cars",
    "uc-merced-land-use-dataset": r"UC Merced",
}

MODEL_DISPLAY = {
    "google_nothink": r"Gemini~3 Flash (no-think)",
    "google_think":   r"Gemini~3 Flash (think)",
    "anthropic":      r"Claude Haiku 4.5",
    "openai":         r"GPT-4o-mini",
}


# ---------------------------------------------------------------------------
# Helpers — mirror summarize_results.py
# ---------------------------------------------------------------------------

def dataset_key(run_name: str) -> str:
    for ds in sorted(DATASETS, key=len, reverse=True):
        if run_name.startswith(ds):
            return ds
    return run_name.split("__")[0]


def load_predictions(run_name: str):
    pred_path = BASE / run_name / "predictions.jsonl"
    if not pred_path.exists():
        return None
    correct = total = 0
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("error"):
                continue
            total += 1
            if rec.get("correct", False):
                correct += 1
        except Exception:
            pass
    return (correct, total) if total > 0 else None


def merge_complement(main_run: str, comp_run: str):
    recs = []
    for run in [main_run, comp_run]:
        p = BASE / run / "predictions.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    if not recs:
        return None
    seen = {}
    for r in recs:
        if not r.get("error") and r["index"] not in seen:
            seen[r["index"]] = r
    total   = len(seen)
    correct = sum(1 for r in seen.values() if r.get("correct", False))
    return (correct, total) if total > 0 else None


def collect_results() -> dict:
    """Returns {experiment: {model_key: {dataset: acc}}}."""
    results: dict = {}
    for attack_exp, _ in ATTACKS:
        mp = BASE / f"batch_manifest__all_datasets__{attack_exp}.json"
        if not mp.exists():
            print(f"  ⚠ Missing manifest: {mp.name}")
            continue
        manifest = json.loads(mp.read_text())

        groups: dict = {}
        for entry in manifest:
            ds  = dataset_key(entry["run_name"])
            key = entry["key"]
            groups.setdefault((ds, key), []).append(entry)

        exp_results = {}
        for (ds, key), entries in groups.items():
            main_entries = [e for e in entries if "__complement" not in e["run_name"]]
            comp_entries = [e for e in entries if "__complement" in e["run_name"]]
            main_ok      = [e for e in main_entries if e["status"] != "failed"]
            main = main_ok[0] if main_ok else (main_entries[0] if main_entries else entries[0])
            comp = comp_entries[0] if comp_entries else None
            if main["status"] != "retrieved":
                continue

            if comp and comp["status"] == "retrieved":
                result = merge_complement(main["run_name"], comp["run_name"])
            else:
                result = load_predictions(main["run_name"])

            if result:
                correct, total = result
                if total > 0:
                    exp_results.setdefault(key, {})[ds] = correct / total

        results[attack_exp] = exp_results
    return results


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------

def fmt_cell(acc, is_best, show_delta, delta):
    if acc is None:
        return "--"
    s = f"{acc * 100:.1f}"
    if show_delta and delta is not None:
        sign = "+" if delta >= 0 else ""
        s = f"{s}{{\\scriptsize\\,({sign}{delta*100:.1f})}}"
    if is_best:
        s = r"\textbf{" + s + "}"
    return s


def build_table(results: dict, models_to_show: list, show_delta: bool,
                highlight: str, fit: str = "resize") -> str:
    """Single unified tabular: one row per (model, dataset), with model name
    in a leftmost column via \\multirow. Renders predictably in all LaTeX
    engines, unlike stacked sub-tables which drift unpredictably.

    highlight in {'worst', 'best', 'none'}
    fit in {'none', 'resize', 'tight', 'both'}
        'resize' — wrap tabular in \\resizebox{\\textwidth}{!}{...}
        'tight'  — \\footnotesize + reduced \\tabcolsep
        'both'   — both of the above (most aggressive)
    """
    n_cols = len(ATTACKS)
    col_spec = "ll" + "c" * n_cols  # model | dataset | attack cols

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")

    # Font size / spacing tweaks (applied inside the table environment so
    # they only affect this table).
    if fit in ("tight", "both"):
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
    else:
        lines.append(r"\small")

    delta_note = r" $\Delta$ vs.\ Random in parentheses." if show_delta else ""
    hl_note = {
        "worst": r" \textbf{Bold} indicates the most effective attack per row (lowest accuracy).",
        "best":  r" \textbf{Bold} indicates the most robust model per row (highest accuracy).",
        "none":  "",
    }[highlight]
    lines.append(
        r"\caption{Classification accuracy (\%) under $L_\infty$ perturbations "
        r"at $\varepsilon = 30/255$: random uniform noise versus transfer attacks "
        r"crafted on four surrogate models." + delta_note + hl_note + r"}"
    )
    lines.append(r"\label{tab:random-vs-transfer}")

    # Optionally wrap tabular in resizebox
    if fit in ("resize", "both"):
        lines.append(r"\resizebox{\textwidth}{!}{%")

    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = ["Model", "Dataset"] + [disp for _, disp in ATTACKS]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Filter to models that actually have data
    active_models = [
        m for m in models_to_show
        if any(m in results.get(exp, {}) and results[exp][m]
               for exp, _ in ATTACKS)
    ]

    for m_idx, model_key in enumerate(active_models):
        model_label = MODEL_DISPLAY.get(model_key, model_key)
        n_rows = len(DATASETS) + 1  # +1 for Mean

        for i, ds in enumerate(DATASETS):
            accs = [
                results.get(attack_exp, {}).get(model_key, {}).get(ds)
                for attack_exp, _ in ATTACKS
            ]
            valid = [a for a in accs if a is not None]
            if not valid or highlight == "none":
                target = None
            elif highlight == "worst":
                target = min(valid)
            else:
                target = max(valid)

            random_acc = accs[0]
            cells = []
            # First cell: model name via multirow, only on first row of block
            if i == 0:
                cells.append(r"\multirow{" + str(n_rows) + r"}{*}{\textit{" +
                             model_label + r"}}")
            else:
                cells.append("")
            cells.append(DATASET_DISPLAY[ds])
            for j, acc in enumerate(accs):
                is_target = (target is not None and acc == target)
                delta = (acc - random_acc) if (show_delta and acc is not None
                                                and random_acc is not None
                                                and j > 0) else None
                cells.append(fmt_cell(acc, is_target, show_delta and j > 0, delta))
            lines.append(" & ".join(cells) + r" \\")

        # Mean row for this model
        mean_accs = []
        for attack_exp, _ in ATTACKS:
            col_accs = [
                results.get(attack_exp, {}).get(model_key, {}).get(ds)
                for ds in DATASETS
            ]
            col_accs = [a for a in col_accs if a is not None]
            mean_accs.append(sum(col_accs) / len(col_accs) if col_accs else None)

        valid = [a for a in mean_accs if a is not None]
        if valid and highlight != "none":
            target = min(valid) if highlight == "worst" else max(valid)
        else:
            target = None

        random_mean = mean_accs[0]
        cells = ["", r"\textit{Mean}"]
        for j, acc in enumerate(mean_accs):
            is_target = (target is not None and acc == target)
            delta = (acc - random_mean) if (show_delta and acc is not None
                                             and random_mean is not None
                                             and j > 0) else None
            cells.append(fmt_cell(acc, is_target, show_delta and j > 0, delta))
        # Use cmidrule before mean to separate it from per-dataset rows
        lines.append(r"\cmidrule(lr){2-" + str(2 + n_cols) + "}")
        lines.append(" & ".join(cells) + r" \\")

        # Separator between models (not after last)
        if m_idx < len(active_models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if fit in ("resize", "both"):
        lines.append(r"}")  # close resizebox

    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models",    nargs="+", default=None,
                   help="Which model_key(s) to include (default: all found)")
    p.add_argument("--delta",     action="store_true",
                   help="Show Δ vs. Random baseline in parentheses")
    p.add_argument("--highlight", default="worst",
                   choices=["worst", "best", "none"],
                   help="Bold target per row (default: worst = most effective attack)")
    p.add_argument("--fit",       default="resize",
                   choices=["none", "resize", "tight", "both"],
                   help="Page-width fitting: 'resize' wraps in \\resizebox "
                        "(default), 'tight' shrinks font + column padding, "
                        "'both' combines them, 'none' leaves the table as-is")
    p.add_argument("--out",       default=None,
                   help="Write LaTeX to this file (also prints to stdout)")
    args = p.parse_args()

    results = collect_results()

    if args.models:
        models_to_show = args.models
    else:
        models_to_show = sorted({
            k for exp_results in results.values() for k in exp_results.keys()
        })

    if not models_to_show:
        print("No models found in any manifest.")
        return

    latex = build_table(results, models_to_show, args.delta, args.highlight,
                        fit=args.fit)
    print(latex)

    if args.out:
        Path(args.out).write_text(latex + "\n")
        print(f"\n% Written to: {args.out}", end="")


if __name__ == "__main__":
    main()