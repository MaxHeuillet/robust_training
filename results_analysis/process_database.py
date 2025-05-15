import pandas as pd

# Re-run necessary steps after code environment reset
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import Table2x2


def process_grouped_df(final_data, size=None):

    df = pd.DataFrame(final_data)

    nan_percentage = (df.isna().sum().sum() / df.size) * 100
    print(f"Percentage of NaN values: {nan_percentage:.2f}%")

    if size:
        df = df[ df.model_size == size ]

    grouped_df = df.pivot_table(
        index=['backbone', "backbone_name", 'loss_function', 'pre_training_strategy', 'model_type', "model_size", "ft_strategy"],
        columns='dataset',
        # dropna=False
    )

    # Rename column levels
    grouped_df.columns.set_names(["metric", "dataset"], inplace=True)

    # Swap column levels → dataset becomes level 0, metric becomes level 1
    grouped_df.columns = grouped_df.columns.swaplevel(0, 1)

    # (Optional) Sort columns so that all metrics are grouped within each dataset
    grouped_df = grouped_df.sort_index(axis=1, level=0)

    return grouped_df


def process_rankings(grouped_df):

    # --- existing shortcuts ----------------------------------------------------
    sum_scores  = grouped_df.xs('sum',  level=1, axis=1)     # (rows × 6 datasets)
    geom_scores = grouped_df.xs('geom', level=1, axis=1)     # (rows × 6 datasets)

    #--- 1. add the TOTAL aggregates you already computed ----------------------
    grouped_df[('TOTAL', 'score_sum')]  = sum_scores.sum(axis=1)
    grouped_df[('TOTAL', 'score_geom')] = geom_scores.sum(axis=1)

    # --- 2. count NaNs across datasets -----------------------------------------
    grouped_df[('TOTAL', 'nan_sum_cnt')]  = sum_scores.isna().sum(axis=1)
    grouped_df[('TOTAL', 'nan_geom_cnt')] = geom_scores.isna().sum(axis=1)

    # --- 3. rank as before ------------------------------------------------------
    df_sorted = grouped_df.sort_values(('TOTAL', 'score_sum'), ascending=False)

    # Add ranks (1 = best/highest)
    df_sorted[('TOTAL', 'rank_sum')]  = df_sorted[('TOTAL', 'score_sum')].rank(ascending=False, method='min')
    df_sorted[('TOTAL', 'rank_geom')] = df_sorted[('TOTAL', 'score_geom')].rank(ascending=False, method='min')

    metrics = ['L1_acc', 'L2_acc', 'Linf_acc', 'clean_acc', 'common_acc']
    dataset_names = [ds for ds in grouped_df.columns.levels[0] if ds != 'TOTAL']

    # Compute per-dataset Borda scores
    for dataset in dataset_names:
        per_metric_ranks = []
        for metric in metrics:
            col = (dataset, metric)
            if col in grouped_df.columns:
                ranks = grouped_df[col].rank(ascending=True, method='min')
                per_metric_ranks.append(ranks)
        if per_metric_ranks:
            grouped_df[(dataset, 'borda')] = sum(per_metric_ranks)

    # Add TOTAL Borda score as the sum across datasets
    borda_scores = grouped_df.xs('borda', level=1, axis=1)
    grouped_df[('TOTAL', 'borda')] = borda_scores.sum(axis=1)

    # Re-rank after Borda is computed
    grouped_df[('TOTAL', 'rank_sum')]   = grouped_df[('TOTAL', 'score_sum')].rank(ascending=False, method='min')
    grouped_df[('TOTAL', 'rank_geom')]  = grouped_df[('TOTAL', 'score_geom')].rank(ascending=False, method='min')
    grouped_df[('TOTAL', 'rank_borda')] = grouped_df[('TOTAL', 'borda')].rank(ascending=False,  method='min')  # lower is better for Borda

    # Optional: sort the columns to keep things tidy
    grouped_df = grouped_df.sort_index(axis=1, level=[0, 1])

    grouped_df = grouped_df.sort_values(('TOTAL', 'borda'), ascending=False)

    return grouped_df

# Recompute odds ratios using 'in_tier1'
def compute_odds_ratio_by_group(df, group_col, target_col="in_tier1"):
    results = []

    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        non_group_df = df[df[group_col] != group]

        A = group_df[target_col].sum()
        B = len(group_df) - A
        C = non_group_df[target_col].sum()
        D = len(non_group_df) - C

        contingency = np.array([[A, B], [C, D]])

        if (A + B) > 0 and (C + D) > 0:
            table = Table2x2(contingency)
            odds_ratio = table.oddsratio
            p_value = table.oddsratio_pvalue()
        else:
            odds_ratio = np.nan
            p_value = np.nan

        results.append({
            group_col: group,
            "in_tier1": int(A),
            "not_in_tier1": int(B),
            "odds_ratio": odds_ratio,
            "p_value": p_value
        })

    return pd.DataFrame(results)


def global_grouped_dataset(df):

    grouped_df = df.pivot_table(
        index=['ft_strategy', 'model_type', 'model_size', 'pre_training_strategy', 'loss_function', 'backbone',],
        columns='dataset',
        # dropna=False
    )

    # Rename column levels
    grouped_df.columns.set_names(["metric", "dataset"], inplace=True)

    # Swap column levels → dataset becomes level 0, metric becomes level 1
    grouped_df.columns = grouped_df.columns.swaplevel(0, 1)

    # (Optional) Sort columns so that all metrics are grouped within each dataset
    grouped_df = grouped_df.sort_index(axis=1, level=0)

    # grouped_df.to_csv("./{}.csv".format(pn1))

    grouped_df.shape

    # --- existing shortcuts ----------------------------------------------------
    sum_scores  = grouped_df.xs('sum',  level=1, axis=1)     # (rows × 6 datasets)
    geom_scores = grouped_df.xs('geom', level=1, axis=1)     # (rows × 6 datasets)

    #--- 1. add the TOTAL aggregates you already computed ----------------------
    grouped_df[('TOTAL', 'score_sum')]  = sum_scores.sum(axis=1)
    grouped_df[('TOTAL', 'score_geom')] = geom_scores.sum(axis=1)

    # --- 2. count NaNs across datasets -----------------------------------------
    grouped_df[('TOTAL', 'nan_sum_cnt')]  = sum_scores.isna().sum(axis=1)
    grouped_df[('TOTAL', 'nan_geom_cnt')] = geom_scores.isna().sum(axis=1)

    return grouped_df