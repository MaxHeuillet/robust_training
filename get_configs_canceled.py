import pandas as pd
import yaml

# df = pd.read_csv("result_with_missing_values.csv")
df = pd.read_csv("result_with_missing_values_max.csv")


def get_restart_job(row):
    is_missing = lambda col: pd.isna(row[col])

    missing = {col: is_missing(col) for col in ['clean_acc', 'Linf_acc', 'L1_acc', 'L2_acc', 'common_acc']}

    if all(missing.values()):
        return "job1_hpo.sh"
    elif not missing['clean_acc'] and not missing['Linf_acc'] and missing['L1_acc']:
        return "job4_test_l1.sh"
    elif not missing['clean_acc'] and not missing['Linf_acc'] and not missing['L1_acc'] and missing['L2_acc']:
        return "job5_test_l2.sh"
    elif not missing['clean_acc'] and not missing['Linf_acc'] and not missing['L1_acc'] and not missing['L2_acc'] and missing['common_acc']:
        return "job6_test_common.sh"
    else:
        return None  # means no job needs to be restarted

df['restart_from'] = df.apply(get_restart_job, axis=1)

to_restart = df[df['restart_from'].notna()][['backbone', 'dataset', 'loss_function', 'restart_from']]
to_restart.to_csv("restart_jobs_max.csv", index=False)