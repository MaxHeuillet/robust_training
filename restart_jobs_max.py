import pandas as pd
import subprocess

df = pd.read_csv("./results_analysis/to_relaunch_max.csv")
# df = pd.read_csv("./results_analysis/to_relaunch_rishika.csv")
# df = pd.read_csv("./results_analysis/to_relaunch_jonas.csv")
email = "maxime.heuillet.1@ulaval.ca"


project_name = "full_fine_tuning_5epochs_article1"
# project_name = "linearprobe_50epochs_paper_final2"
# project_name = 'full_fine_tuning_50epochs_paper_final2'

ACCOUNT = "rrg-adurand"

for _, row in df.iterrows():
    backbone = row["backbone"]
    dataset = row["dataset"]
    loss = row["loss_function"]
    job_script = "job1_hpo.sh"#row["restart_from"] 
    cmd = [
    "sbatch",
    f"--account={ACCOUNT}",
    f"--mail-user={email}",
    f"--export=ALL,ACCOUNT={ACCOUNT},BCKBN={backbone},DATA={dataset},SEED=1,LOSS={loss},PRNM={project_name},EMAIL={email}",
    job_script
    ]   
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)