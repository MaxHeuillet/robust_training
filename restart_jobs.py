import pandas as pd
import subprocess

df = pd.read_csv("restart_jobs_max.csv")
email = "rishika.bhagwatkar@umontreal.ca"
# project_name = "full_fine_tuning_50epochs_paper_final2"  # <-- change this
project_name = "full_fine_tuning_5epochs_article1"
ACCOUNT = "rrg-bengioy-ad"

for _, row in df.iterrows():
    backbone = row["backbone"]
    dataset = row["dataset"]
    loss = row["loss_function"]
    job_script = row["restart_from"]
    cmd = [
    "sbatch",
    f"--account={ACCOUNT}",
    f"--mail-user={email}",
    f"--export=ALL,ACCOUNT={ACCOUNT},BCKBN={backbone},DATA={dataset},SEED=1,LOSS={loss},PRNM={project_name},EMAIL={email}",
    job_script
    ]   
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)