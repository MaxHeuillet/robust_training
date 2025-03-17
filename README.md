## 🛠️ Setup Instructions

### ✅ Create the Python Environment

```
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a

python3.11 -m venv ~/scratch/myenv_reprod
source ~/scratch/myenv_reprod/bin/activate
cd ./my_project_directory
pip install -r requirements.txt
```

> 💡 **Note:** The code runs with **Python 3.11**.

---

### 🚀 Launch All Jobs on the Cluster

*(Add your job launching instructions here if applicable)*

---

### 💻 Run in an Interactive Session

```
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a httpproxy

source ~/scratch/myenv_reprod/bin/activate
cd ./project_directory

bash ./dataset_to_tmpdir.sh 'uc-merced-land-use-dataset'
python distributed_experiment_final.py
```

> 💡 **Note:** The code runs a default configuration specified in `./utils/arguments.py`.
