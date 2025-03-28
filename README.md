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

Specify the path to compressed archives, line 10 in `./dataset_to_tmpdir.sh`
Specify the paths to state_dicts and data folders, in the default configuration in `./configs`
Modify comet ML loging details in `./distributed_experiment_final.py` the method `initialize_logger()`.


----


You can run unit_tests to validate loading of datasets and backbones.

```
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a

source ~/scratch/myenv_reprod/bin/activate
cd ./my_project_directory
python ./unit_tests/architecture_loader_test.py
python ./unit_tests/dataset_transform_test.py
```


### 🚀 Launch All Jobs on the Cluster

Don't forget to change email and allocation credentials in ```./job*.sh``` scripts.

Maxime :
```
bash ./execute_experiment.sh 'full_fine_tuning_50epochs_final4'
```

Rishika :

```
bash ./execute_experiment.sh 'full_fine_tuning_5epochs_final4'
```

Jonas :
```
bash ./execute_experiment.sh 'linearprobe_50epochs_final4'
```

Yann :
```
bash ./execute_experiment.sh 'full_fine_tuning_50epochs_edge_final4'
```


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
