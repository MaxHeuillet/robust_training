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

### ✅ Specify loading paths and login to comet-ml

Specify the path to compressed archives, line 10 in `./dataset_to_tmpdir.sh`
Specify the paths to state_dicts and data folders, in the default configuration in `./configs`
Modify comet ML loging details in `./distributed_experiment_final.py` the method `initialize_logger()`.

### ✅ Cluster specific considerations

If you run jobs on Beluga, you need to enable internet connection for comet-ml with httpsproxy module in the job*_*.sh chain. If you run jobs on Cedar, you need to remove the loading of this module as it causes errors in environment packages loading.

### ✅ Run unit tests

```
salloc --account=def-adurand --time=2:59:00 --cpus-per-task=16 --mem=60000M --gpus-per-node=1

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a

source ~/scratch/myenv_reprod/bin/activate
cd ./my_project_directory
python ./unit_tests/architecture_loader_test.py
```

This test verifies i) that the backbone loads correctly, ii) that the output of the forward pass of the backbone is aligned with the nb of classes in the fine-tuning task, iii) that both CLASSIC_AT and TRADES loss output a float number, iv) that the learning rate and weight decay for the feature extractor and classification head are correctly split, v) that the gradient tracking for linear probing and end-to-end fine-tuning are correctly distinguished.

```
python ./unit_tests/data_transform_test.py
```

This test verifies that i) each dataset included in the study loads correctly, and ii) that the distinct transforms for train, and text-val are correctly associated with each dataset.

### 🚀 Launch All Jobs on the Cluster

Don't forget to change email and allocation credentials in your ```./execute_experiment*.sh``` script.

## Maxime :
```
bash ./execute_experiment_maxime.sh 'full_fine_tuning_5epochs_paper_final2'
```

## Rishika :
```
bash ./execute_experiment_rishika.sh 'full_fine_tuning_50epochs_paper_final2'
```

## Jonas :
```
bash ./execute_experiment_jonas.sh 'linearprobe_50epochs_paper_final2'
```

## Yann :
```
bash ./execute_experiment_yann.sh 'full_fine_tuning_50epochs_edge_paper_final2'
```
If we have time:
```
bash ./execute_experiment_yann.sh 'linearprobe_50epochs__edge_paper_final2'
```


---

### 💻 Run in an Interactive Session


```
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a

source ~/scratch/myenv_reprod/bin/activate
cd ./project_directory

bash ./dataset_to_tmpdir.sh 'uc-merced-land-use-dataset'
python distributed_experiment_final.py
```

> 💡 **Note:** The code runs a default configuration specified in `./utils/arguments.py`.
