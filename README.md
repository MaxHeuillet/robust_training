## 📚 Using the RobustGenBench Benchmark

### Create the Python environment

```bash
python3.11 -m venv ~/myenv_reprod
source ~/myenv_reprod/bin/activate
cd ./robust_training
pip install -r ./requirements.txt
```

> **Note (Slurm clusters):** the environment can be built automatically by
> running `source ./execute_setup.sh`, which handles module loading and dependency
> installation for Slurm-managed HPC environments.

### 📦 Download the RobustGenBench data

The benchmark is hosted on HuggingFace at
[`legolasflagstaff/RobustGenBench`](https://huggingface.co/datasets/legolasflagstaff/RobustGenBench).

To download only the clean archives (training, validation, and test splits)
without the pre-crafted adversarial perturbations — which are large and not
required for white-box evaluation — use the `ignore_patterns` filter:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="legolasflagstaff/RobustGenBench",
    repo_type="dataset",
    local_dir="~/data/",
    ignore_patterns="adversarial/*",
)
```

If you also want to prepare the data from scratch, the relevant scripts are:

```bash
python ./databases/download_data.py --save_path ~/data
python ./databases/save_final_dataset.py --datasets_path ~/data
```

### 🧪 Evaluate a model (white-box) on RobustGenBench

We provide a self-contained quickstart script for white-box adversarial
evaluation of any PyTorch model on RobustGenBench:

```
evaluate_quickstart.py
```

The script assumes **white-box access** to the model — adversarial examples
are crafted on-the-fly using [AutoAttack](https://github.com/fra31/auto-attack)
directly against the user's model, rather than relying on pre-crafted
perturbations. It covers:

- **Clean accuracy** — standard forward pass on the original test images
- **Adversarial accuracy** — under L-inf (ε=4/255), L2 (ε=2.0), and L1
  (ε=75.0) threat models, using AutoAttack's standard version

The only interface requirement is:

```python
model(images: Tensor[B, 3, H, W]) -> logits: Tensor[B, N]
```

Any `timm`, `torchvision`, or custom PyTorch model satisfies this. A toy CNN
is included so the script runs out of the box with no additional downloads.

**Basic usage:**

```bash
# Evaluate on all benchmark datasets
python evaluate_quickstart.py
```
Results are written as one CSV per dataset under `./quickstart_results/`.

## 📚 Accessing the open-source artifacts from the paper

In `./robust_training/configs/HPO_results` you will find the hyper-parameters optimized for all the fine-tuned configurations of the study.

In `./robust_training/results/results_dataset.csv` you will find the full measurement table for the robust fine-tuning experiments.

In `./robust_training/results/llm_classification_results` you will find the Ve-LLMs measurement results.

All the Figures of the paper can be reproducing with scripts in ```./results_analysis```

## 📚 Reproducing Paper Results 

This project provides a pipeline for reproducing the training and evaluation of various models under different pre-training and fine-tuning strategies, including adversarial robustness and transfer learning.

## Reproduce training

### 🗂️ Understanding the file system

The argument mode in ```utils/arguments.py``` specifies which step of the code to execute. At the end of ```mode='hpo'```, the code stores the results of HPO optimization in a separate folder of ```configs```. At the beginning of ```mode='train'```, the config is loaded to train the model with optimized HPO. Then the model is saved. At testing, the model is loaded and the results are saved in a folder named after project name in ```results``` folder.

### 🧠 Download the models

```python ./architectures/download_architectures.py --save_path ~/my_backbones```

To download the robust checkpoints:
		- Download link ('robust_convnext_tiny', 'robust_deit_small_patch16_224', 'robust_convnext_base', 'robust_vit_base_patch16_224’):  [Download link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/XLLnoCnJxp74Zqn) . This is from o "Revisiting Adversarial Training for Imagenet" (Neurips 2023) paper.
		- Download link of resnet50: [Download link](https://www.dropbox.com/scl/fi/7f2p987eg4pwugw2r660b/imagenet_linf_4.pt?rlkey=e5nv0f5lrktppjlv2c9dcccz9&e=2&dl=0); this is from [Madry-robustness repo](https://github.com/MadryLab/robustness?tab=readme-ov-file) .

```python ./architectures/process_robust_architectures.py --path ~/my_backbones```

### ✅ Launch code

> 💡 **Note:** The code runs a default configuration specified in `./utils/arguments.py`.
> 
> 💡 **Note:** We have added ```break``` statements in the train and test loops (L.517, L.491, L.273) to simplify execution of toy code.

Locally:

```python distributed_experiment_final.py```

On a SLURM cluster:
For FFT-5: ```bash ./execute_experiment.sh 'full_fine_tuning_5epochs_reproduce'```
For FFT-50: ```bash ./execute_experiment.sh 'full_fine_tuning_50epochs_reproduce'```
For LP-50: ```bash ./execute_experiment.sh 'linearprobe_50epochs_reproduce'```

### 🧪 Run unit tests

```
python ./unit_tests/architecture_loader_test.py
```

