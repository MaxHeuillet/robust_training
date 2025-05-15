## ✅ Create python environment

```
python3.11 -m venv ~/myenv_reprod
source ~/myenv_reprod/bin/activate
cd ./robust_training
pip install -r ./requirements.txt
``` 

## ✅ Reproduce paper results

The database with all the measurements is ```results_dataset.csv```

All the Figures of the paper can be reproducing with scripts in ```./results_analysis```

## ✅ Reproduce training

### ✅ Download and process datasets

```python ./databases/download_data.py --save_path ~/data```
For Caltech101, LandUse and StanfordCard, use Kaggle. 
Relevant info: https://stackoverflow.com/questions/63067515/http-error-404-not-found-in-downloading-caltech101-dataset

```python ./databases/save_final_dataset.py --datasets_path ~/data```

### ✅ Download and process backbones

```python ./architectures/download_architectures.py --save_path ~/my_backbones```

To download the robust checkpoints:
		- Download link ('robust_convnext_tiny', 'robust_deit_small_patch16_224', 'robust_convnext_base', 'robust_vit_base_patch16_224’):  https://nc.mlcloud.uni-tuebingen.de/index.php/s/XLLnoCnJxp74Zqn . This is from official github (https://github.com/nmndeep/revisiting-at?tab=readme-ov-file) of "Revisiting Adversarial Training for Imagenet" (Neurips 2023) paper.
		- Download link of resnet50: https://www.dropbox.com/scl/fi/7f2p987eg4pwugw2r660b/imagenet_linf_4.pt?rlkey=e5nv0f5lrktppjlv2c9dcccz9&e=2&dl=0); this is from https://github.com/MadryLab/robustness?tab=readme-ov-file .

```python ./architectures/process_robust_architectures.py --path ~/my_backbones```

### ✅ Launch code

> 💡 **Note:** The code runs a default configuration specified in `./utils/arguments.py`.
> 💡 **Note:** We have added ```break``` statements in the train and test loops (L.517, L.491, L.273) to simplify execution of toy code.

Locally:

```python distributed_experiment_final.py```

On a SLURM cluster:

```
bash ./execute_experiment.sh 'full_fine_tuning_5epochs_reproduce'
```

```
bash ./execute_experiment.sh 'full_fine_tuning_50epochs_reproduce'
```

```
bash ./execute_experiment.sh 'linearprobe_50epochs_reproduce'
```
### ✅ Understanding the file system

At the end of ```mode='hpo```, the code stores the results of HPO optimization in a separate folder of ```configs```. At the beginning of ```mode='train'```, the config is loaded to train the model with optimized HPO. Then the model is saved. At testing, the model is loaded and the results are saved in a folder named after project name in ```results``` folder.


### ✅ Run unit tests

```
python ./unit_tests/architecture_loader_test.py
python ./unit_tests/dataset_transform_test.py
```

This test verifies i) that the backbone loads correctly, ii) that the output of the forward pass of the backbone is aligned with the nb of classes in the fine-tuning task, iii) that both CLASSIC_AT and TRADES loss output a float number, iv) that the learning rate and weight decay for the feature extractor and classification head are correctly split, v) that the gradient tracking for linear probing and end-to-end fine-tuning are correctly distinguished.
