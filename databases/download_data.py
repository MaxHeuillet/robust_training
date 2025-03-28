from torchvision import datasets
import os
import kaggle



save_path = os.path.expanduser('~/scratch/data')

########## UC MERCED LAND USE DATASET #########
KAGGLE_DATASET = "abdulhasibuddin/uc-merced-land-use-dataset"
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=save_path, unzip=False)

########## STANFORD CARS #########
KAGGLE_DATASET = 'rickyyyyyyy/torchvision-stanford-cars'
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=save_path, unzip=False)
# this is because pytorch download does not work

########## OxfordIIITPet #########
datasets.OxfordIIITPet(root=save_path,  download=True, )

########## Caltech101 #########
datasets.Caltech101(root=save_path,  download=True, )

########## DTD #########
datasets.DTD(root=save_path,  download=True, )

########## FGVCAircraft #########
datasets.FGVCAircraft(root=save_path, download=True, )

########## Flowers102 #########
train_dataset = datasets.Flowers102(root=save_path, download=True,)



