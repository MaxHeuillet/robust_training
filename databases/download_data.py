import argparse
import os
from torchvision import datasets


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download datasets to a specified path.")
parser.add_argument("--save_path", type=str, default="./data", help="Directory to save the datasets")
args = parser.parse_args()

# Expand and create save path
save_path = os.path.expanduser(args.save_path)
os.makedirs(save_path, exist_ok=True)

########## Flowers102 #########
datasets.Flowers102(root=save_path, download=True)

########## OxfordIIITPet #########
datasets.OxfordIIITPet(root=save_path, download=True)

########## FGVCAircraft #########
datasets.FGVCAircraft(root=save_path, download=True)


import kaggle
# ########## Caltech101 #########
# Download: https://data.caltech.edu/records/mzrjq-6wc02 
# mv /Downloads/caltech-101.zip ~/data/caltech101.zip
# unzip ~/data/caltech101.zip -d ~/data
# mv ~/data/caltech-101 ~/data/caltech101
# cd ~/data/caltech101
# tar -xzf 101_ObjectCategories.tar.gz
# tar -xf Annotations.tar

# ########## UC MERCED LAND USE DATASET #########
KAGGLE_DATASET = "abdulhasibuddin/uc-merced-land-use-dataset"
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=save_path, unzip=True)

# ########## STANFORD CARS #########
KAGGLE_DATASET = 'rickyyyyyyy/torchvision-stanford-cars'
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=save_path, unzip=True)

