from torchvision import datasets
import os
import os
import kaggle


###################
### KVASIR DATASET
###################

import os
import tarfile
import zipfile
import kaggle

save_path = os.path.expanduser('~/scratch/data')

######### KVASIR DATASET #########  

KAGGLE_DATASET = "meetnagadia/kvasir-dataset"
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=save_path, unzip=False)

########## UC MERCED LAND USE DATASET #########

KAGGLE_DATASET = "abdulhasibuddin/uc-merced-land-use-dataset"
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=save_path, unzip=False)




# print('load1')
# datasets.StanfordCars(root='./data',  download=False, )
#

# Ensure environment variables are set
# username = os.getenv('KAGGLE_USERNAME')
# key = os.getenv('KAGGLE_KEY')

# if not username or not key:
#     raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set as environment variables.")

# # Authenticate using environment variables
# os.environ['KAGGLE_USERNAME'] = username
# os.environ['KAGGLE_KEY'] = key

# # Define dataset and download path
# dataset = 'zillow/zecon'  # Replace with your desired dataset identifier
# download_path = 'data/'    # Replace with your desired download path

# # Create download directory if it doesn't exist
# os.makedirs(download_path, exist_ok=True)

# # Download and unzip the dataset
# kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)

# print(f"Dataset '{dataset}' downloaded and extracted to '{download_path}'.")



# # Verify environment variables
# print("KAGGLE_USERNAME:", os.getenv('KAGGLE_USERNAME'))
# print("KAGGLE_KEY:", os.getenv('KAGGLE_KEY'))

# import kaggle
# # you need to configure API key through https://www.kaggle.com/docs/api
# kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path='~/scratch/data', unzip=True)


# print('load2')
# datasets.OxfordIIITPet(root='$SLURM_TMPDIR/data',  download=True, )


# print('load3')

# datasets.Caltech101(root='~/scratch/data',  download=True, )


# print('load4')
# datasets.DTD(root='~/scratch/data',  download=True, )




# print ('download EuroSAT')

# import torchvision
# import ssl
# import pandas as pd
# ssl._create_default_https_context = ssl._create_unverified_context
# torchvision.datasets.EuroSAT('./data', download=True)

# print('download aircraft')

# datasets.FGVCAircraft(root='~/scratch/data', split='train', download=True, )
# datasets.FGVCAircraft(root='~/scratch/data', split='val', download=True, )
# datasets.FGVCAircraft(root='~/scratch/data', split='test', download=True, )

# print('download CIFAR100')

# datasets.CIFAR100(root='~/scratch/data', train=True, download=True)
# datasets.CIFAR100(root='~/scratch/data', train=False,  download=True)

# print('download CIFAR10')

# datasets.CIFAR10(root='~/scratch/data', train=True, download=True)
# datasets.CIFAR10(root='~/scratch/data', train=False,  download=True)

# print('download MNIST')

# datasets.MNIST(root='~/scratch/data', train=True, download=True)
# datasets.MNIST(root='~/scratch/data', train=False,  download=True)



# from datasets import load_dataset 

# print('load')
# dataset = load_dataset("frgfm/imagenette", "full_size", cache_dir='/home/mheuill/scratch')
# print('save')
# dataset.save_to_disk('/home/mheuill/scratch/imagenette')


# print('load')
# dataset = load_dataset('imagenet-1k',  cache_dir='/home/mheuill/scratch')
# print('save')
# dataset.save_to_disk('/home/mheuill/scratch/imagenet-1k')


# from omegaconf import OmegaConf
# from datasets import load_data
# import numpy as np
# config = OmegaConf.load("./configs/default_config.yaml")

# for dataset in ['StanfordCars', 'OxfordIIITPet', 
#                 'Caltech101', 'DTD',
#                  'Imagenette', 'Flowers', 
#                 'EuroSAT', 'Aircraft',
#                 'CIFAR10', 'CIFAR100',  ]:
    
#     print(dataset)
    
#     config.dataset = dataset
#     train_dataset, val_dataset, test_dataset, N, train_transform, transform = load_data(False, config) 
#     image, label = val_dataset[0]  # Extract the image and label
#     pixels = np.array(image)
#     print(dataset, len(train_dataset), len(val_dataset), len(test_dataset), min(pixels[0][0]),  max(pixels[0][0]), N)