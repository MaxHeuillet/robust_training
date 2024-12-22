from torchvision import datasets

# print('load1')
# datasets.StanfordCars(root='~/scratch/data',  download=True, )

# import kaggle
# you need to configure API key through https://www.kaggle.com/docs/api
# kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path='~/scratch/data', unzip=True)


# print('load2')
# datasets.OxfordIIITPet(root='~/scratch/data',  download=True, )


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


from omegaconf import OmegaConf
from datasets import load_data
import numpy as np
config = OmegaConf.load("./configs/default_config.yaml")

for dataset in ['Imagenette', 'Flowers', 
                'EuroSAT', 'Aircraft',
                'CIFAR10', 'CIFAR100',
                'StanfordCars', 'OxfordIIITPet', 
                'Caltech101', 'DTD' ]:
    
    config.dataset = dataset
    train_dataset, val_dataset, test_dataset, N, train_transform, transform = load_data(False, config) 
    image, label = val_dataset[0]  # Extract the image and label
    pixels = np.array(image)
    print(dataset, len(train_dataset), len(val_dataset), len(test_dataset), min(pixels[0][0]),  max(pixels[0][0]), N)