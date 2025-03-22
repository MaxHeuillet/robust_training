from torchvision import datasets
import os
from datasets.data_processing import process_trainvaltest, process_trainval, stratified_subsample
import torch
from corruptions import apply_portfolio_of_corruptions
from transforms import load_data_transforms
from torch.utils.data import Subset

from torch.utils.data import Dataset

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None, target_transform=None):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.subset)

def load_data(config, common_corruption=False):

    dataset = config.dataset
    datadir = os.path.abspath(os.path.expandvars(os.path.expanduser(config.datasets_path)))


    train_transform, transform = load_data_transforms()

    if dataset == 'uc-merced-land-use-dataset':

        N = 21
        path = datadir +'/UCMerced_LandUse/Images'
        dataset_train_full = datasets.ImageFolder( root=path )
        dataset_val_full = datasets.ImageFolder( root=path )
        dataset_test_full = datasets.ImageFolder( root=path )
        train_dataset, _, _ = process_trainvaltest(dataset_train_full, )
        _, val_dataset, _ = process_trainvaltest(dataset_val_full, )
        _, _, test_dataset = process_trainvaltest(dataset_test_full, )
 
    elif dataset == 'caltech101':

        N = 101
        dataset_train_full = datasets.Caltech101(root=datadir, download=False, )
        dataset_val_full = datasets.Caltech101(root=datadir, download=False, )
        dataset_test_full = datasets.Caltech101(root=datadir, download=False, )
        train_dataset, _, _ = process_trainvaltest(dataset_train_full, )
        _, val_dataset, _ = process_trainvaltest(dataset_val_full, )
        _, _, test_dataset = process_trainvaltest(dataset_test_full, )
                
    elif dataset == 'fgvc-aircraft-2013b': 

        N = 100
        train_dataset = datasets.FGVCAircraft(root=datadir, split='train', download=False, )
        val_dataset =   datasets.FGVCAircraft(root=datadir, split='val', download=False, )
        test_dataset = datasets.FGVCAircraft(root=datadir, split='test', download=False, )
        train_dataset = Subset(train_dataset, list(range(len(train_dataset)))  )
        val_dataset = Subset(val_dataset, list(range(len(val_dataset))) )
        test_dataset = Subset(test_dataset,  list(range(len(test_dataset))) )


    elif dataset == 'flowers-102':
        
        N = 102
        train_dataset = datasets.Flowers102(root=datadir, split='train', download=False)
        val_dataset = datasets.Flowers102(root=datadir, split='val', download=False)
        test_dataset = datasets.Flowers102(root=datadir, split='test', download=False)
        train_dataset = Subset(train_dataset, list(range(len(train_dataset)))  )
        val_dataset = Subset(val_dataset, list(range(len(val_dataset))) )
        test_dataset = Subset(test_dataset,  list(range(len(test_dataset))) )

    elif dataset == 'oxford-iiit-pet':

        N = 37
        dataset_train_full = datasets.OxfordIIITPet(root=datadir, split='trainval', download=False,)
        dataset_val_full = datasets.OxfordIIITPet(root=datadir, split='trainval', download=False,)
        dataset_test_full = datasets.OxfordIIITPet(root=datadir,split='test',download=False, )
        train_dataset, _ = process_trainval(dataset_train_full, )
        _, val_dataset = process_trainval(dataset_val_full, )
        test_dataset = dataset_test_full

    elif dataset == 'stanford_cars':

        N = 196
        dataset_train_full = datasets.StanfordCars( root=datadir, split='train', download=False,)
        dataset_val_full = datasets.StanfordCars( root=datadir, split='train', download=False,)
        dataset_test_full = datasets.StanfordCars( root=datadir, split='test', download=False, )
        train_dataset, _ = process_trainval(dataset_train_full, )
        _, val_dataset = process_trainval(dataset_val_full, )
        test_dataset = dataset_test_full
    
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    test_dataset = stratified_subsample(test_dataset, sample_size=1000)

    if common_corruption:
        test_dataset = apply_portfolio_of_corruptions(test_dataset, severity=3)

    train_dataset = TransformedSubset(train_dataset, transform=train_transform)
    val_dataset  = TransformedSubset(val_dataset,  transform=transform)
    test_dataset  = TransformedSubset(test_dataset,  transform=transform)
                       
    return train_dataset, val_dataset, test_dataset, N