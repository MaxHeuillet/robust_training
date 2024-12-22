
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import os
import torch
from datasets.semisupervised_dataset import SemiSupervisedDataset
from torch.utils.data import TensorDataset
import random
from torch.utils.data import Subset
from torch.utils.data import random_split
from datasets.eurosat import EuroSATDataset
from torchvision.transforms import RandAugment
from torchvision.transforms.v2 import CutMix

from sklearn.model_selection import train_test_split
from utils import get_data_dir

#   transforms.RandomCrop(224, padding=4), 
#   transforms.RandomHorizontalFlip(),  # Random horizontal flip
##   RandAugment(),  # RandAugment with N=9, M=0.5


#Change Grayscale Image to RGB for the shape
class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img
    

def load_data(hp_opt,config,):

    dataset =config.dataset
    datadir = get_data_dir(hp_opt,config)

    train_transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to 224x224
                                          GrayscaleToRGB(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(.25,.25,.25),
                                          transforms.RandomRotation(2),
                                          transforms.ToTensor(),
                                          transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),])

    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to 224x224
                                    GrayscaleToRGB(),
                                    transforms.ToTensor(),
                                    transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),])
        
    if dataset == 'CIFAR10':

        N = 10
        print('datadir', datadir)
        train_dataset = datasets.CIFAR10(root=datadir, train=True, download=False, transform=train_transform)
        dataset = datasets.CIFAR10(root=datadir, train=False, download=False, transform=transform)
        labels = [label for _, label in dataset]

        test_indices, val_indices = train_test_split( range(len(labels)), test_size=0.25, stratify=labels, random_state=42 )  

        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices) 

    elif dataset == 'CIFAR100':

        N = 100
        print('datadir', datadir)
        train_dataset = datasets.CIFAR100(root=datadir, train=True, download=False, transform=train_transform)
        dataset = datasets.CIFAR100(root=datadir, train=False, download=False, transform=transform)
        labels = [label for _, label in dataset]

        test_indices, val_indices = train_test_split( range(len(labels)), test_size=0.25, stratify=labels, random_state=42 )  

        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices) 
        

    elif dataset == 'Aircraft':

        N = 100
        
        train_dataset = datasets.FGVCAircraft(root=datadir, split='train', download=False, transform=train_transform)
        val_dataset =   datasets.FGVCAircraft(root=datadir, split='val', download=False, transform=transform)
        test_dataset = datasets.FGVCAircraft(root=datadir, split='test', download=False, transform=transform)

    elif dataset == 'DTD':

        N = 47
        
        train_dataset = datasets.DTD(root=datadir, split='train', download=False, transform=train_transform)
        val_dataset =   datasets.DTD(root=datadir, split='val', download=False, transform=transform)
        test_dataset = datasets.DTD(root=datadir, split='test', download=False, transform=transform)

    elif dataset == 'EuroSAT':

        N = 10

        dataset_train_full = datasets.EuroSAT(root=datadir, download=False, transform=train_transform)
        dataset_val_test_full = datasets.EuroSAT(root=datadir, download=False, transform=transform)

        labels = [label for _, label in dataset_train_full]

        train_val_indices, test_indices = train_test_split(
            range(len(labels)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_val_labels = [labels[i] for i in train_val_indices]

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.15,  # 0.15 * 0.8 = 0.12 -> 12% for validation
            stratify=train_val_labels,
            random_state=42
        )

        train_dataset = torch.utils.data.Subset(dataset_train_full, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_val_test_full, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_val_test_full, test_indices)

    elif dataset == 'Flowers':
            
        # Load the Flowers102 dataset
        N = 102
        train_dataset = datasets.Flowers102(root=datadir, split='train', download=False, transform=train_transform)
        val_dataset = datasets.Flowers102(root=datadir, split='val', download=False, transform=transform)
        test_dataset = datasets.Flowers102(root=datadir, split='test', download=False,  transform=transform)

    elif dataset == 'Imagenette':

        N = 10
        print('datadir', datadir)
        train_dataset = datasets.Imagenette(root=datadir, split='train', download=False,  transform=train_transform)
        dataset = datasets.Imagenette(root=datadir, split='val', download=False,  transform=transform)
        labels = [label for _, label in dataset]

        test_indices, val_indices = train_test_split( range(len(labels)), test_size=0.25, stratify=labels, random_state=42 )  

        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

    elif dataset == 'Caltech101':

        N = 101
        dataset_train_full = datasets.Caltech101(root=datadir, download=False, transform=train_transform)
        dataset_val_test_full = datasets.Caltech101(root=datadir, download=False, transform=transform)

        labels = [label for _, label in dataset_train_full]

        train_val_indices, test_indices = train_test_split(
            range(len(labels)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_val_labels = [labels[i] for i in train_val_indices]

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.15,  # 0.15 * 0.8 = 0.12 -> 12% for validation
            stratify=train_val_labels,
            random_state=42
        )

        train_dataset = torch.utils.data.Subset(dataset_train_full, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_val_test_full, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_val_test_full, test_indices)

    elif dataset == 'OxfordIIITPet':

        N = 37

        # Step 1: Load the 'trainval' split twice with different transforms
        dataset_train_val_full_train = datasets.OxfordIIITPet(
            root=datadir,
            split='trainval',
            download=False,
            transform=train_transform
        )

        dataset_train_val_full_val = datasets.OxfordIIITPet(
            root=datadir,
            split='trainval',
            download=False,
            transform=transform
        )

        # Step 2: Load the 'test' split with transform
        dataset_test_full = datasets.OxfordIIITPet(
            root=datadir,
            split='test',
            download=False,
            transform=transform
        )

        # Step 3: Extract labels from the 'trainval' split
        labels = [label for _, label in dataset_train_val_full_train]

        # Step 4: Split 'trainval' into 'train' and 'val' using stratified sampling
        train_indices, val_indices = train_test_split(
            range(len(labels)),
            test_size=0.15,  # 15% for validation
            stratify=labels,
            random_state=42
        )

        # Step 5: Create training and validation subsets with appropriate transforms
        train_dataset = torch.utils.data.Subset(dataset_train_val_full_train, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_train_val_full_val, val_indices)
        test_dataset = dataset_test_full

    elif 'StanfordCars':

        N = 196
        print('datadir', datadir)

        # Step 1: Load the 'train' split with train_transform
        dataset_train_full_train = datasets.StanfordCars(
            root=datadir,
            split='train',
            download=False,
            transform=train_transform
        )

        # Step 2: Load the 'train' split again with transform for validation
        dataset_train_full_val = datasets.StanfordCars(
            root=datadir,
            split='train',
            download=False,
            transform=transform
        )

        # Step 3: Load the 'test' split with transform
        dataset_test_full = datasets.StanfordCars(
            root=datadir,
            split='test',
            download=False,
            transform=transform
        )

        # Step 4: Extract labels from the 'train' split
        labels = [label for _, label in dataset_train_full_train]

        # Step 5: Split 'train' into 'train' and 'val' using stratified sampling
        train_indices, val_indices = train_test_split(
            range(len(labels)),
            test_size=0.15,  # 15% for validation
            stratify=labels,
            random_state=42
        )

        # Step 6: Create training and validation subsets with appropriate transforms
        train_dataset = torch.utils.data.Subset(dataset_train_full_train, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_train_full_val, val_indices)
        test_dataset = dataset_test_full
                 
    return train_dataset, val_dataset, test_dataset, N, train_transform, transform

def to_rgb(x):
    return x.convert("RGB")



    # if args.dataset == 'MNIST':

    #     # Load the dataset
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))    ])

    #     N = 10
        
    #     dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, )

    #     train_size = int(0.95 * len(dataset))
    #     val_size = len(dataset) - train_size

    #     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #     test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, )

    # elif args.dataset == 'CIFAR10s':

    #     if args.aug == 'aug':
    #         transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.RandomCrop(32, padding=4), 
    #                                     transforms.RandomHorizontalFlip(0.5), 
    #                                     transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ),])
    #     elif args.aug == 'noaug':
    #         transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ),])
    #     else: 
    #         print('undefined augmentation')

    #     # transform = transforms.Compose([
    #     #         transforms.ToTensor(),
    #     #          ])

    #     if train:
    #         dataset = SemiSupervisedDataset(args, train=True, )
    #     else:
    #         dataset = SemiSupervisedDataset(args, train=False, )

    #     N = 10


            
    # elif args.dataset == 'Imagenet1k':

    #     transform = transforms.Compose([
    #             transforms.Lambda(to_rgb),
    #             transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  # Resize the image to 232x232
    #             transforms.CenterCrop(224),  # Crop the center of the image to make it 224x224
    #             transforms.ToTensor(),  # Convert the image to a tensor
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ])

    #     if train:
    #         dataset = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/train'))
    #     else:
    #         dataset = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/val'))

    #         # Set the seed for reproducibility
    #         seed = 42  # You can choose any seed value
    #         random.seed(seed)

    #         all_indices = list(range(len(dataset)))

    #         # Randomly sample 10,000 indices from the dataset
    #         sampled_indices = random.sample(all_indices, 10000)

    #         # Create a subset using the randomly sampled indices
    #         dataset = Subset(dataset, sampled_indices)


    #     # pool_dataset = IndexedDataset('Imagenet1k', train_folder, transform= transform ) 
    #     # test_dataset = IndexedDataset('Imagenet1k', test_folder, transform= transform) 
    #     N = 1000

    #     print('load dataloader')

    # elif args.dataset == 'random':

    #     transform = None
        
    #     # Create 10 unique integers as observations
    #     observations = torch.arange(10)

    #     # Create corresponding labels (for example, labels can be the same as observations, or any other related value)
    #     labels = observations.clone()  # Here, labels are the same as observations for simplicity

    #     # Combine observations and labels into a TensorDataset
    #     dataset = TensorDataset(observations.view(-1, 1), labels.view(-1, 1))
    #     N = 10
            


    # elif self.data == 'Imagenette':

    #     transform = transforms.Compose([
    #             transforms.Lambda(to_rgb),
    #             transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  
    #             transforms.CenterCrop(224),  
    #             transforms.ToTensor(), 
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ])

    #     dataset = load_dataset("frgfm/imagenette", "full_size", cache_dir='/home/mheuill/scratch')
    #     dataset = load_from_disk('/home/mheuill/scratch/imagenette')
    #     train_dataset, test_dataset = dataset['train'], dataset['validation']

    #     # pool_dataset = IndexedDataset('Imagenette', dataset['train'], transform= transform )
    #     # test_dataset = IndexedDataset('Imagenette', dataset['validation'], transform= transform )

    #     print('load dataloader')