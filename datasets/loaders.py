
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

from sklearn.model_selection import train_test_split

def load_data(args):

    if args.dataset == 'MNIST':

        # Load the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))    ])

        N = 10
        
        dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, )

        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, )


    elif args.dataset == 'CIFAR10':

        if args.aug == 'aug' and args.pre_trained == 'no':
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomCrop(32, padding=4), 
                                        transforms.RandomHorizontalFlip(0.5), 
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ),])
        elif args.aug == 'noaug' and args.pre_trained == 'no':
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ),])
        
        elif args.aug == 'aug' and args.pre_trained != 'no': 
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomCrop(32, padding=4), 
                                        transforms.RandomHorizontalFlip(0.5), 
                                        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),])
        else:
            print('undefined case')

        N = 10
        

        dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, )

        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        print("train size", train_size, "val size", val_size)

        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator1)  
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, )

    elif args.dataset == 'CIFAR100':
        
        if args.aug == 'aug' and args.pre_trained != 'no': 
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomCrop(32, padding=4), 
                                        transforms.RandomHorizontalFlip(0.5), 
                                        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),])
        else:
            print('undefined case')

        N = 10
        
        dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=False, )

        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        print("train size", train_size, "val size", val_size)

        generator1 = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator1)  
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, )

    elif args.dataset == 'Aircraft':
        
        if args.aug == 'aug' and args.pre_trained != 'no': 
            transform = transforms.Compose([
                                        transforms.Resize((224, 224)),  # Resize images to 224x224
                                        transforms.ToTensor(),
                                        transforms.RandomCrop(224, padding=4), 
                                        transforms.RandomHorizontalFlip(0.5), 
                                        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),])
        else:
            print('undefined case')

        N = 100
        
        train_dataset = datasets.FGVCAircraft(root=args.data_dir, split='train', download=False, )
        val_dataset =   datasets.FGVCAircraft(root=args.data_dir, split='val', download=False, )
        test_dataset = datasets.FGVCAircraft(root=args.data_dir, split='test', download=False, )

    elif args.dataset == 'Eurosat':

        if args.aug == 'aug' and args.pre_trained != 'no': 
            transform = transforms.Compose([
                                        transforms.Resize((64, 64)),  # Resize images to 224x224
                                        transforms.ToTensor(),
                                        transforms.RandomCrop(224, padding=4), 
                                        transforms.RandomHorizontalFlip(0.5), 
                                        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),])
        else:
            print('undefined case')

        # Load the dataset
        dataset = EuroSATDataset(root_dir='./data/EuroSAT/2750')

        # Extract labels from the dataset
        labels = [label for _, label in dataset]

        # Split the dataset into train+val and test, keeping stratification
        train_val_indices, test_indices = train_test_split( range(len(labels)), test_size=0.2, stratify=labels, random_state=42 )

        # Extract labels for the train+val set for further stratification
        train_val_labels = [labels[i] for i in train_val_indices]

        # Split the train+val set into train and validation, keeping stratification
        train_indices, val_indices = train_test_split( train_val_indices, test_size=0.15, stratify=train_val_labels, random_state=42 )  # 0.25 * 0.8 = 0.2 of the dataset

        # Create subsets for train, validation, and test

        N = 10

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
    return train_dataset, val_dataset, test_dataset, N, transform

def to_rgb(x):
    return x.convert("RGB")



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