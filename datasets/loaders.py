
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import os
import torch
# from datasets import load_dataset, load_from_disk

def load_data(args, train=True):

    if args.dataset == 'MNIST':

        # Load the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        if train:
            dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        else:
            dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
        K = 10

    elif args.dataset == 'CIFAR10':

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) )  ])

        if train:
            dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform)
        else:
            dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform)

        # pool_dataset = IndexedDataset('CIFAR10', train_folder, transform= transform ) 
        # test_dataset = IndexedDataset('CIFAR10', test_folder, transform= transform) 
        K = 10

        print('load dataloader')
            
    elif args.dataset == 'Imagenet1k':

        transform = transforms.Compose([
                transforms.Lambda(to_rgb),
                transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  # Resize the image to 232x232
                transforms.CenterCrop(224),  # Crop the center of the image to make it 224x224
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ])

        if train:
            dataset = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/train'), transform=transform)
        else:
            dataset = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/val'), transform=transform)

        # pool_dataset = IndexedDataset('Imagenet1k', train_folder, transform= transform ) 
        # test_dataset = IndexedDataset('Imagenet1k', test_folder, transform= transform) 
        K = 1000

        print('load dataloader')

    elif args.dataset == 'random':


        from torch.utils.data import TensorDataset

        # Create 10 unique integers as observations
        observations = torch.arange(7)

        # Create corresponding labels (for example, labels can be the same as observations, or any other related value)
        labels = observations.clone()  # Here, labels are the same as observations for simplicity

        # Combine observations and labels into a TensorDataset
        dataset = TensorDataset(observations.view(-1, 1), labels.view(-1, 1))
        K = 10

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
            
    else:
        print('undefined data')

    return dataset, K

def to_rgb(x):
    return x.convert("RGB")