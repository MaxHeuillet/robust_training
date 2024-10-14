# import torchdatasets as td
# from datasets.datasets import get_dataset

# class IndexedDataset(td.Dataset):
#     def __init__(self, args, train=True, train_transform=False):
#         super().__init__()
#         self.dataset = None #get_dataset(args, train=train, train_transform=train_transform)

#     def __getitem__(self, index):
#         data, target = self.dataset[index]
#         return data, target, index

#     def __len__(self):
#         return len(self.dataset)

#     def clean(self):
#         self._cachers = []

import torch
from torch.utils.data import Dataset
from PIL import Image
import io
from datasets.loaders import load_data

# self.transform = transform
# self.dataset = dataset
# self.imagenette_to_imagenet = {
#         0: 0,    # n01440764
#         1: 217,    # n02102040
#         2: 491,    # n02979186
#         3: 491,   # n03028079
#         4: 497,   # n03394916
#         5: 566,  # n03417042
#         6: 569,  # n03425413
#         7: 571,  # n03445777
#         8: 574,  # n03888257
#         9: 701   # n04251144  }
        
# Initializing the targets attribute
# if self.dataset_name == 'Imagenette':
#     self.targets = [self.imagenette_to_imagenet[d['label']] for d in self.dataset]
# elif self.dataset_name in ['Imagenet1k', 'CIFAR10', 'MNIST']:
#     self.targets = [d[1] for d in self.dataset]  # Assuming the dataset is a list of tuples (image, label)
# else:
#     print('Dataset name error')

# def get_item_imagenette(self,idx):
#     image = self.dataset[idx]['image']
#     label = self.imagenette_to_imagenet[ self.dataset[idx]['label'] ]
#     return image, label
    
class IndexedDataset(Dataset): #Dataset

    def __init__(self, args, dataset, transform, N): #, transform=None

        super().__init__()

        self.args = args
        self.dataset = dataset
        self.N = N
        self.dataset_name = args.dataset
        self.global_indices = list(range(len(dataset)))
        self.transform = transform  # Add transform parameter

    
    def __len__(self):
        return len(self.dataset)
    
    def get_item_skip(self, idx):
        try:
            data = self.dataset[idx]
            if isinstance(data, tuple) and len(data) == 2:
                return data  # Unpack directly if data is correctly formatted
            else:
                raise ValueError(f"Unexpected data format at index {idx}: {data}")
        except Exception as e:
            next_idx = (idx + 1) % self.__len__()
            print(f"Error with index {idx}: {e}. Trying next index {next_idx}")
            return self.get_item_imagenet(next_idx)

    def get_item_noskip(self,idx):
        image, label = self.dataset[idx]
        return image, label

    def __getitem__(self, idx):
        # print(idx)
        # Print the index or indices
        # print('idx', idx)

        # Handle the case where idx is a list of indices
        if isinstance(idx, list):
            # Pre-allocate lists or use list comprehension
            images = [None] * len(idx)
            labels = [None] * len(idx)
            indices = [None] * len(idx)

            for i, single_idx in enumerate(idx):
                image, label, _ = self._get_single_item(single_idx)
                images[i] = image
                labels[i] = label
                indices[i] = single_idx
            
            # print( torch.stack(images), torch.tensor(labels), torch.tensor(indices) )
            return torch.stack(images), torch.tensor(labels), torch.tensor(indices)
        else:
            # Handle the case where idx is a single index
            return self._get_single_item(idx)
        
    def _get_single_item(self, idx):
        if self.dataset_name == 'Imagenet1k':
            image_data, label = self.get_item_skip(idx)
        elif self.dataset_name in ['CIFAR10', 'CIFAR10s', 'MNIST', 'random']:
            image_data, label = self.get_item_noskip(idx)
        else:
            raise ValueError("Unsupported dataset name")

        # Process the image data
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data

        # Apply the transform if it exists
        if self.transform is not None:
            image = self.transform(image)
        
        # Return the processed image, label, and index
        return image, label, idx
    
