import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet50
from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoImageProcessor, ResNetModel
import torch
from datasets import load_dataset

import os
import io

from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from PIL import Image
import torch

class CustomImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Get the image and label from the Hugging Face dataset
        item = self.hf_dataset[idx]
        image = item['image']

        print(image)

        image = self.transform(images=image)['pixel_values'][0]

        # Labels can be handled here if needed
        label = item.get('label', torch.tensor(-1))  # Dummy label handling

        print(image)
        print(label)
        print()

        return image, label

print('load dataset')
dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch')
test_dataset = dataset['test']  # Select the test split

print('initialize image processor')
image_processor = AutoImageProcessor.from_pretrained("/home/mheuill/scratch/resnet-50", local_files_only=True)

print('create custom dataset')
custom_dataset = CustomImageDataset(test_dataset, transform=image_processor)

print('load dataloader')
dataloader = DataLoader(custom_dataset, batch_size=1)

time = 0
for inputs, _ in dataloader:
    print(inputs.shape)
    time +=1
    if time == 5:
        break
            

