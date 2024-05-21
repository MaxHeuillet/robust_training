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



def setup(rank, world_size):
    # Initialize the distributed environment.

    print('set up the master adress and port')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Set up the local GPU for this process
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print('init process group ok')

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):


    setup(rank, world_size)

    print('load dataset')
    dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch')

    print('load sampler')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    print('load dataloader')
    dataloader = DataLoader(dataset, batch_size=1024, sampler=sampler, num_workers=4)

    # Model, Loss, and Optimizer
    model = resnet50().cuda(rank)
    model = DDP(model, device_ids=[rank])
    print('load model')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(1):  # number of epochs

        print( 'epoch {}'.format(epoch) )
        sampler.set_epoch(epoch)

        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(rank)
            targets = targets.cuda(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {i}, Loss {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
