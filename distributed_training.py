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
        # Here 'image' is likely a PIL image; you can check by adding print(type(image)) if unsure
        image_data = self.hf_dataset[idx]['image']
        #print(image_data)
        
        # If the images are being loaded as PIL Images, apply the transform
        # Make sure that the image is opened correctly if it's not already a PIL Image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        if self.transform:
            #print("transorm appplied")
            image = self.transform(image)

        #print(image_data, image_data)
        label = self.hf_dataset[idx]['label']
        return image, label



def setup(rank, world_size):
  #Initialize the distributed environment.
  print('set up the master adress and port')
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  #Set environment variables for offline usage of Hugging Face libraries
  os.environ['HF_DATASETS_OFFLINE'] = '1'
  os.environ['TRANSFORMERS_OFFLINE'] = '1'
  
  #Set up the local GPU for this process
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl", rank=rank, world_size=world_size)
  print('init process group ok')

def cleanup():
   dist.destroy_process_group()


def inference(world_size, rank):
    print(rank, world_size)
    setup(rank, world_size)

    print('load dataset')
    dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch',)
    
    print('initialize image processor')
    # image_processor = AutoImageProcessor.from_pretrained("/home/mheuill/scratch/resnet-50", local_files_only=True)
    # dataset = image_processor(dataset)

    print('create custom dataset')
    # print(dataset['train'][0]['image'])

    # Define your transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB") ),
        transforms.Resize(256),  # Resize so the shortest side is 256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Instantiate the custom dataset
    dataset = CustomImageDataset(dataset['test'], transform=transform)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    print('load dataloader')
    dataloader = DataLoader(dataset, batch_size=1024, sampler=sampler, num_workers=4)

    # Load model
    model = ResNetModel.from_pretrained('/home/mheuill/scratch/resnet-50', local_files_only=True) #resnet50().cuda(rank)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()

    predictions = []
    time = 0
    print('start the loop')
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.cuda(rank)
            outputs = model(inputs).last_hidden_state
            predictions.append(outputs)

            time +=1
            
    # Gather all predictions to the process 0
    predictions = torch.cat(predictions, dim=0)
    gather_list = [torch.zeros_like(predictions) for _ in range(world_size)]
    dist.all_gather(gather_list, predictions)

    if rank == 0:
        # Concatenate all predictions on rank 0
        all_predictions = torch.cat(gather_list, dim=0)
        print(all_predictions.shape)  # This will show the total number of predictions

    cleanup()

if __name__ == "__main__":
    print('begin experiment')
    print(torch.cuda.device_count())
    world_size = 4  
    torch.multiprocessing.spawn(inference, args=(world_size,), nprocs=world_size)
    
    
    
    
    
    
    
    
#torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
# Gather all predictions to the process 0
#predictions = torch.cat(predictions, dim=0)
#gather_list = [torch.zeros_like(predictions) for _ in range(world_size)]
#dist.all_gather(gather_list, predictions)
#if rank == 0:
# Concatenate all predictions on rank 0
#    all_predictions = torch.cat(gather_list, dim=0)
#   print(all_predictions.shape)  # This will show the total number of predictions
#cleanup()

# def train(rank, world_size):
#     setup(rank, world_size)
#     print('load dataset')
#     dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch')
#     dataset = dataset['train'] 
#     print(dataset['train'][0]['image'])
#     print('load sampler')
#     #sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#     print('load dataloader')
#     dataloader = DataLoader(dataset, batch_size=1024, num_workers=1)
#     # Model, Loss, and Optimizer
#     model = resnet50().cuda()
#     #model = DDP(model, device_ids=[rank])
#     print('load model')
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     # Training Loop
#     for epoch in range(1):  # number of epochs
#         print( 'epoch {}'.format(epoch) )
#         sampler.set_epoch(epoch)
#         for i, (inputs, targets) in enumerate(dataloader):
#             inputs = inputs.cuda()
#             targets = targets.cuda()
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             if i % 10 == 0:
#                 print(f"Rank {rank}, Epoch {epoch}, Batch {i}, Loss {loss.item()}")
#     cleanup()