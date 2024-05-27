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


import torch.nn.utils.parametrize as parametrize
import lora 

import active
import math

from torch.utils.data import Subset
import trades
import utils
import sys

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



def setup(world_size, rank):
  #Initialize the distributed environment.
  print( ' world size {}, rank {}'.format(world_size,rank) )
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


def inference(rank, world_size):
    print('inference', world_size, rank)
    setup(world_size, rank)

    print('load dataset')
    # dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch',)
    
    # print('initialize image processor')
    # # image_processor = AutoImageProcessor.from_pretrained("/home/mheuill/scratch/resnet-50", local_files_only=True)
    # # dataset = image_processor(dataset)

    # print('create custom dataset')
    # # print(dataset['train'][0]['image'])

    # # Define your transformations
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda x: x.convert("RGB") ),
    #     transforms.Resize(256),  # Resize so the shortest side is 256 pixels
    #     transforms.CenterCrop(224),  # Crop the center 224x224 pixels
    #     transforms.ToTensor(),  # Convert image to tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # # Instantiate the custom dataset
    # dataset = CustomImageDataset(dataset['test'], transform=transform)

    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # print('load dataloader')
    # dataloader = DataLoader(dataset, batch_size=1024, sampler=sampler, num_workers=4)

    # # Load model
    # model = ResNetModel.from_pretrained('/home/mheuill/scratch/resnet-50', local_files_only=True) #resnet50().cuda(rank)
    # model = model.cuda(rank)
    # model = DDP(model, device_ids=[rank])
    
    # model.eval()

    # predictions = []
    # time = 0
    # print('start the loop')
    # with torch.no_grad():
    #     for inputs, _ in dataloader:
    #         inputs = inputs.cuda(rank)
    #         outputs = model(inputs).last_hidden_state
    #         predictions.append(outputs)

    #         time +=1
            
    # # Gather all predictions to the process 0
    # predictions = torch.cat(predictions, dim=0)
    # gather_list = [torch.zeros_like(predictions) for _ in range(world_size)]
    # dist.all_gather(gather_list, predictions)

    # if rank == 0:
    #     # Concatenate all predictions on rank 0
    #     all_predictions = torch.cat(gather_list, dim=0)
    #     print(all_predictions.shape)  # This will show the total number of predictions

    cleanup()

if __name__ == "__main__":
    print('begin experiment')
    print(torch.cuda.device_count())
    world_size = 4  
    torch.multiprocessing.spawn(inference, args=(world_size,), nprocs=world_size)
    
    
    

import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Experiment:

    def __init__(self, n_rounds, size, nb_epochs, seed, active_strategy, data, model, world_size):

        self.n_rounds = n_rounds
        self.size = size
        self.nb_epochs = nb_epochs
        self.seed = seed
        self.active_strategy = active_strategy
        self.data = data
        self.model = model

        self.world_size = world_size

    def load_model(self,rank):
        if self.model == 'resnet50' and self.data == 'CIFAR10':
            model = models.resnet.resnet50(pretrained=True, progress=True, device="cuda").to('cuda')

        elif self.model == 'resnet50' and self.data == 'Imagenet-1k':
            # Load model
            model = models.resnet50(pretrained=True)
            model = model.cuda(rank)
            model = DDP(model, device_ids=[rank])
            print('model undefined')
        return model
    
    def add_lora(self, model):

        layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
                model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
                model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
                model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]

        for conv_layer in layers:
            lora_param = lora.layer_parametrization(conv_layer, device="cuda", rank=10, lora_alpha=1)
            parametrize.register_parametrization(conv_layer, 'weight', lora_param)

        lora.set_lora_gradients(model, layers)
        
    
    def load_data(self,rank):

        if self.data == 'CIFAR10':

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) )  ])

            pool_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

            test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
            

        elif self.data == 'Imagenet-1k':

            transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB") ),
            transforms.Resize(256),  # Resize so the shortest side is 256 pixels
            transforms.CenterCrop(224),  # Crop the center 224x224 pixels
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  ])

            dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch',)

            pool_dataset = CustomImageDataset(dataset['train'], transform=transform)
        
            test_dataset = CustomImageDataset(dataset['test'], transform=transform)

            print('load dataloader')
            

        else:
            print('undefined data')

        pool_sampler = DistributedSampler(pool_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        pool_loader = DataLoader(dataset, batch_size=1024, sampler=pool_sampler, num_workers=self.world_size)
        pool_indices = list( range(len(pool_loader.dataset ) ) )

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1024, sampler=test_sampler, num_workers=self.world_size)

        return pool_dataset, pool_sampler, pool_loader, pool_indices, test_loader



    def launch_experiment(self, world_size, rank ):
        result = {}
        result['active_strategy'] = self.active_strategy
        result['n_rounds'] = self.n_rounds
        result['size'] = self.size
        result['nb_epochs'] = self.nb_epochs
        result['seed'] = self.seed
        result['data'] = self.data
        result['model'] = self.model

        pool_dataset, pool_sampler, pool_loader, pool_indices, test_loader = self.load_data(rank)

        model = self.load_model(rank)

        self.add_lora(model)

        adapt_dataset = active.EmptyDataset()

        epoch_counter = 0
        round_size = math.ceil(self.size / self.n_rounds)

        for i in range(self.n_rounds):

            print( 'iteration {}'.format(i) )

            pool_loader = DataLoader( Subset(pool_dataset, pool_indices), batch_size=1000, shuffle=False)
                
            if self.active_strategy == 'uncertainty':
                query_indices = active.uncertainty_sampling(model, pool_loader, round_size)
                pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            elif self.active_strategy == 'entropy':
                query_indices = active.entropy_sampling(model, pool_loader, round_size)
                pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            elif self.active_strategy == 'margin':
                query_indices = active.margin_sampling(model, pool_loader, round_size)
                pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            elif self.active_strategy == 'random':
                query_indices = active.random_sampling(model, pool_loader, round_size)
                pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            elif self.active_strategy == 'full':
                query_indices = utils.get_indices_for_round( len(pool_loader.dataset), self.n_rounds, i)
                _ = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif args.active_strategy == 'attack':
            #     query_indices = active.attack_sampling(model, pool_loader, round_size)
            #     pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif args.active_strategy == 'attack_uncertainty':
            #     query_indices = active.attack_uncertainty_sampling(model, pool_loader, round_size)
            #     pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            else:
                print('error')

            adapt_loader = DataLoader(adapt_dataset, batch_size=128, shuffle=True)

            ################# Update the ResNet through low rank matrices:
            print('start training process')
            sys.stdout.flush()

            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer = torch.optim.SGD( model.parameters(),lr=0.001, weight_decay=0.0001, momentum=0.9, nesterov=True, )

            for _ in range(self.nb_epochs):
                print('epoch {}'.format(_) )
                sys.stdout.flush()
                model.train()

                #Check if epoch_counter is at a milestone to reduce learning rate
                if epoch_counter == 30 or epoch_counter == 50:
                    current_lr = optimizer.param_groups[0]['lr']
                    new_lr = current_lr * 0.1
                    utils.update_learning_rate(optimizer, new_lr)
                    print("Reduced learning rate to {}".format(new_lr))

                # if args.loss == 'AT':
                #     err, loss = utils.epoch_AT_vanilla(adapt_loader, model, device, optimizer)
                # elif args.loss == 'FAT':
                #     err, loss = utils.epoch_fast_AT(adapt_loader, model, device, optimizer)
                # elif args.loss == 'TRADES':
                err, loss = utils.epoch_TRADES(adapt_loader, model, device, optimizer)

                print('epoch finished {} - {}'.format(err, loss) )
                sys.stdout.flush()
                epoch_counter +=1

        print('total amount of data used: {}'.format( len(adapt_loader)  )  )









    
    
    
    
    
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