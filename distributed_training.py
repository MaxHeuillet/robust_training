import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
# from torchvision.models import resnet50
from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoImageProcessor, ResNetModel
import torch
from datasets import load_dataset, load_from_disk

import torchvision.models as models

import os
import io

from torch.utils.data import Dataset

from PIL import Image

from torchvision.models import resnet50, ResNet50_Weights



import torch.nn.utils.parametrize as parametrize
import lora 

import active
import math

from torch.utils.data import Subset
import trades
import utils
import sys

import models_local
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.distributed as dist
from torchvision.transforms.functional import InterpolationMode


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
        return image, label, idx



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


def to_rgb(x):
    return x.convert("RGB")


class Experiment:

    def __init__(self, n_rounds, size, nb_epochs, seed, active_strategy, data, model, world_size):

        self.n_rounds = n_rounds
        self.size = size
        self.epochs = nb_epochs
        self.seed = seed
        self.active_strategy = active_strategy
        self.data = data
        self.model = model

        self.world_size = world_size

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

    def load_model(self,):
        if self.model == 'resnet50' and self.data == 'CIFAR10':
            # TODO
            model = models_local.resnet.resnet50(pretrained=True, progress=True, device="cuda").to('cuda')

        elif self.model == 'resnet50' and self.data in ['Imagenet-1k' , 'Imagenette']:
            model = models.resnet50()
            state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
            model.load_state_dict(state_dict)
            model.to('cuda')

        #self.add_lora(model)

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
        
    
    def load_data(self,):

        if self.data == 'CIFAR10':

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) )  ])

            pool_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

            test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
            
        elif self.data == 'Imagenet-1k':

            # transform = transforms.Compose([
            #     transforms.Lambda(to_rgb),
            #     transforms.Resize(256),
            #     transforms.CenterCrop(224),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])

            transform = transforms.Compose([
                transforms.Lambda(to_rgb),
                transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  # Resize the image to 232x232
                transforms.CenterCrop(224),  # Crop the center of the image to make it 224x224
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
            ])

            dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch',)

            pool_dataset = CustomImageDataset(dataset['train'], transform= transform )
        
            test_dataset = CustomImageDataset(dataset['test'], transform= transform ) 

            print('load dataloader')

        elif self.data == 'Imagenette':

            transform = transforms.Compose([
                transforms.Lambda(to_rgb),
                transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  # Resize the image to 232x232
                transforms.CenterCrop(224),  # Crop the center of the image to make it 224x224
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
            ])


            # dataset = load_dataset("frgfm/imagenette", "full_size", cache_dir='/home/mheuill/scratch')
            dataset = load_from_disk('/home/mheuill/scratch/imagenette')

            pool_dataset = CustomImageDataset(dataset['train'], transform= transform )
        
            test_dataset = CustomImageDataset(dataset['validation'], transform= transform )

            print('load dataloader')
            
        else:
            print('undefined data')

        
        # test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=1024, sampler=test_sampler, num_workers=self.world_size)

        return pool_dataset, test_dataset
    
    def evaluation(self, rank, args):

        state_dict, test_dataset, accuracy_tensor = args

        setup(self.world_size, rank)

        sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(test_dataset, batch_size=1024, sampler=sampler, num_workers=self.world_size)

        model = self.load_model()
        model.load_state_dict(state_dict)
        model.to(rank)
        model.eval()  # Ensure the model is in evaluation mode
        model = DDP(model, device_ids=[rank])
        

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs,labels, _ in loader:
                labels = labels.cuda(rank)
                inputs = inputs.cuda(rank)
                outputs = model(inputs) 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

         # Tensorize the correct and total for reduction
        correct_tensor = torch.tensor(correct).cuda(rank)
        total_tensor = torch.tensor(total).cuda(rank)

        print(correct_tensor, total_tensor)

        # Use dist.all_reduce to sum the values across all processes
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        print(correct_tensor, total_tensor)

        # Compute accuracy and store it in the shared tensor
        accuracy_tensor[rank] = correct_tensor.item() / total_tensor.item()

        print(accuracy_tensor)

    
    def uncertainty_sampling(self, rank, args):
        state_dict, pool_dataset, N, top_n_indices = args

        setup(self.world_size, rank)

        sampler = DistributedSampler(pool_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(pool_dataset, batch_size=1024, sampler=sampler, num_workers=self.world_size, shuffle=False)

        model = self.load_model()
        model.load_state_dict(state_dict)
        model.to(rank)
        model.eval()  # Ensure the model is in evaluation mode
        model = DDP(model, device_ids=[rank])

        predictions = []
        indices_list = []
        with torch.no_grad():
            for inputs, _, indices in loader:
                inputs = inputs.cuda(rank)
                outputs = model(inputs) 
                predictions.append(outputs)
                indices_list.append( indices.cuda(rank) )

        predictions = torch.cat(predictions, dim=0)
        indices = torch.cat(indices_list, dim=0)

 
        gather_list = [torch.zeros_like(predictions) for _ in range(self.world_size)]
        index_gather_list = [torch.zeros_like(indices) for _ in range(self.world_size)]
        print(gather_list)
        print(index_gather_list)

        dist.all_gather(gather_list, predictions)
        dist.all_gather(index_gather_list, indices)

        # print(gather_list.shape)
        # print(index_gather_list.shape)

        if rank == 0:
            all_predictions = torch.cat(gather_list, dim=0)
            all_indices = torch.cat(index_gather_list, dim=0)
            print(all_predictions)
            print(all_indices)
            outputs = torch.softmax(all_predictions, dim=1)
            uncertainty = 1 - torch.max(outputs, dim=1)[0]
            # Direct indexing instead of sorting
            _, top_n_indices_sub = torch.topk(uncertainty, N, largest=True)
            top_n_indices.copy_(all_indices[top_n_indices_sub])

        cleanup()



    def update(self,rank, args):

        state_dict, pool_dataset = args

        setup(self.world_size, rank)

        sampler = DistributedSampler(pool_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(pool_dataset, batch_size=1024, sampler=sampler, num_workers=self.world_size)

        model = self.load_model()
        model.load_state_dict(state_dict)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD( model.parameters(),lr=0.001, weight_decay=0.0001, momentum=0.9, nesterov=True, )

        for epoch in range(self.epochs):
            sampler.set_epoch(epoch)
            for data, target in loader:
                data, target = data.to(rank), target.to(rank)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f'Rank {rank}, Epoch {epoch}, Loss {loss.item()}')

        print('clean up')
        cleanup()


    def launch_experiment(self,  ):

        result = {}
        result['active_strategy'] = self.active_strategy
        result['n_rounds'] = self.n_rounds
        result['size'] = self.size
        result['nb_epochs'] = self.epochs
        result['seed'] = self.seed
        result['data'] = self.data
        result['model'] = self.model

        utils.set_seeds(self.seed)

        pool_dataset, test_dataset = self.load_data()

        model = self.load_model()
        state_dict = model.state_dict()
        
        accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
        accuracy_tensor.share_memory_()  # 
        arg = (state_dict, test_dataset, accuracy_tensor)
        torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
        clean_acc = accuracy_tensor[0]
        
        print('clean_acc', clean_acc)
        
        epoch_counter = 0
        round_size = math.ceil(self.size / self.n_rounds)

        total_indices = list( range( len( pool_dataset ) ) )
        collected_indices = []

        for i in range(self.n_rounds):

            print( 'iteration {}'.format(i) )

            # pool_loader = DataLoader( Subset(pool_dataset, pool_indices), batch_size=1000, shuffle=False, num_workers=self.world_size)
            
            if self.active_strategy == 'uncertainty':
                top_n_indices = torch.zeros(round_size, dtype=torch.long)  # Placeholder for the indices, shared memory
                top_n_indices.share_memory_()
                arg = (state_dict, pool_dataset, round_size, top_n_indices)
                torch.multiprocessing.spawn(self.uncertainty_sampling,
                                               args=(arg,),
                                               nprocs=self.world_size, join=True)
                # for i in top_n_indices:
                #     print(i)
                #selected_indices =  selected_indices[0] 
                print(top_n_indices)
            
            else:
                print('error')

                        

if __name__ == "__main__":
    n_rounds = 1
    size = 100
    nb_epochs = 1
    seed = 1
    active_strategy = 'uncertainty'
    data = 'Imagenette'
    model = 'resnet50'
    world_size = torch.cuda.device_count()
    evaluator = Experiment(n_rounds, size, nb_epochs, seed, active_strategy, data, model, world_size)
    print('begin experiment')
    evaluator.launch_experiment()



    # torch.multiprocessing.spawn(inference, args=(world_size,), nprocs=world_size)

            #pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif self.active_strategy == 'entropy':
            #     query_indices = active.entropy_sampling(model, pool_loader, round_size)
            #     pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif self.active_strategy == 'margin':
            #     query_indices = active.margin_sampling(model, pool_loader, round_size)
            #     pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif self.active_strategy == 'random':
            #     query_indices = active.random_sampling(model, pool_loader, round_size)
            #     pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif self.active_strategy == 'full':
            #     query_indices = utils.get_indices_for_round( len(pool_loader.dataset), self.n_rounds, i)
            #     _ = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif args.active_strategy == 'attack':
            #     query_indices = active.attack_sampling(model, pool_loader, round_size)
            #     pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
            # elif args.active_strategy == 'attack_uncertainty':
            #     query_indices = active.attack_uncertainty_sampling(model, pool_loader, round_size)
            #     pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)


            ################# Update the ResNet through low rank matrices:

        #     adapt_loader = DataLoader(adapt_dataset, batch_size=128, shuffle=True)

        #     print('start training process')
        #     sys.stdout.flush()

        #     for _ in range(self.nb_epochs):
        #         print('epoch {}'.format(_) )
        #         sys.stdout.flush()
        #         model.train()

        #         #Check if epoch_counter is at a milestone to reduce learning rate
        #         if epoch_counter == 30 or epoch_counter == 50:
        #             current_lr = optimizer.param_groups[0]['lr']
        #             new_lr = current_lr * 0.1
        #             utils.update_learning_rate(optimizer, new_lr)
        #             print("Reduced learning rate to {}".format(new_lr))

        #         # if args.loss == 'AT':
        #         #     err, loss = utils.epoch_AT_vanilla(adapt_loader, model, device, optimizer)
        #         # elif args.loss == 'FAT':
        #         #     err, loss = utils.epoch_fast_AT(adapt_loader, model, device, optimizer)
        #         # elif args.loss == 'TRADES':
        #         err, loss = utils.epoch_TRADES(adapt_loader, model, device, optimizer)

        #         print('epoch finished {} - {}'.format(err, loss) )
        #         sys.stdout.flush()
        #         epoch_counter +=1

        # print('total amount of data used: {}'.format( len(adapt_loader)  )  )
    
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

# def inference(rank, world_size):
#     print('inference', world_size, rank)
#     setup(world_size, rank)

#     print('load dataset')
#     dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch',)
    
#     print('initialize image processor')
#     # image_processor = AutoImageProcessor.from_pretrained("/home/mheuill/scratch/resnet-50", local_files_only=True)
#     # dataset = image_processor(dataset)

#     print('create custom dataset')
#     # print(dataset['train'][0]['image'])


#     transform = transforms.Compose([
#         transforms.Lambda(to_rgb),
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

#     # Instantiate the custom dataset
#     dataset = CustomImageDataset(dataset['test'], transform=transform)

#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

#     print('load dataloader')
#     dataloader = DataLoader(dataset, batch_size=1024, sampler=sampler, num_workers=world_size)

#     # Load model
#     model = models.resnet50().to("cuda")
#     # Load the state dictionary from the file
#     state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
#     model.load_state_dict(state_dict)

#     # model = ResNetModel.from_pretrained('/home/mheuill/scratch/resnet-50', local_files_only=True) #resnet50().cuda(rank)
#     model = model.cuda(rank)
#     model = DDP(model, device_ids=[rank])
    
#     model.eval()

#     predictions = []
#     time = 0
#     print('start the loop')
#     with torch.no_grad():
#         for inputs, _ in dataloader:
#             inputs = inputs.cuda(rank)
#             outputs = model(inputs) #.last_hidden_state
#             predictions.append( outputs )

#             time +=1
            
#     print('Gather all predictions to the process 0')
#     predictions = torch.cat(predictions, dim=0)
#     gather_list = [torch.zeros_like(predictions) for _ in range(world_size)]
#     dist.all_gather(gather_list, predictions)


#     if rank == 0:
#         # Concatenate all predictions on rank 0
#         all_predictions = torch.cat(gather_list, dim=0).cpu()
#         print(all_predictions.shape)  # This will show the total number of predictions
#         return all_predictions

#     print('clean up')
#     cleanup()