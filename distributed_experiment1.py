import torch
# import torch.nn as nn
import torch.distributed as dist
# import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
# from torchvision.models import resnet50
# from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

# from transformers import AutoImageProcessor, ResNetModel
import torch
from datasets import load_dataset, load_from_disk

# import torchvision.models as models
import argparse
import os
import io

from torch.utils.data import Dataset

from PIL import Image

# from torchvision.models import resnet50, ResNet50_Weights

import gzip

import pickle as pkl

import torch.nn.utils.parametrize as parametrize
import lora 

# import active
import math

from torch.utils.data import Subset
import trades
import utils
# import sys

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.distributed as dist
from torchvision.transforms.functional import InterpolationMode

from models_local import resnet_imagenet, resnet_cifar10

import random

import sys

# import torch.multiprocessing as mp


class CustomImageDataset(Dataset):
    def __init__(self, data, hf_dataset, transform=None):
        self.data = data
        self.hf_dataset = hf_dataset
        self.transform = transform

        # if self.dataset == 'CIFAR10':
        # elif self.dataset == 'Imagenette':
        # elif self.dataset == 'Imagenet1k':

        self.imagenette_to_imagenet = {
                0: 0,    # n01440764
                1: 217,    # n02102040
                2: 491,    # n02979186
                3: 491,   # n03028079
                4: 497,   # n03394916
                5: 566,  # n03417042
                6: 569,  # n03425413
                7: 571,  # n03445777
                8: 574,  # n03888257
                9: 701   # n04251144 
                }

    def __len__(self):
        return len(self.hf_dataset)
    
    def get_item_imagenet(self, idx):
        try:
            data = self.hf_dataset[idx]
            if isinstance(data, tuple) and len(data) == 2:
                return data  # Unpack directly if data is correctly formatted
            else:
                raise ValueError(f"Unexpected data format at index {idx}: {data}")
        except Exception as e:
            next_idx = (idx + 1) % self.__len__()
            print(f"Error with index {idx}: {e}. Trying next index {next_idx}")
            return self.get_item_imagenet(next_idx)

    def get_item_imagenette(self,idx):
        image = self.hf_dataset[idx]['image']
        label = self.imagenette_to_imagenet[ self.hf_dataset[idx]['label'] ]
        return image, label
    
    def get_item_cifar10(self,idx):
        image, label = self.hf_dataset[idx]
        return image, label
    
    def __getitem__(self, idx):
        if self.data == 'Imagenet1k':
            image_data,label = self.get_item_imagenet(idx)
        elif self.data == 'Imagenette':
            image_data,label = self.get_item_imagenette(idx)
        elif self.data == 'CIFAR10':
            image_data,label = self.get_item_cifar10(idx)
        else:    
            print('error')

        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        if self.transform:
            #print("transorm appplied")
            image = self.transform(image)

        #print(image_data, image_data)
        
        return image, label, idx



def setup(world_size, rank):
  #Initialize the distributed environment.
  print( ' world size {}, rank {}'.format(world_size,rank) )
  print('set up the master adress and port')
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12354'
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
        if os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'beluga':
            self.batch_size_uncertainty = 512
            self.batch_size_update = 64
            self.batch_size_pgdacc = 64
            self.batch_size_cleanacc = 512
        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval':
            self.batch_size_uncertainty = 1024
            self.batch_size_update = 128
            self.batch_size_pgdacc = 128
            self.batch_size_cleanacc = 1024
        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'cedar':
            self.batch_size_uncertainty = 1024
            self.batch_size_update = 256
            self.batch_size_pgdacc = 256
            self.batch_size_cleanacc = 1024
        else:
            print('error')


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
            model = resnet_cifar10.ResNet(resnet_cifar10.Bottleneck, [3, 4, 6, 3] )
            state_dict = torch.load('./state_dicts/resnet50_cifar10.pt')
            model.load_state_dict(state_dict)
            model.to('cuda')

        elif self.model == 'resnet50' and self.data in ['Imagenet1k' , 'Imagenette']:

            # model = CustomResNet50()
            model = resnet_imagenet.ResNet(resnet_imagenet.Bottleneck, [3, 4, 6, 3], )
            state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
            model.load_state_dict(state_dict)
            model.to('cuda')

        self.add_lora(model)

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

            train_folder = datasets.CIFAR10(root='~/scratch/data', train=True)
            test_folder = datasets.CIFAR10(root='~/scratch/data', train=False)

            pool_dataset = CustomImageDataset('CIFAR10', train_folder, transform= transform ) 
            test_dataset = CustomImageDataset('CIFAR10', test_folder, transform= transform) 

            print('load dataloader')
            
        elif self.data == 'Imagenet1k':

            transform = transforms.Compose([
                transforms.Lambda(to_rgb),
                transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  # Resize the image to 232x232
                transforms.CenterCrop(224),  # Crop the center of the image to make it 224x224
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ])

            train_folder = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/train'))
            test_folder = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/val'))

            pool_dataset = CustomImageDataset('Imagenet1k', train_folder, transform= transform ) 
            test_dataset = CustomImageDataset('Imagenet1k', test_folder, transform= transform) 

            print('load dataloader')

        elif self.data == 'Imagenette':

            transform = transforms.Compose([
                transforms.Lambda(to_rgb),
                transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  
                transforms.CenterCrop(224),  
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ])

            # dataset = load_dataset("frgfm/imagenette", "full_size", cache_dir='/home/mheuill/scratch')
            dataset = load_from_disk('/home/mheuill/scratch/imagenette')

            pool_dataset = CustomImageDataset('Imagenette', dataset['train'], transform= transform )
        
            test_dataset = CustomImageDataset('Imagenette', dataset['validation'], transform= transform )

            print('load dataloader')
            
        else:
            print('undefined data')

        return pool_dataset, test_dataset
    

    
    def evaluation(self, rank, args):

        state_dict, test_dataset, accuracy_tensor, type = args

        setup(self.world_size, rank)

        sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        
        model = self.load_model()
        model.load_state_dict(state_dict)
        model.to(rank)
        model.eval()  # Ensure the model is in evaluation mode
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)

        if type=='clean_accuracy':
            loader = DataLoader(test_dataset, batch_size=self.batch_size_cleanacc, sampler=sampler, num_workers=self.world_size)
            correct, total = utils.compute_clean_accuracy(model, loader, rank)
        elif type == 'pgd_accuracy':
            loader = DataLoader(test_dataset, batch_size=self.batch_size_pgdacc, sampler=sampler, num_workers=self.world_size)
            correct, total = utils.compute_PGD_accuracy(model, loader, rank)
        else:
            print('error')

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
        loader = DataLoader(pool_dataset, batch_size=self.batch_size_uncertainty, sampler=sampler, num_workers=self.world_size, shuffle=False)

        model = self.load_model()
        model.load_state_dict(state_dict)
        model.to(rank)
        model.eval()  # Ensure the model is in evaluation mode
        model = DDP(model, device_ids=[rank])

        print('do the predictions')
        predictions = []
        indices_list = []
        with torch.no_grad():
            for inputs, _, indices in loader:
                inputs = inputs.cuda(rank)
                outputs = model(inputs) 
                predictions.append(outputs)
                indices_list.append( indices.cuda(rank) )
        print('prepare the predictions')
        predictions = torch.cat(predictions, dim=0)
        outputs = torch.softmax(predictions, dim=1)
        uncertainty = 1 - torch.max(outputs, dim=1)[0]
        gather_list = [torch.zeros_like(uncertainty) for _ in range(self.world_size)]
        
        indices = torch.cat(indices_list, dim=0)
        index_gather_list = [torch.zeros_like(indices) for _ in range(self.world_size)]
        print('gather the predictions')

        dist.all_gather(gather_list, uncertainty)
        dist.all_gather(index_gather_list, indices)

        dist.barrier()  # Synchronization point

        if rank == 0:
            all_uncertainties = torch.cat(gather_list, dim=0)
            all_indices = torch.cat(index_gather_list, dim=0)
            _, top_n_indices_sub = torch.topk(all_uncertainties, N, largest=True)
            top_n_indices.copy_(all_indices[top_n_indices_sub])

        cleanup()

    def random_sampling(self, args):
        pool_dataset, N = args

        # List of all dataset indices
        all_indices = list( range( len(pool_dataset) ) )
        
        # Check if the requested number of instances is greater than the dataset size
        if N > len(all_indices):
            print(f"Requested number of instances ({N}) exceeds the dataset size ({len(all_indices)}). Returning all available indices.")
            return all_indices
        
        # Randomly select 'n_instances' indices
        random_indices = random.sample(all_indices, N)
        return random_indices


    def update(self, rank, args): 

        state_dict, subset_dataset = args

        setup(self.world_size, rank)

        sampler = DistributedSampler(subset_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(subset_dataset, batch_size=self.batch_size_update, sampler=sampler, num_workers=self.world_size) #

        model = self.load_model()
        model.load_state_dict(state_dict)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        model.train()
        
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD( model.parameters(),lr=0.001, weight_decay=0.0001, momentum=0.9, nesterov=True, )
        
        for epoch in range(self.epochs):
            total_loss, total_err = 0.,0.
            sampler.set_epoch(epoch)
            for data, target, _ in loader:

                data, target = data.to(rank), target.to(rank)
                # print(data.shape, target.shape)
                optimizer.zero_grad()
                # logits = model(data)
                # loss = criterion(logits,target)
                logits, loss = trades.trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer,)
                print('loss', loss)
                loss.backward()
                optimizer.step()
                #total_err += (logits.max(dim=1)[1] != target).sum().item()
                #total_loss += loss.item() * data.shape[0]
            
            print(f'Rank {rank}, Epoch {epoch}, ') #Error {total_err / len(loader.dataset)}, Loss {total_loss / len(loader.dataset)}

        dist.barrier()  # Synchronization point
        if rank == 0:
            torch.save(model.state_dict(), "./state_dicts/{}_{}_{}_{}_{}_{}_{}.pt".format(self.data, self.model, self.active_strategy, self.n_rounds, self.size, self.epochs, self.seed) )

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

        pool_dataset, test_dataset = self.load_data()

        model = self.load_model()
        state_dict = model.state_dict()
        
        accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
        accuracy_tensor.share_memory_()  # 
        arg = (state_dict, test_dataset, accuracy_tensor, 'clean_accuracy')
        torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
        clean_acc = accuracy_tensor[0]
        print('clean_acc', clean_acc)
        result['init_clean_accuracy'] = clean_acc.item()
        
        card =  math.ceil( len( pool_dataset ) * self.size/100 )
        result['card'] = card
        round_size = math.ceil( card / self.n_rounds )
        total_indices = list( range( len( pool_dataset ) ) )
        collected_indices = []

        for i in range(self.n_rounds):

            print( 'iteration {}'.format(i) )

            if self.active_strategy == 'uncertainty':
                top_n_indices = torch.zeros(round_size, dtype=torch.long)  # Placeholder for the indices, shared memory
                top_n_indices.share_memory_()
                arg = (state_dict, pool_dataset, round_size, top_n_indices)
                torch.multiprocessing.spawn(self.uncertainty_sampling, args=(arg,), nprocs=self.world_size, join=True)
                collected_indices.extend( top_n_indices.tolist() )
            elif self.active_strategy == 'random':
                arg = (pool_dataset, round_size)
                n_indices = self.random_sampling(arg)
                collected_indices.extend( n_indices )            
            else:
                print('error')
            print(len(collected_indices))
            print('start update')

            subset_dataset = Subset(pool_dataset, collected_indices)
            
            # arg = (state_dict, subset_dataset)
            # self.update(arg)

            arg = (state_dict, subset_dataset)
            torch.multiprocessing.spawn(self.update,  args=(arg,),  nprocs=self.world_size, join=True)

            # Load the updated state_dict
            # state_dict = torch.load("./state_dicts/resnet50_imagenet1k_lora.pt")
            temp_state_dict = torch.load("./state_dicts/{}_{}_{}_{}_{}_{}_{}.pt".format(self.data, self.model, self.active_strategy, self.n_rounds, self.size, self.epochs, self.seed) )
            state_dict = {}
            for key, value in temp_state_dict.items():
                if key.startswith("module."):
                    state_dict[key[7:]] = value  # remove 'module.' prefix
                else:
                    state_dict[key] = value
            # model.load_state_dict(new_state_dict)

        accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
        accuracy_tensor.share_memory_()  # 
        arg = (state_dict, test_dataset, accuracy_tensor, 'clean_accuracy')
        torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
        clean_acc = accuracy_tensor[0]
        print('clean_acc', clean_acc)
        result['final_clean_accuracy'] = clean_acc.item()

        accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
        accuracy_tensor.share_memory_()  # 
        arg = (state_dict, test_dataset, accuracy_tensor, 'pgd_accuracy')
        torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
        pgd_acc = accuracy_tensor[0]
        print('pgd_acc', pgd_acc)
        result['final_PGD_accuracy'] = pgd_acc.item()

        print(result)
        print('finished')
        with gzip.open( './results/{}_{}_{}_{}_{}_{}_{}.pkl.gz'.format(self.data, self.model, self.active_strategy, n_rounds, size, nb_epochs, seed) ,'ab') as f:
                pkl.dump(result,f)
        print('saved')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_rounds", required=True, help="nb of active learning rounds")
    parser.add_argument("--nb_epochs", required=True, help="nb of Lora epochs")
    parser.add_argument("--size", required=True, help="wether to compute clean accuracy, PGD robustness or AA robustness")
    parser.add_argument("--active_strategy", required=True, help="which observation strategy to choose")
    parser.add_argument("--seed", required=True, help="the random seed")
    parser.add_argument("--data", required=True, help="the data used for the experiment")
    parser.add_argument("--model", required=True, help="the model used for the experiment")

    args = parser.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)

    n_rounds = int(args.n_rounds)
    size = float(args.size)
    nb_epochs = int(args.nb_epochs)
    seed = int(args.seed)
    active_strategy = args.active_strategy
    data = args.data
    model = args.model
    world_size = torch.cuda.device_count()
    utils.set_seeds(seed)
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