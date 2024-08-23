from comet_ml import Experiment

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch

import io

import gzip

import pickle as pkl

import math

import utils

from torch.utils.data import DataLoader

import torch.distributed as dist

import sys
import torch.nn as nn

from utils import Setup

from samplers import DistributedCustomSampler
from datasets import WeightedDataset
from architectures import load_architecture, load_statedict
from losses import get_loss
from utils import get_args, get_exp_name, set_seeds






# def setup(world_size, rank):
#   #Initialize the distributed environment.
#   print( ' world size {}, rank {}'.format(world_size,rank) )
#   print('set up the master adress and port')
#   os.environ['MASTER_ADDR'] = 'localhost'
#   os.environ['MASTER_PORT'] = '12354'
#   #Set environment variables for offline usage of Hugging Face libraries
#   os.environ['HF_DATASETS_OFFLINE'] = '1'
#   os.environ['TRANSFORMERS_OFFLINE'] = '1'
  
#   #Set up the local GPU for this process
#   torch.cuda.set_device(rank)
#   dist.init_process_group("nccl", rank=rank, world_size=world_size)
#   print('init process group ok')

# def cleanup():
#    dist.destroy_process_group()

def compute_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class BaseExperiment:

    def __init__(self, args, world_size, ):

        self.args = args
        self.config_name = get_exp_name(args)
        self.setup = Setup(args, self.config_name)
        self.world_size = world_size

    def training(self, rank, ): 

        print('set up the distributed setup,', rank)
        self.setup.distributed_setup(self.world_size, rank)

        dist.barrier()
        if rank == 0:
            print('initialize monitor', rank)
            self.setup.initialize_monitor()
            print('monitor initialized')

        dist.barrier()

        print('initialize dataset', rank) 
        train_dataset = WeightedDataset(self.args, train=True, prune_ratio = self.args.pruning_ratio, )

        print('initialize sampler', rank) 
        dist_sampler = DistributedCustomSampler(self.args, train_dataset, num_replicas=self.world_size, rank=rank, drop_last=True)
        
        print('initialize dataoader', rank) 
        trainloader = DataLoader(train_dataset, batch_size=None, sampler=dist_sampler, num_workers=self.world_size) #

        print('load model', rank) 
        model, target_layers = load_architecture(self.args)

        print('load statedict', rank) 
        statedict = load_statedict(self.args)

        model.load_state_dict(statedict)

        model.to(rank)
        model = DDP(model, device_ids=[rank])
        

        scaler = GradScaler()
        optimizer = torch.optim.SGD( model.parameters(),lr=self.args.init_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum, nesterov=True, )
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        
        for iteration in range(self.args.iterations):

            print('start iteration', iteration, rank) 


            model.train()
            dist_sampler.set_epoch(iteration)

            for batch_id, batch in enumerate( trainloader ):

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                optimizer.zero_grad()

                with autocast():
                    loss_values, clean_values, robust_values, logits_nat, logits_adv = get_loss(self.args, model, data, target, optimizer)

                train_dataset.update_scores(idxs, clean_values, robust_values, loss_values, logits_nat, logits_adv)
                loss = train_dataset.compute_weighted_loss(idxs, loss_values)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if self.args.sched == 'sched':
                scheduler.step()

            # if rank == 0:
            #     print('compute gradient norm', rank)
            #     gradient_norm = compute_gradient_norms(model)
            #     print('update monitor', rank)
            #     self.setup.update_monitor( iteration, optimizer, loss, gradient_norm )
                
            print(f'Rank {rank}, Iteration {iteration}, ') 

        dist.barrier() 

        if rank == 0:
            torch.save(model.state_dict(), "./state_dicts/{}.pt".format(self.config_name) )

            # self.setup.end_monitor()

        print('clean up')
        self.setup.cleanup()


    def evaluation(self, rank, args):

        state_dict, test_dataset, accuracy_tensor, type = args

        setup.distributed_setup(self.world_size, rank)

        sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        
        model = self.load_model()

        model.load_state_dict(state_dict)

        model.load_state_dict(state_dict)
        model.to('cuda')

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

    def launch_evaluation(self,  ):

        result = self.conf.copy()

        pool_dataset, test_dataset = self.load_data()

        model = self.load_model()
        state_dict = model.state_dict()
        
        accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
        accuracy_tensor.share_memory_()  
        arg = (state_dict, test_dataset, accuracy_tensor, 'clean_accuracy')
        torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
        clean_acc = accuracy_tensor[0]
        print('clean_acc', clean_acc)
        result['init_clean_accuracy'] = clean_acc.item()
        
        card =  math.ceil( len( pool_dataset ) * self.size/100 )
        result['card'] = card

        # try: 
        temp_state_dict = torch.load("./state_dicts/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(self.loss, self.lr, self.sched, self.data, self.model, self.active_strategy, self.n_rounds, self.size, self.epochs, self.seed) )
        state_dict = {}
        for key, value in temp_state_dict.items():
            if key.startswith("module."):
                state_dict[key[7:]] = value  # remove 'module.' prefix
            else:
                state_dict[key] = value
            # model.load_state_dict(new_state_dict)

        accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
        accuracy_tensor.share_memory_()  
        arg = (state_dict, test_dataset, accuracy_tensor, 'clean_accuracy')
        torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
        clean_acc = accuracy_tensor[0]
        print('clean_acc', clean_acc)
        result['final_clean_accuracy'] = clean_acc.item()

        accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
        accuracy_tensor.share_memory_()  
        arg = (state_dict, test_dataset, accuracy_tensor, 'pgd_accuracy')
        torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
        pgd_acc = accuracy_tensor[0]
        print('pgd_acc', pgd_acc)
        result['final_PGD_accuracy'] = pgd_acc.item()
        
        print(result)
        print('finished')
        with gzip.open( './results/{}.pkl.gz'.format(self.config_name) ,'wb') as f:
            pkl.dump(result,f)
        print('saved')



if __name__ == "__main__":

    print('begining of the execution')

    args = get_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)

    world_size = torch.cuda.device_count()
    
    set_seeds(args.seed)

    experiment = BaseExperiment(args, world_size)

    if args.task == "train":
        print('begin training')
        torch.multiprocessing.spawn(experiment.training,  nprocs=experiment.world_size, join=True)
    else:
        print('begin evaluation')
        experiment.launch_evaluation()

