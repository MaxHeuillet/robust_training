from comet_ml import Experiment



import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
# from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import torch

import io

import gzip

import pickle as pkl

import math
from autoattack import AutoAttack
import utils

from torch.utils.data import DataLoader

import torch.distributed as dist

import sys
import torch.nn as nn

from utils import Setup

from samplers import DistributedCustomSampler
from datasets import WeightedDataset, IndexedDataset
from architectures import load_architecture, load_statedict, add_lora
from losses import get_loss, get_eval_loss
from utils import get_args, get_exp_name, set_seeds
from cosine import CosineLR

import numpy as np
import tqdm



def compute_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def check_for_nans(tensors, tensor_names):
    for tensor, name in zip(tensors, tensor_names):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaNs!")

def fit_one_curve(x):

    x = np.array(x, dtype=np.float32)

    if len(x) < 2:
        x = np.insert(x, 0, 5)  

    z = np.log(x)

    n = len(x)
    S_x = np.sum(x)
    S_z = np.sum(z)
    S_xz = np.sum(x * z)
    S_xx = np.sum(x ** 2)

    slope = (n * S_xz - S_x * S_z) / (n * S_xx - S_x ** 2)
    B = -slope  
    C = (S_z - slope * S_x) / n
    A = np.exp(C) 

    m = n+1
    next_value = A * np.exp(-B * m)  
    last_observed_value = x[-1]
    D = (last_observed_value - next_value) / last_observed_value 

    return (A, B, C, n, D)


class BaseExperiment:

    def __init__(self, args, world_size, ):

        self.args = args
        self.config_name = get_exp_name(args)
        self.setup = Setup(args, self.config_name)
        self.world_size = world_size

    def training(self, rank, ): 

        print('set up the distributed setup,', rank, flush=True)
        self.setup.distributed_setup(self.world_size, rank)

        # dist.barrier()

        experiment = Experiment(api_key="I5AiXfuD0TVuSz5UOtujrUM9i",
                                project_name="robust_training",
                                workspace="maxheuillet",
                                auto_metric_logging=False,
                                auto_output_logging=False)
        
        experiment.log_parameter("run_id", os.getenv('SLURM_JOB_ID') )

        experiment.log_parameter("global_process_rank", rank)

        experiment.set_name( self.config_name )
        experiment.log_parameters(self.args)

        print('initialize dataset', rank,flush=True) 
        train_dataset = WeightedDataset(self.args, train=True, prune_ratio = self.args.pruning_ratio, )
        val_dataset = IndexedDataset(self.args, train=False)  

        print('initialize sampler', rank,flush=True) 
        train_sampler = DistributedCustomSampler(self.args, train_dataset, num_replicas=self.world_size, rank=rank, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=rank, drop_last=False)
        
        print('initialize dataoader', rank,flush=True) 
        trainloader = DataLoader(train_dataset, 
                                 batch_size=None, 
                                 sampler=train_sampler, 
                                 num_workers=3, 
                                 pin_memory=True) 
        
        valloader = DataLoader(val_dataset, 
                               batch_size=256, 
                               sampler=val_sampler, 
                               num_workers=3,
                               pin_memory=True)

        model, target_layers = load_architecture(self.args)
        
        if self.args.pre_trained:
            statedict = load_statedict(self.args)
            model.load_state_dict(statedict)
        if self.args.lora:
            add_lora(target_layers, model)

        model.to(rank)
        model = DDP(model, device_ids=[rank])

        
        scaler = GradScaler()
        print(self.args.init_lr, self.args.weight_decay, self.args.momentum) 
        optimizer = torch.optim.SGD( model.parameters(),lr=self.args.init_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum, nesterov=True, )
        scheduler = CosineLR( optimizer, max_lr=self.args.init_lr, epochs=int(self.args.iterations) )

        print('here is it?')
        self.validate(valloader, model, experiment, 0, rank)
        print('start the loop')

        for iteration in range(self.args.iterations):

            model.train()
            train_sampler.set_epoch(iteration)

            print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                # print('batch in gpu memory')

                optimizer.zero_grad()

                with autocast():
                    
                    loss_values, clean_values, robust_values, logits_nat, logits_adv = get_loss(self.args, model, data, target, optimizer)

                # print('got losses')
                train_dataset.update_scores(rank, idxs, clean_values, robust_values, loss_values, logits_nat, logits_adv)

                # List of tensors and their names for easy reference in the check
                # tensors = [loss_values, clean_values, robust_values, logits_nat, logits_adv]
                # tensor_names = ['loss_values', 'clean_values', 'robust_values', 'logits_nat', 'logits_adv']

                # check_for_nans(tensors, tensor_names)

                # # Check for negative values and raise an error if found
                # if (loss_values < 0).any():
                #     raise ValueError("The tensor 'loss_values' contains negative values.")

                loss = train_dataset.compute_loss(idxs, loss_values)

                # loss.backward()
                # optimizer.step()
                # print('backward')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if self.args.sched == 'sched':
                scheduler.step()

            # print('before update',rank, train_dataset.Sigma_inv, train_dataset.mu)
            # train_dataset.update_contextual_TS_parameters()
            # print('after update',rank, train_dataset.Sigma_inv, train_dataset.mu)

            #### compute TS update with master and synchronize the update.

            gradient_norm = compute_gradient_norms(model)
            current_lr = optimizer.param_groups[0]['lr']
            experiment.log_metric("iteration", iteration, epoch=iteration)
            experiment.log_metric("loss_value", loss, epoch=iteration)
            experiment.log_metric("clean_value", clean_values.mean(), epoch=iteration)
            experiment.log_metric("adv_value", robust_values.mean(), epoch=iteration)
            experiment.log_metric("lr_schedule", current_lr, epoch=iteration)
            experiment.log_metric("gradient_norm", gradient_norm, epoch=iteration)
            experiment.log_metric("reward", loss_values.sum(), epoch=iteration)  

            print('start validation') 
            self.validate(valloader, model, experiment, iteration+1, rank)

            if self.args.pruning_strategy in ['decay_based', 'decay_based_v2',  'decay_based_v3']:
                indices = train_sampler.process_indices
                train_dataset.decay_model.reset_counters()
                results = torch.tensor([ train_dataset.decay_model.fit_predict( train_dataset.global_scores2[idx], ) for idx in indices ])
                experiment.log_metric("solver_fails", train_dataset.decay_model.fail, epoch=iteration)
                experiment.log_metric("solver_total", train_dataset.decay_model.total, epoch=iteration)

                results = results.to(dtype=torch.float32)
                train_dataset.alphas[indices] = results[:,0]
                train_dataset.betas[indices] = results[:,1]
                train_dataset.cetas[indices] = results[:,2]
                train_dataset.pred_decay[indices] = results[:,4]
  
            print(f'Rank {rank}, Iteration {iteration},', flush=True) 

        dist.barrier() 

        # if rank == 0:
            # torch.save(model.state_dict(), "./state_dicts/{}.pt".format(self.config_name) )
        
        self.final_validation(valloader, model, experiment, iteration, rank )
        
        experiment.end()

        print('clean up',flush=True)
        self.setup.cleanup()

    def final_validation(self, valloader, model, experiment, iteration, rank ):
        total_correct_nat, total_correct_adv, total_examples = self.compute_AA_accuracy(valloader, model, rank)
        dist.barrier() 
        clean_accuracy, robust_accuracy  = self.sync_final_result(total_correct_nat, total_correct_adv, total_examples, rank)
        experiment.log_metric("final_clean_accuracy", clean_accuracy, epoch=iteration)
        experiment.log_metric("final_robust_accuracy", robust_accuracy, epoch=iteration)

    def sync_final_result(self, total_correct_nat, total_correct_adv, total_examples, rank):

        # Aggregate results across all processes
        total_correct_nat_tensor = torch.tensor([total_correct_nat], dtype=torch.float32, device=rank)
        total_correct_adv_tensor = torch.tensor([total_correct_adv], dtype=torch.float32, device=rank)
        total_examples_tensor = torch.tensor([total_examples], dtype=torch.float32, device=rank)

        dist.all_reduce(total_correct_nat_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_adv_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)

        # Compute global averages
        clean_accuracy = total_correct_nat_tensor.item() / total_examples_tensor.item()
        robust_accuracy = total_correct_adv_tensor.item() / total_examples_tensor.item()

        return clean_accuracy, robust_accuracy

    def compute_AA_accuracy(self, valloader, model, rank):

        model.eval()

        def forward_pass(x):
            return model(x)
        
        norm = 'Linf'
        epsilon = 8/255
        adversary = AutoAttack(forward_pass, norm=norm, eps=epsilon, version='standard')
        
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0
        
        for batch_id, batch in enumerate( valloader ):

            data, target, idxs = batch

            data, target = data.to(rank), target.to(rank) 

            batch_size = data.size(0)

            x_adv = adversary.run_standard_evaluation(data, target, bs=batch_size )
            
            # Evaluate the model on adversarial examples
            adv_outputs = model(x_adv)
            _, preds_adv = torch.max(adv_outputs.data, 1)

            nat_outputs = model(data)
            _, preds_nat = torch.max(nat_outputs.data, 1)
            
            total_correct_nat += (preds_nat == target).sum().item()
            total_correct_adv += (preds_adv == target).sum().item()
            total_examples += target.size(0)

        return total_correct_nat, total_correct_adv, total_examples
    
    def validate(self, valloader, model, experiment, iteration, rank):
        total_loss, total_correct_nat, total_correct_adv, total_examples = self.validation_metrics(valloader, model, rank)
        dist.barrier() 
        avg_loss, clean_accuracy, robust_accuracy  = self.sync_validation_results(total_loss, total_correct_nat, total_correct_adv, total_examples, rank)
        experiment.log_metric("val_loss", avg_loss, epoch=iteration)
        experiment.log_metric("val_clean_accuracy", clean_accuracy, epoch=iteration)
        experiment.log_metric("val_robust_accuracy", robust_accuracy, epoch=iteration)
    
    def sync_validation_results(self, total_loss, total_correct_nat, total_correct_adv, total_examples, rank):

        # Aggregate results across all processes
        total_loss_tensor = torch.tensor([total_loss], dtype=torch.float32, device=rank)
        total_correct_nat_tensor = torch.tensor([total_correct_nat], dtype=torch.float32, device=rank)
        total_correct_adv_tensor = torch.tensor([total_correct_adv], dtype=torch.float32, device=rank)
        total_examples_tensor = torch.tensor([total_examples], dtype=torch.float32, device=rank)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_nat_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_adv_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)

        # Compute global averages
        avg_loss = total_loss_tensor.item() / total_examples_tensor.item()
        clean_accuracy = total_correct_nat_tensor.item() / total_examples_tensor.item()
        robust_accuracy = total_correct_adv_tensor.item() / total_examples_tensor.item()

        return avg_loss, clean_accuracy, robust_accuracy

    def validation_metrics(self, valloader, model, rank):

        model.eval()

        total_loss = 0.0
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0

        for batch_id, batch in enumerate( valloader ):

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                loss_values, _, _, logits_nat, logits_adv = get_eval_loss(self.args, model, data, target, )

                total_loss += loss_values.sum().item()
                # Compute predictions
                preds_nat = torch.argmax(logits_nat, dim=1)  # Predicted classes for natural examples
                preds_adv = torch.argmax(logits_adv, dim=1)  # Predicted classes for adversarial examples

                # Accumulate correct predictions
                total_correct_nat += (preds_nat == target).sum().item()
                total_correct_adv += (preds_adv == target).sum().item()
                total_examples += target.size(0)

        return total_loss, total_correct_nat, total_correct_adv, total_examples

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
