from comet_ml import Experiment

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import torch.multiprocessing as mp 
from multiprocessing import Queue 
import os
import sys

import torch

import io

import math
from autoattack import AutoAttack

from torch.utils.data import DataLoader

import torch.distributed as dist

import hashlib
import json

from utils import Setup

# from samplers import DistributedCustomSampler
from torch.utils.data.distributed import DistributedSampler
from datasets import WeightedDataset, IndexedDataset, load_data
from architectures import load_architecture, CustomModel
from losses import get_loss, get_eval_loss
from utils import get_args2, get_exp_name, set_seeds, load_optimizer
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from ray.air import session

from hydra import initialize, compose
from omegaconf import OmegaConf

from omegaconf import DictConfig, OmegaConf
from utils import Hp_opt
from ray import train

import os
from ray import tune
import numpy as np

import ray
from ray import train, tune

def compute_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            if torch.isnan(param_norm):
                print(f"Gradient contains NaN : {p}")
            elif torch.isinf(param_norm):
                print(f"Gradient contains Inf : {p}")
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def check_for_nans(tensors, tensor_names):
    for tensor, name in zip(tensors, tensor_names):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaNs!")

class BaseExperiment:

    def __init__(self, setup, ):

        self.setup = setup

    def initialize_logger(self, rank):

        logger = Experiment(api_key="I5AiXfuD0TVuSz5UOtujrUM9i",
                                project_name="robust_training_{}_2".format(self.setup.cluster_name),
                                workspace="maxheuillet",
                                auto_metric_logging=False,
                                auto_output_logging=False)
        
        logger.log_parameter("run_id", os.getenv('SLURM_JOB_ID') )
        logger.log_parameter("global_process_rank", rank)
        logger.set_name( self.setup.exp_id )
        logger.log_parameters(self.setup.config)

        return logger
        
    def initialize_loaders(self, config, rank):

        train_dataset, val_dataset, test_dataset, N, train_transform, transform = load_data(self.setup.hp_opt, config) 
        
        train_dataset = IndexedDataset(train_dataset, train_transform, N, ) #prune_ratio = self.args.pruning_ratio, )
        val_dataset = IndexedDataset(val_dataset, transform,  N,) 
        test_dataset = IndexedDataset(test_dataset, transform,  N,) 

        # train_sampler = DistributedCustomSampler(self.args, train_dataset, num_replicas=self.world_size, rank=rank, drop_last=True)
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.setup.world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.setup.world_size, rank=rank, drop_last=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=self.setup.world_size, rank=rank, shuffle=True, drop_last=True)
        
        print('initialize dataoader', rank,flush=True) 
        trainloader = DataLoader(train_dataset, 
                                 batch_size=self.setup.train_batch_size(), 
                                 sampler=train_sampler, 
                                 num_workers=3, 
                                 pin_memory=True) 
        
        valloader = DataLoader(val_dataset, 
                               batch_size=int( 3/4 * self.setup.train_batch_size() ), 
                               sampler=val_sampler, 
                               num_workers=3,
                               pin_memory=True)
        
        testloader = DataLoader(test_dataset, 
                               batch_size=self.setup.test_batch_size(), 
                               sampler=test_sampler, 
                               num_workers=3,
                               pin_memory=True)
        
        return trainloader, valloader, testloader, train_sampler, val_sampler, test_sampler, N

    def training(self, update_config, ):

        if self.setup.hp_opt:
            config = OmegaConf.merge(self.setup.config, update_config)
            rank = train.get_context().get_world_rank()
            logger = None
            
        else:
            self.setup.distributed_setup()
            logger = self.initialize_logger(rank)

        rank = dist.get_rank()

        print('initialize dataset', rank, flush=True) 

        trainloader, valloader, _, train_sampler, val_sampler, _, N = self.initialize_loaders(config, rank)

        model = load_architecture(self.setup.hp_opt, config, N, )

        optimizer = load_optimizer(config, model,)   
        
        model = CustomModel(config, model, )
        # model.set_fine_tuning_strategy()
        # model._enable_all_gradients()
        model.to(rank)

        # for name, param in model.named_parameters():
        #     print(f"Parameter: {name}, strides: {param.data.stride()}")
        model = DDP(model, device_ids=[rank])

        # torch.autograd.set_detect_anomaly(True)

        #self.validation(valloader, model, experiment, 0, rank)
        
        print('start the loop')
        
        scheduler = CosineAnnealingLR( optimizer, T_max=config.epochs, eta_min=0 )

        self.fit(config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank, logger)

        dist.barrier() 

        if not self.setup.hp_opt and rank == 0:
            model_to_save = model.module
            model_to_save = model_to_save.cpu()
            torch.save(model_to_save.state_dict(), self.setup.get_statedictdir() +'trained_model_{}.pt'.format(self.setup.exp_id) )
            print('Model saved by rank 0')
            logger.end()
        
        self.setup.cleanup() 

    def hyperparameter_optimization(self, ):  

        self.setup.hp_opt = True 

        hp_search = Hp_opt()
        tuner = hp_search.get_tuner( self.training )
        result_grid = tuner.fit()

        best_result = result_grid.get_best_result()
        print("Best hyperparameters found were: ", best_result.config)

        self.setup.hp_opt = False

        # Extract relevant data (configurations and metrics) from result_grid
        extractable_data = [
            {
                "config": trial.config,  # Hyperparameters
                "metrics": trial.metrics,  # Performance metrics
            }
            for trial in result_grid
        ]

        final_data = { 'best_result':best_result,  'extractable_data':extractable_data } 
        
        import pickle
        # Serialize only the extracted data
        with open("./results/HPO/result_grid_{}.pkl".format(self.setup.exp_id), "wb") as f:
            pickle.dump(final_data, f)

                
    def fit(self, config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank, logger=None):

        # Gradient accumulation:
        effective_batch_size = 1024
        per_gpu_batch_size = self.setup.train_batch_size() # Choose a batch size that fits in memory
        accumulation_steps = effective_batch_size // (self.setup.world_size * per_gpu_batch_size)
        global_step = 0  # Track global iterations across accumulation steps
        print('effective batch size', effective_batch_size, 'per_gpu_batch_size', per_gpu_batch_size, 'accumulation steps', accumulation_steps)

        scaler = GradScaler()  
        
        model.train()
        
        for iteration in range(config.epochs):

            train_sampler.set_epoch(iteration)
            val_sampler.set_epoch(iteration)

            print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                loss_values, logits = get_loss(self.setup, model, data, target, )

                loss = loss_values.mean() #train_dataset.compute_loss(idxs, loss_values)
                loss = loss / accumulation_steps  # Scale the loss

                scaler.scale(loss).backward()
                
                global_step += 1

                if self.setup.hp_opt == False:
                    metrics = { "global_step": global_step, "loss_value": loss.item() * accumulation_steps, }
                    logger.log_metrics(metrics, epoch=iteration)

                if (batch_id + 1) % max(1, accumulation_steps) == 0 or (batch_id + 1) == len(trainloader):

                    if not self.setup.hp_opt:
                        scaler.unscale_(optimizer)
                        gradient_norm = compute_gradient_norms(model)
                        logger.log_metric("gradient_norm", gradient_norm, epoch=iteration)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Clear gradients after optimizer step
                
                break

            self.validation( valloader, model, logger, iteration+1, rank)
                
            # model.module.update_fine_tuning_strategy(iteration)
                
            if scheduler is not None: scheduler.step()

            print(f'Rank {rank}, Iteration {iteration},', flush=True) 

    def validation(self, valloader, model, logger, iteration, rank):

        total_loss, total_correct_nat, total_correct_adv, total_examples = self.validation_loop(valloader, model, rank)
        
        dist.barrier() 
        val_loss, _, _ = self.setup.sync_value(total_loss, total_examples, rank)
        nat_acc, _, _ = self.setup.sync_value(total_correct_nat, total_examples, rank)
        adv_acc, _, _ = self.setup.sync_value(total_correct_adv, total_examples, rank)

        if self.setup.hp_opt and rank == 0:
            print('val loss', val_loss)
            session.report({"loss": val_loss})
        else:
            metrics = { "val_loss": val_loss, "val_nat_accuracy": nat_acc, "val_adv_accuracy": adv_acc }
            logger.log_metrics(metrics, epoch=iteration)

    def validation_loop(self, valloader, model, rank):

        model.eval()

        total_loss = 0.0
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0

        for batch_id, batch in enumerate( valloader ):

            data, target, idxs = batch

            data, target = data.to(rank), target.to(rank) 
            # print('shape validation batch', data.shape)
                
            with torch.autocast(device_type='cuda'):
                loss_values, _, _, logits_nat, logits_adv = get_eval_loss(self.setup, model, data, target, )

            total_loss += loss_values.sum().item()
            # Compute predictions
            preds_nat = torch.argmax(logits_nat, dim=1)  # Predicted classes for natural examples
            preds_adv = torch.argmax(logits_adv, dim=1)  # Predicted classes for adversarial examples

            # Accumulate correct predictions
            total_correct_nat += (preds_nat == target).sum().item()
            total_correct_adv += (preds_adv == target).sum().item()
            total_examples += target.size(0)
        
            break

        return total_loss, total_correct_nat, total_correct_adv, total_examples

        

if __name__ == "__main__":

    print('begining of the execution')

    torch.multiprocessing.set_start_method("spawn", force=True)

    # Initialize Hydra configuration
    initialize(config_path="configs", version_base=None)
    config = compose(config_name="default_config")  # Store Hydra config in a variable
    args_dict = get_args2()
    config = OmegaConf.merge(config, args_dict)
    
    set_seeds(config.seed)

    world_size = torch.cuda.device_count()
    setup = Setup(config, world_size)
    experiment = BaseExperiment(setup)

    # experiment.setup.pre_training_log()

    # experiment.hyperparameter_optimization()
    
    mp.spawn(experiment.training, nprocs=experiment.world_size, join=True)
    
    # experiment.setup.post_training_log()
    
    # experiment.launch_test()



    # avg_loss, clean_accuracy, adv_acc  = self.sync_validation_results(total_loss, total_correct_nat, total_correct_adv, total_examples, rank)

    
    # def sync_validation_results(self, total_loss, total_correct_nat, total_correct_adv, total_examples, rank):

    #     # Aggregate results across all processes
    #     total_loss_tensor = torch.tensor([total_loss], dtype=torch.float32, device=rank)
    #     total_correct_nat_tensor = torch.tensor([total_correct_nat], dtype=torch.float32, device=rank)
    #     total_correct_adv_tensor = torch.tensor([total_correct_adv], dtype=torch.float32, device=rank)
    #     total_examples_tensor = torch.tensor([total_examples], dtype=torch.float32, device=rank)

    #     dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_correct_nat_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_correct_adv_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)

    #     # Compute global averages
    #     avg_loss = total_loss_tensor.item() / total_examples_tensor.item()
    #     clean_accuracy = total_correct_nat_tensor.item() / total_examples_tensor.item()
    #     robust_accuracy = total_correct_adv_tensor.item() / total_examples_tensor.item()

    #     return avg_loss, clean_accuracy, robust_accuracy