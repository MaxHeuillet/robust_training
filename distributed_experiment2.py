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

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ray import tune
from ray.air.config import RunConfig
from ray.tune.tuner import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def get_unique_id(args):

    args_string = json.dumps(args, sort_keys=True)
  
    unique_id = hashlib.md5(args_string.encode()).hexdigest()
    
    return unique_id


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

def adjust_epochs(original_data_size, pruned_data_size, batch_size, original_epochs):

    # Calculate the total number of iterations with the original data
    original_iterations_per_epoch = math.ceil(original_data_size / batch_size)
    total_iterations_original = original_iterations_per_epoch * original_epochs

    # Calculate the number of iterations per epoch with the pruned data
    pruned_iterations_per_epoch = math.ceil(pruned_data_size / batch_size)

    # Calculate the number of epochs needed for pruned data to match total iterations
    adjusted_epochs = math.ceil(total_iterations_original / pruned_iterations_per_epoch)
    
    return adjusted_epochs


class BaseExperiment:

    def __init__(self, args, world_size, ):

        self.args = args
        self.config_name = get_exp_name(args)
        self.world_size = world_size
        self.current_experiment = vars(self.args)
        self.exp_id = get_unique_id(self.current_experiment)
        self.setup = Setup(args, self.config_name, self.exp_id, self.current_experiment)
        

    def initialize_logger(self, rank):

        logger = Experiment(api_key="I5AiXfuD0TVuSz5UOtujrUM9i",
                                project_name="robust_training_{}_2".format(self.setup.cluster_name),
                                workspace="maxheuillet",
                                auto_metric_logging=False,
                                auto_output_logging=False)
        
        logger.log_parameter("run_id", os.getenv('SLURM_JOB_ID') )
        logger.log_parameter("global_process_rank", rank)
        logger.set_name( self.config_name )
        logger.log_parameters(self.args)

        return logger
    

    def sync_value(self, value, nb_examples, rank):

        # Aggregate results across all processes
        value_tensor = torch.tensor([value], dtype=torch.float32, device=rank)
        nb_examples_tensor = torch.tensor([nb_examples], dtype=torch.float32, device=rank)

        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb_examples_tensor, op=dist.ReduceOp.SUM)

        # Compute global averages
        avg_value = value_tensor.item() / nb_examples_tensor.item()

        return avg_value, value_tensor.item(), nb_examples_tensor.item() 

    def initialize_loaders(self, rank):

        train_dataset, val_dataset, test_dataset, N, train_transform, transform = load_data(self.args) 
        
        train_dataset = IndexedDataset(self.args, train_dataset, train_transform, N, ) #prune_ratio = self.args.pruning_ratio, )
        val_dataset = IndexedDataset(self.args, val_dataset, transform,  N,) 
        test_dataset = IndexedDataset(self.args, test_dataset, transform,  N,) 

        print('initialize sampler', rank,flush=True) 
        self.args.batch_size = self.setup.train_batch_size()
        # train_sampler = DistributedCustomSampler(self.args, train_dataset, num_replicas=self.world_size, rank=rank, drop_last=True)
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=rank, drop_last=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=rank, shuffle=True, drop_last=True)
        
        print('initialize dataoader', rank,flush=True) 
        trainloader = DataLoader(train_dataset, 
                                 batch_size=self.args.batch_size, 
                                 sampler=train_sampler, 
                                 num_workers=3, 
                                 pin_memory=True) 
        
        valloader = DataLoader(val_dataset, 
                               batch_size=int( 3/4 * self.args.batch_size ), 
                               sampler=val_sampler, 
                               num_workers=3,
                               pin_memory=True)
        
        testloader = DataLoader(test_dataset, 
                               batch_size=self.args.batch_size, 
                               sampler=test_sampler, 
                               num_workers=3,
                               pin_memory=True)
        
        # original_data_size = len( train_dataset.global_indices )  # Size of original dataset
        # batch_size = self.args.batch_size             # Batch size for training
        # original_epochs = self.args.iterations        # Number of epochs for original dataset
        # pruned_data_size = train_sampler.pruner.N_tokeep #int(0.30 * original_data_size)
        # adjusted_epochs = adjust_epochs(original_data_size, pruned_data_size, batch_size, original_epochs)
        # experiment.log_parameter("adjusted_epochs", adjusted_epochs)
        
        return trainloader, valloader, testloader, train_sampler, val_sampler, test_sampler, N

    def training(self, config, ):
        from ray.train.torch import TorchTrainer, get_device, get_devices

        # device = get_device()
        # self.world_size = len( get_devices() )
        # rank = dist.get_rank()
        # print(rank)

        # self.setup.distributed_setup(world_size, rank)

        # Get rank and world size from Ray
        from ray import train
        rank = train.get_context().get_world_rank()
        world_size = train.get_context().get_world_size()
        device = train.torch.get_device()

        print('initialize dataset', rank, flush=True) 

        trainloader, valloader, _, train_sampler, val_sampler, _, N = self.initialize_loaders(rank)

        model = load_architecture(self.args, N, rank)
        optimizer = load_optimizer(self.args, config, model,)   
        
        model = CustomModel(self.args, model, )
        # model.set_fine_tuning_strategy()
        # model._enable_all_gradients()
        model.to(rank)

        # for name, param in model.named_parameters():
        #     print(f"Parameter: {name}, strides: {param.data.stride()}")
        model = DDP(model, device_ids=[rank])

        # torch.autograd.set_detect_anomaly(True)

        #self.validate(valloader, model, experiment, 0, rank)
        
        print('start the loop')
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR( optimizer, T_max=config['epochs'], eta_min=0 )

        self.fit(config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank,)

        dist.barrier() 

        # if rank == 0:
        #     # Access the underlying model
        #     model_to_save = model.module
        #     # print(model_to_save)
            
        #     # Move the model to CPU
        #     model_to_save = model_to_save.cpu()
        #     # print(model_to_save)

        #     torch.save(model_to_save.state_dict(), './state_dicts/trained_model_{}.pt'.format(self.exp_id))
        #     print('Model saved by rank 0')

        # dist.barrier()

        print('start the loop 3')

        # logger.end()
        self.setup.cleanup() 

    def hyperparameter_optimization(self, ):
        # Define the hyperparameter search space
        config = {
            "lr1": tune.loguniform(1e-5, 1e-1),
            "lr2": tune.loguniform(1e-5, 1e-1),
            "weight_decay": tune.loguniform(1e-6, 1e-2),
            "epochs": 25  # Fixed number of epochs for each trial
        }

        # Configure the scheduler WITHOUT metric and mode
        # scheduler = ASHAScheduler(
        #     max_t=10,
        #     grace_period=1,
        #     reduction_factor=2
        # )

        from ray.tune.search.bayesopt import BayesOptSearch
        # Set up the Bayesian optimization search algorithm
        scheduler = BayesOptSearch(
            metric="mean_loss",  # Metric to optimize
            mode="min",      ) # Optimization direction

        # Determine the number of workers and GPU usage
        num_gpus = torch.cuda.device_count()
        print(num_gpus)

        # Initialize the TorchTrainer
        # Initialize the TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=self.training,
            scaling_config=ScalingConfig(
                num_workers=1,  # Number of workers
                use_gpu=True,  # Use GPUs
                resources_per_worker={"CPU": 4, "GPU": 1},  # Resources per worker
            ),
        )
    
        # Set up the Tuner with metric and mode specified
        tuner = Tuner(
            trainer,
            param_space={"train_loop_config": config},
            tune_config=tune.TuneConfig(
                metric="loss",  # Specify the metric to optimize
                mode="min",     # Specify the optimization direction
                scheduler=scheduler,
                num_samples=10,
            ),
            run_config=RunConfig(
                name="training_experiment",
            ),
        )

        # Execute the hyperparameter search
        result_grid = tuner.fit()

        # Retrieve the best result
        best_result = result_grid.get_best_result()
        print("Best hyperparameters found were: ", best_result.config)
        
    def fit(self, config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank, logger=None):

        # Gradient accumulation:
        effective_batch_size = 1024
        per_gpu_batch_size = self.args.batch_size  # Choose a batch size that fits in memory
        accumulation_steps = effective_batch_size // (self.world_size * per_gpu_batch_size)
        global_step = 0  # Track global iterations across accumulation steps
        print('effective batch size', effective_batch_size, 'per_gpu_batch_size', per_gpu_batch_size, 'accumulation steps', accumulation_steps)

        scaler = GradScaler()  
        
        model.train()

        # cutmix = v2.CutMix(num_classes=N)
        # mixup = v2.MixUp(num_classes=N)
        # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        
        for iteration in range(config['epochs']):

            train_sampler.set_epoch(iteration)
            val_sampler.set_epoch(iteration)

            print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                loss_values, logits = get_loss(self.args, model, data, target, )

                loss = loss_values.mean() #train_dataset.compute_loss(idxs, loss_values)
                loss = loss / accumulation_steps  # Scale the loss

                scaler.scale(loss).backward()
                
                global_step += 1

                # print('gradient norm:', gradient_norm, 'rank:', rank, 'loss', loss)
                current_lr = optimizer.param_groups[0]['lr']
                if logger is not None:
                    logger.log_metric("global_step", global_step, epoch=iteration)
                    logger.log_metric("loss_value", loss.item() * accumulation_steps, epoch=iteration)
                    logger.log_metric("lr_schedule", current_lr, epoch=iteration)

                if (batch_id + 1) % max(1, accumulation_steps) == 0 or (batch_id + 1) == len(trainloader):

                    if logger is not None:
                        scaler.unscale_(optimizer)
                        gradient_norm = compute_gradient_norms(model)
                        logger.log_metric("gradient_norm", gradient_norm, epoch=iteration)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Clear gradients after optimizer step
                
                # break

            val_loss, _, _ = self.validate(valloader, model, logger, iteration+1, rank)

            if rank == 0:
                from ray.air import session
                session.report({"loss": val_loss})
                

            # model.module.update_fine_tuning_strategy(iteration)
                
            if scheduler is not None:
                scheduler.step()
                for i, param_group in enumerate(optimizer.param_groups):
                    print(f"Group {i}: Learning Rate = {param_group['lr']}, Parameters Count = {len(param_group['params'])}")

  
            print(f'Rank {rank}, Iteration {iteration},', flush=True) 



            


    def validate(self, valloader, model, logger, iteration, rank):
        total_loss, total_correct_nat, total_correct_adv, total_examples = self.validation_metrics(valloader, model, rank)
        dist.barrier() 
        avg_loss, clean_accuracy, robust_accuracy  = self.sync_validation_results(total_loss, total_correct_nat, total_correct_adv, total_examples, rank)
        if logger is not None:
            logger.log_metric("val_loss", avg_loss, epoch=iteration)
            logger.log_metric("val_clean_accuracy", clean_accuracy, epoch=iteration)
            logger.log_metric("val_robust_accuracy", robust_accuracy, epoch=iteration)
        return avg_loss, clean_accuracy, robust_accuracy
    
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
            # print('shape validation batch', data.shape)
                
            with torch.autocast(device_type='cuda'):
                loss_values, _, _, logits_nat, logits_adv = get_eval_loss(self.args, model, data, target, )

            total_loss += loss_values.sum().item()
            # Compute predictions
            preds_nat = torch.argmax(logits_nat, dim=1)  # Predicted classes for natural examples
            preds_adv = torch.argmax(logits_adv, dim=1)  # Predicted classes for adversarial examples

            # Accumulate correct predictions
            total_correct_nat += (preds_nat == target).sum().item()
            total_correct_adv += (preds_adv == target).sum().item()
            total_examples += target.size(0)
        
            # break

        return total_loss, total_correct_nat, total_correct_adv, total_examples

        
        
if __name__ == "__main__":

    print('begining of the execution')

    torch.multiprocessing.set_start_method("spawn", force=True)

    args = get_args2()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)

    world_size = torch.cuda.device_count()
    
    set_seeds(args.seed)

    experiment = BaseExperiment(args, world_size)

    # experiment.setup.pre_training_log()

    experiment.hyperparameter_optimization()
    
    # mp.spawn(experiment.training, nprocs=experiment.world_size, join=True)
    
    # experiment.setup.post_training_log()
    
    # experiment.launch_test()