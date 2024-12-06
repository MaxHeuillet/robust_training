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
from utils import get_args2, get_exp_name, set_seeds, load_optimizer, get_state_dict_dir
from utils import ActivationTracker, register_hooks, compute_stats
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
            param_norm = p.grad.detach().data.norm(2).item()
            # if torch.isnan(param_norm):
            #     print(f"Gradient contains NaN : {p}")
            # elif torch.isinf(param_norm):
            #     print(f"Gradient contains Inf : {p}")
            total_norm += param_norm ** 2
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

        logger = None
        
        if rank == 0:
            logger = Experiment(api_key="I5AiXfuD0TVuSz5UOtujrUM9i",
                                    project_name=self.setup.project_name,
                                    workspace="maxheuillet",
                                    auto_metric_logging=False,
                                    auto_output_logging=False)
            
            logger.set_name( self.setup.exp_id )
            
            logger.log_parameter("run_id", os.getenv('SLURM_JOB_ID') )
            logger.log_parameter("global_process_rank", rank)
            logger.log_parameters(self.setup.config)

        return logger
        
    def initialize_loaders(self, config, rank):

        train_dataset, val_dataset, test_dataset, N, train_transform, _ = load_data(self.setup.hp_opt, config) 
        
        # train_dataset = IndexedDataset(train_dataset, train_transform, N, ) #prune_ratio = self.args.pruning_ratio, )
        # val_dataset = IndexedDataset(val_dataset, transform,  N,) 
        # test_dataset = IndexedDataset(test_dataset, transform,  N,) 

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
                               #sampler=test_sampler, 
                               num_workers=3,
                               pin_memory=True)
        
        return trainloader, valloader, testloader, train_sampler, val_sampler, test_sampler, N

    def training(self, update_config, rank=None ):

        if self.setup.hp_opt:
            config = OmegaConf.merge(self.setup.config, update_config)
            rank = train.get_context().get_world_rank()
            logger = None
            resources = session.get_trial_resources()
            print(f"Trial resource allocation: {resources}")
            
        else:
            config = OmegaConf.load( "./configs/HPO_{}.yaml".format(self.setup.exp_id) )
            self.setup.distributed_setup(rank)
            logger = self.initialize_logger(rank)

        print('initialize dataset', rank, flush=True) 
        print(config, rank, flush=True) 

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
        if not self.setup.hp_opt:
            self.validation( valloader, model, logger, 0, rank)
        
        print('start the loop')
        
        scheduler = CosineAnnealingLR( optimizer, T_max=config.epochs, eta_min=0 )

        self.fit(config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank, logger)

        dist.barrier() 

        if not self.setup.hp_opt and rank == 0:
            model_to_save = model.module
            model_to_save = model_to_save.cpu()
            torch.save(model_to_save.state_dict(), get_state_dict_dir(self.setup.hp_opt, config) +'trained_model_{}_{}.pt'.format(self.setup.project_name, self.setup.exp_id) )
            print('Model saved by rank 0')
            logger.end()
        
        self.setup.cleanup() 

    def hyperparameter_optimization(self, ):  

        self.setup.hp_opt = True 
        import logging

        ray.init(logging_level=logging.DEBUG)

        hp_search = Hp_opt()
        print('epochs in the HP function', self.setup.config.epochs)
        tuner = hp_search.get_tuner( self.setup.config.epochs, self.training )
        result_grid = tuner.fit()

        best_result = result_grid.get_best_result()
        print("Best hyperparameters found were: ", best_result.config)

        self.setup.hp_opt = False

        # Extract relevant data (configurations and metrics) from result_grid
        best_config = OmegaConf.merge(self.setup.config, best_result.config['train_loop_config']  )
        OmegaConf.save(best_config, './configs/HPO_{}_{}.yaml'.format(self.setup.project_name, self.setup.exp_id) )

        self.setup.log_results(hpo_results = result_grid)

        ray.shutdown()

                
    def fit(self, config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank, logger=None):

        # Gradient accumulation:
        effective_batch_size = 1024
        per_gpu_batch_size = self.setup.train_batch_size() # Choose a batch size that fits in memory
        accumulation_steps = effective_batch_size // (self.setup.world_size * per_gpu_batch_size)
        global_step = 0  # Track global iterations across accumulation steps
        print('effective batch size', effective_batch_size, 'per_gpu_batch_size', per_gpu_batch_size, 'accumulation steps', accumulation_steps)

        scaler = GradScaler()  
        
        model.train()

        print('epochs', config.epochs)
        step_nb = 0
        
        for iteration in range(1, config.epochs+1):

            train_sampler.set_epoch(iteration)
            val_sampler.set_epoch(iteration)

            print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target = batch

                data, target = data.to(rank), target.to(rank) 

                # print('get loss', rank, flush=True)

                # Use autocast for mixed precision during forward pass
                with autocast():
                    loss_values, logits = get_loss(self.setup, model, data, target)
                    loss = loss_values.mean()
                    loss = loss / accumulation_steps  # Scale the loss

                # print('scale loss', rank, flush=True)

                scaler.scale(loss).backward()
                
                global_step += 1

                if not self.setup.hp_opt and rank == 0:
                    metrics = { "global_step": global_step, "loss_value": loss.item() * accumulation_steps, }
                    logger.log_metrics(metrics, epoch=iteration, step=step_nb)

                if (batch_id + 1) % max(1, accumulation_steps) == 0 or (batch_id + 1) == len(trainloader):

                    if not self.setup.hp_opt and rank == 0:
                        # print('unscale', rank, flush=True)
                        scaler.unscale_(optimizer)
                        gradient_norm = compute_gradient_norms(model)
                        logger.log_metric("gradient_norm", float(gradient_norm), epoch=iteration)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Clear gradients after optimizer step
                
                break
            if self.setup.hp_opt:
                self.validation( valloader, model, logger, iteration, rank)
            elif not self.setup.hp_opt and iteration % 10 == 0:
                self.validation( valloader, model, logger, iteration, rank)
                
            if scheduler is not None: scheduler.step()

            print(f'Rank {rank}, Iteration {iteration},', flush=True) 

    def validation(self, valloader, model, logger, iteration, rank):

        total_loss, total_correct_nat, total_correct_adv, total_examples, stats_nat = self.validation_loop(valloader, model, rank)

        print(stats_nat)
        
        dist.barrier() 
        val_loss, _, _ = self.setup.sync_value(total_loss, total_examples, rank)
        nat_acc, _, _ = self.setup.sync_value(total_correct_nat, total_examples, rank)
        adv_acc, _, _ = self.setup.sync_value(total_correct_adv, total_examples, rank)

        zero, _, _ = self.setup.sync_value(stats_nat['zero_count'], stats_nat['total_neurons'], rank)
        dormant, _, _ = self.setup.sync_value(stats_nat['dormant_count'], stats_nat['total_neurons'], rank)
        overact, _, _ = self.setup.sync_value(stats_nat['overactive_count'], stats_nat['total_neurons'], rank)

        print(zero, dormant, overact)

        if self.setup.hp_opt:
            print('val loss', val_loss)
            session.report({"loss": val_loss})
        elif not self.setup.hp_opt and rank == 0:
            metrics = { "val_loss": val_loss, "val_nat_accuracy": nat_acc, "val_adv_accuracy": adv_acc,
                        "zero_count": zero, "dormant_count": dormant, "overactive_count": overact, }
            logger.log_metrics(metrics, epoch=iteration)

    def validation_loop(self, valloader, model, rank):

        model.eval()

        tracker_nat = ActivationTracker()
        handles = register_hooks(model, tracker_nat,)

        total_loss = 0.0
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0
        stats_nat = { "zero_count": 0, "dormant_count": 0, "overactive_count": 0, "total_neurons": 0 }

        for batch_id, batch in enumerate( valloader ):

            data, target = batch

            data, target = data.to(rank), target.to(rank) 
            # print('shape validation batch', data.shape)
                
            with torch.autocast(device_type='cuda'):
                loss_values, _, _, logits_nat, logits_adv = get_eval_loss(self.setup, model, data, target, )

            total_loss += loss_values.sum().item()

            preds_nat = torch.argmax(logits_nat, dim=1)  # Predicted classes for natural examples
            preds_adv = torch.argmax(logits_adv, dim=1)  # Predicted classes for adversarial examples

            # Accumulate correct predictions
            total_correct_nat += (preds_nat == target).sum().item()
            total_correct_adv += (preds_adv == target).sum().item()
            total_examples += target.size(0)

            # Compute neuron statistics
            res = compute_stats(tracker_nat.activations)

            print(stats_nat)

            # Accumulate the statistics
            for key in ["zero_count", "dormant_count", "overactive_count", "total_neurons"]:
                stats_nat[key] += res[key]

            break
        
        # Remove hooks
        for handle in handles:
            handle.remove()

        return total_loss, total_correct_nat, total_correct_adv, total_examples, stats_nat
    

    def test(self, rank, result_queue):

        config = OmegaConf.load("./configs/HPO_{}_{}.yaml".format(self.setup.project_name, self.setup.exp_id) )

        _, _, testloader, _, _, _, N = self.initialize_loaders(config, rank)
        # test_sampler.set_epoch(0)  
        print('dataloader', flush=True) 
        

        model = load_architecture(self.setup.hp_opt, config, N, )

        model = CustomModel(config, model, )
        # model.set_fine_tuning_strategy()
        trained_state_dict = torch.load('./state_dicts/trained_model_{}_{}.pt'.format(self.setup.project_name, self.setup.exp_id), map_location='cpu')
        model.load_state_dict(trained_state_dict)
        model.to(rank)

        model.eval()

        print('start AA accuracy', flush=True) 
        stats, stats_nat = self.test_loop(testloader, config, model, rank)
        print('end AA accuracy', flush=True) 

        statistics = { 'stats': stats, 'stats_nat': stats_nat, }
        
        # Put results into the queue
        result_queue.put((rank, statistics))
        print(f"Rank {rank}: Results sent to queue", flush=True)
        
    def test_loop(self, testloader, config, model, rank):

        tracker_nat = ActivationTracker()
        handles = register_hooks(model, tracker_nat,)

        def forward_pass(x):
            return model(x)
        
        device = torch.device(f"cuda:{rank}")
        adversary = AutoAttack(forward_pass, norm=config.distance, eps=config.epsilon, version='standard', verbose = False, device = device)
        print('adversary instanciated', flush=True) 
        
        stats = {'nb_correct_nat': 0, 'nb_correct_adv': 0, 'nb_examples': 0}
        stats_nat = { "zero_count": 0, "dormant_count": 0, "overactive_count": 0, "total_neurons": 0 }
        
        for _, batch in enumerate( testloader ):

            print('start batch iterations',rank, _, len(testloader), flush=True) 

            data, target = batch

            data, target = data.to(rank), target.to(rank) 

            batch_size = data.size(0)

            # print('prepare attack', flush=True)

            x_adv = adversary.run_standard_evaluation(data, target, bs = batch_size )

            # print('predict clean', flush=True)
            
            # print('Evaluate the model on natural data', rank, flush=True)
            nat_outputs = model(data)
            _, preds_nat = torch.max(nat_outputs.data, 1)

            # print('predict adv', flush=True)

            # print('Evaluate the model on adversarial examples', rank, flush=True)
            adv_outputs = model(x_adv)
            _, preds_adv = torch.max(adv_outputs.data, 1)

            # print('Compute statitistics', rank, flush=True)

            # print('update', flush=True)

            stats['nb_correct_nat'] += (preds_nat == target).sum().item()
            stats['nb_correct_adv'] += (preds_adv == target).sum().item()
            stats['nb_examples'] += target.size(0)

            # Compute neuron statistics
            res = compute_stats(tracker_nat.activations)

            # Accumulate the statistics
            for key in ["zero_count", "dormant_count", "overactive_count", "total_neurons"]:
                stats_nat[key] += res[key]

            # Clear activations for next batch
            tracker_nat.activations.clear()

            # if _ == 3:
            break

        # Remove hooks
        for handle in handles:
            handle.remove()

        return stats, stats_nat
    
    def launch_test(self):
        # import psutil  # Requires: pip install psutil
        # Create a Queue to gather results
        result_queue = Queue()

        # Define core groups (6 cores per process)
        # num_cores_per_process = 6
        # core_groups = [
        #     list(range(i * num_cores_per_process, (i + 1) * num_cores_per_process))
        #     for i in range(self.setup.world_size)
        # ]

        # Launch evaluation processes
        processes = []

        for rank in range(self.setup.world_size): # 

            p = mp.Process(target=self.test, args=(rank, result_queue))
            p.start()

            # Set the CPU affinity for the process
            # process = psutil.Process(p.pid)
            # process.cpu_affinity(core_groups[rank])

            # print(f"Process {p.pid} assigned to cores: {core_groups[rank]}", flush=True)
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Gather results from the Queue
        all_statistics = {}
        while not result_queue.empty():
            rank, stats = result_queue.get()
            all_statistics[rank] = stats

        print(all_statistics)

        # Log the aggregated results
        stats = self.setup.aggregate_results(all_statistics)

        self.setup.log_results(statistics=stats)

def training_wrapper(rank, experiment, config ):
    hpo_config = OmegaConf.load("./configs/HPO_{}_{}.yaml".format(experiment.setup.project_name, experiment.setup.exp_id) )
    # hpo_config = OmegaConf.merge(config, hpo_config)
    experiment.training(hpo_config, rank=rank)

if __name__ == "__main__":

    print('begining of the execution')

    torch.multiprocessing.set_start_method("spawn", force=True)

    # Initialize Hydra configuration
    initialize(config_path="configs", version_base=None)
    config = compose(config_name="default_config")  # Store Hydra config in a variable
    args_dict = get_args2()
    task = args_dict['task']
    del args_dict["task"]
    config = OmegaConf.merge(config, args_dict)
    
    set_seeds(config.seed)

    world_size = torch.cuda.device_count()
    setup = Setup(config, world_size)
    experiment = BaseExperiment(setup)

    # experiment.setup.pre_training_log()
    # if task == 'HPO':
    experiment.hyperparameter_optimization()
    # elif task == 'train':
    #     mp.spawn(training_wrapper, args=(experiment, config), nprocs=world_size, join=True)
    # elif task == 'test':
    #     experiment.launch_test()
    # elif task == 'dormant':
    #     experiment.launch_dormant()




    # def dormant(self, rank, result_queue):

    #     config = OmegaConf.load("./configs/HPO_{}.yaml".format(self.setup.exp_id) )

    #     _, _, _, _, _, _, N = self.initialize_loaders(config, rank)

    #     model = load_architecture(self.setup.hp_opt, config, N, )
    #     model = CustomModel(config, model, )
    #     trained_state_dict = torch.load('./state_dicts/trained_model_{}.pt'.format(self.setup.exp_id), map_location='cpu')
    #     model.load_state_dict(trained_state_dict)
    #     model.to(rank)

    #     config.dataset = 'Imagenette'
    #     _, _, testloader, _, _, test_sampler, _ = self.initialize_loaders(config, rank)
    #     test_sampler.set_epoch(0)    

    #     model.eval()

    #     stats_nat = self.dormant_loop(testloader, model, rank)

    #     result_queue.put( (rank, stats_nat) )
            
    # def dormant_loop(self, testloader, model, rank):

    #     from utils import ActivationTracker, register_hooks, compute_stats

    #     tracker_nat = ActivationTracker()
    #     tracker_adv = ActivationTracker()
    #     model.current_tracker = 'nat'
    #     handles = register_hooks(model, tracker_nat, tracker_adv)

    #     stats_nat = { "zero_count": 0, "dormant_count": 0, "overactive_count": 0, "total_neurons": 0 }

    #     for _, batch in enumerate( testloader ):

    #         print('start batch iterations',rank, _, len(testloader), flush=True) 

    #         data, target, _ = batch

    #         data, target = data.to(rank), target.to(rank) 

    #         # print('Evaluate the model on natural data', rank, flush=True)
    #         model.current_tracker = 'nat'  # Set the tracker to natural data
    #         _ = model(data)

    #         # Compute neuron statistics
    #         stats_nat = compute_stats(tracker_nat.activations)

    #         # Accumulate the statistics
    #         for key in ["zero_count", "dormant_count", "overactive_count", "total_neurons"]:
    #             stats_nat[key] += stats_nat[key]

    #         # Clear activations for next batch
    #         tracker_nat.activations.clear()

    #         # break

    #     # Remove hooks
    #     for handle in handles:
    #         handle.remove()

    #     return stats_nat
        
    # def launch_dormant(self,):
    #     #Create a Queue to gather results
    #     result_queue = Queue()

    #     # Launch evaluation processes
    #     processes = []
    #     for rank in range(self.setup.world_size):
    #         p = mp.Process(target=self.dormant, args=(rank, result_queue) )
    #         p.start()
    #         processes.append(p)

    #     # Wait for all processes to finish
    #     for p in processes:
    #         p.join()

    #     # Gather results from the Queue
    #     all_statistics = {}
    #     while not result_queue.empty():
    #         rank, stats = result_queue.get()
    #         all_statistics[rank] = stats

    #     # Log the aggregated results
    #     dorms = self.setup.aggregate_results2(all_statistics)

    #     self.setup.log_results(dormants = dorms)