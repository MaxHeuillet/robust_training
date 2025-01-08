import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from utils import ActivationTrackerAggregated, register_hooks_aggregated, compute_stats_aggregated
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

import timm.data.mixup as mixup

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
                               batch_size=int( 0.65 * self.setup.train_batch_size() ), 
                               sampler=val_sampler, 
                               num_workers=3,
                               pin_memory=True)
        
        testloader = DataLoader(test_dataset, 
                               batch_size=self.setup.test_batch_size(), 
                               sampler=test_sampler, 
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
            config = OmegaConf.load( "./configs/HPO_{}_{}.yaml".format(self.setup.project_name, self.setup.exp_id) )
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

        ray.init() #logging_level=logging.DEBUG

        hp_search = Hp_opt(self.setup)
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
        loss_scale = 0.50 if self.setup.config.loss_function == 'TRADES_v2' else 1.00
        per_gpu_batch_size = int( self.setup.train_batch_size() * loss_scale ) # Choose a batch size that fits in memory
        accumulation_steps = effective_batch_size // (self.setup.world_size * per_gpu_batch_size)
        global_step = 0  # Track global iterations across accumulation steps
        print('effective batch size', effective_batch_size, 'per_gpu_batch_size', per_gpu_batch_size, 'accumulation steps', accumulation_steps)

        scaler = GradScaler() 

        # mixup_fn = mixup.Mixup(mixup_alpha=0.2, cutmix_alpha=0, label_smoothing=0.1, num_classes=N )

        tracker_nat = ActivationTrackerAggregated(train=True)
        tracker_adv = ActivationTrackerAggregated(train=True)
        handles = register_hooks_aggregated(model, tracker_nat, tracker_adv)
         
        model.train()

        print('epochs', config.epochs)
        batch_step = 0
        update_step = 0
        
        for iteration in range(1, config.epochs+1):

            train_sampler.set_epoch(iteration)
            val_sampler.set_epoch(iteration)

            print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target = batch

                data, target = data.to(rank), target.to(rank) 

                # Use autocast for mixed precision during forward pass
                with autocast():
                    loss_values, logits = get_loss(self.setup, model, data, target)
                    loss = loss_values.mean()
                    loss = loss / accumulation_steps  # Scale the loss

                # print('scale loss', rank, flush=True)

                scaler.scale(loss).backward()
                
                global_step += 1

                if not self.setup.hp_opt and rank == 0:
                    metrics = { "global_step": global_step, 
                                "loss_value": loss.item() * accumulation_steps, }
                    logger.log_metrics(metrics, epoch=iteration, step = batch_step )

                batch_step += 1

                if (batch_id + 1) % max(1, accumulation_steps) == 0 or (batch_id + 1) == len(trainloader):

                    if not self.setup.hp_opt and rank == 0:
                        # print('unscale', rank, flush=True)
                        scaler.unscale_(optimizer)
                            
                        gradient_norm = compute_gradient_norms(model)

                        # print(tracker_nat.counts, data.shape)
                            
                        res_nat = compute_stats_aggregated(tracker_nat)
                        res_adv = compute_stats_aggregated(tracker_adv)

                        if self.setup.config.loss_function == 'TRADES_v2':

                            metrics = { "gradient_norm": float(gradient_norm),
                                        "zero_nat_train": res_nat['zero_count'] / res_nat['total_neurons'],
                                        "dormant_nat_train":res_nat['dormant_count'] / res_nat['total_neurons'], 
                                        "overactive_nat_train":res_nat['overactive_count'] / res_nat['total_neurons'], 
                                        "zero_adv_train": res_adv['zero_count'] / res_adv['total_neurons'],
                                        "dormant_adv_train":res_adv['dormant_count'] / res_adv['total_neurons'], 
                                        "overactive_adv_train":res_adv['overactive_count'] / res_adv['total_neurons'],  }
                        elif config.loss_function == 'CLASSIC_AT':
                            metrics = { "gradient_norm": float(gradient_norm), 
                                        "zero_adv_train": res_adv['zero_count'] / res_adv['total_neurons'],
                                        "dormant_adv_train":res_adv['dormant_count'] / res_adv['total_neurons'], 
                                        "overactive_adv_train":res_adv['overactive_count'] / res_adv['total_neurons'],  }
                            
                        logger.log_metrics(metrics, epoch=iteration, step=update_step, )

                        tracker_nat.clear()
                        tracker_adv.clear()

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Clear gradients after optimizer step

                    update_step += 1

                    
                # break
            if self.setup.hp_opt:
                self.validation( valloader, model, logger, iteration, rank)
            
            elif not self.setup.hp_opt and iteration in [10, 25, 40]:
                self.validation( valloader, model, logger, iteration, rank)
                
            if scheduler is not None: scheduler.step()

            print(f'Rank {rank}, Iteration {iteration},', flush=True) 
                            
        # Remove hooks
        for handle in handles:
            handle.remove()

    def validation(self, valloader, model, logger, iteration, rank):

        total_loss, total_correct_nat, total_correct_adv, total_examples, res_nat, res_adv = self.validation_loop(valloader, model, rank)

        print(res_nat, rank, flush=True)
        print(res_adv, rank, flush=True)
        
        dist.barrier() 
        val_loss, _, _ = self.setup.sync_value(total_loss, total_examples, rank)
        nat_acc, _, _ = self.setup.sync_value(total_correct_nat, total_examples, rank)
        adv_acc, _, _ = self.setup.sync_value(total_correct_adv, total_examples, rank)

        if self.setup.config.loss_function == 'TRADES_v2':
            nat_zero, _, _ = self.setup.sync_value(res_nat['zero_count'], res_nat['total_neurons'], rank)
            nat_dormant, _, _ = self.setup.sync_value(res_nat['dormant_count'], res_nat['total_neurons'], rank)
            nat_overact, _, _ = self.setup.sync_value(res_nat['overactive_count'], res_nat['total_neurons'], rank)

        adv_zero, _, _ = self.setup.sync_value(res_adv['zero_count'], res_adv['total_neurons'], rank)
        adv_dormant, _, _ = self.setup.sync_value(res_adv['dormant_count'], res_adv['total_neurons'], rank)
        adv_overact, _, _ = self.setup.sync_value(res_adv['overactive_count'], res_adv['total_neurons'], rank)

        if self.setup.hp_opt:
            print('val loss', val_loss)
            session.report({"loss": val_loss})

        elif not self.setup.hp_opt and rank == 0:

            print('hey', flush=True)

            if self.setup.config.loss_function == 'TRADES_v2':
                metrics = { "val_loss": val_loss, "val_nat_accuracy": nat_acc, "val_adv_accuracy": adv_acc,
                            "zero_nat_val": nat_zero, "dormant_nat_val": nat_dormant, "overactive_nat_val": nat_overact,
                            "zero_adv_val": adv_zero, "dormant_adv_val": adv_dormant, "overactive_adv_val": adv_overact, }
            else:
                metrics = { "val_loss": val_loss, "val_nat_accuracy": nat_acc, "val_adv_accuracy": adv_acc,
                            "zero_adv_val": adv_zero, "dormant_adv_val": adv_dormant, "overactive_adv_val": adv_overact, }
                
            logger.log_metrics(metrics, epoch=iteration)

    def validation_loop(self, valloader, model, rank):

        model.eval()

        tracker_nat = ActivationTrackerAggregated(train=False)
        tracker_adv = ActivationTrackerAggregated(train=False)

        handles = register_hooks_aggregated(model, tracker_nat, tracker_adv)

        total_loss = 0.0
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0
        # stats_nat = { "zero_count": 0, "dormant_count": 0, "overactive_count": 0, "total_neurons": 0 }
        # stats_adv = { "zero_count": 0, "dormant_count": 0, "overactive_count": 0, "total_neurons": 0 }

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

            # break

            # Compute neuron statistics
        res_nat = compute_stats_aggregated(tracker_nat)
        res_adv = compute_stats_aggregated(tracker_adv)

        tracker_nat.clear()
        tracker_adv.clear()   
        # Remove hooks
        for handle in handles:
            handle.remove()

        return total_loss, total_correct_nat, total_correct_adv, total_examples, res_nat, res_adv
    

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
        stats, res_nat, res_adv = self.test_loop(testloader, config, model, rank)
        print('end AA accuracy', flush=True) 

        statistics = { 'stats': stats, 'stats_nat': res_nat, 'stats_adv': res_adv, }
        print(statistics, flush=True)
        
        # Put results into the queue
        result_queue.put((rank, statistics))
        print(f"Rank {rank}: Results sent to queue", flush=True)
        
    def test_loop(self, testloader, config, model, rank):

        # tracker_nat = ActivationTrackerAggregated(train=False)
        # tracker_adv = ActivationTrackerAggregated(train=False)
        # handles = register_hooks_aggregated(model, tracker_nat, tracker_adv)

        def forward_pass(x):
            return model(x)
        
        device = torch.device(f"cuda:{rank}")
        adversary = AutoAttack(forward_pass, norm=config.distance, eps=config.epsilon, version='standard', verbose = False, device = device)
        print('adversary instanciated', flush=True) 
        
        nb_correct_nat = 0
        nb_correct_adv = 0
        nb_examples = 0

        print('stats', nb_correct_nat, nb_correct_adv, nb_examples, flush=True)

        for _, batch in enumerate( testloader ):

            x_nat, target = batch

            x_nat, target = x_nat.to(rank), target.to(rank) 

            batch_size = x_nat.size(0)

            print('start batch iterations', rank, _,batch_size, len(testloader), flush=True) 

            # print('prepare attack', flush=True)

            x_adv = adversary.run_standard_evaluation(x_nat, target, bs = batch_size )

            # print('predict clean', flush=True)

            # model.module.current_task = 'val_infer'
            logits_nat, logits_adv = model(x_nat, x_adv)
            # model.module.current_task = None

            preds_nat = torch.argmax(logits_nat, dim=1)
            preds_adv = torch.argmax(logits_adv, dim=1)

            # print('Compute statitistics', rank, flush=True)

            # print('update', flush=True)

            nb_correct_nat += (preds_nat == target).sum().item()
            nb_correct_adv += (preds_adv == target).sum().item()
            nb_examples += target.size(0)

            # break

        print('stats2', nb_correct_nat, nb_correct_adv, nb_examples, flush=True)

        stats = { 'nb_correct_nat':nb_correct_nat,
                  'nb_correct_adv':nb_correct_adv,
                  'nb_examples':nb_examples }
        
        print('stats dict, rank', stats, rank, flush=True)

        # res_nat = compute_stats_aggregated(tracker_nat)
        # res_adv = compute_stats_aggregated(tracker_adv)

        res_nat = {}
        res_adv = {}

        # tracker_nat.clear()
        # tracker_adv.clear()
        # Remove hooks
        # for handle in handles:
        #     handle.remove()

        return stats, res_nat, res_adv
    
    def launch_test(self):
        # import psutil  # Requires: pip install psutil
        # Create a Queue to gather results
        result_queue = Queue()


        # Launch evaluation processes
        processes = []

        for rank in range(self.setup.world_size): # 

            p = mp.Process(target=self.test, args=(rank, result_queue))
            p.start()

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

    print('begining of the execution', flush=True)

    torch.multiprocessing.set_start_method("spawn", force=True)

    # Initialize Hydra configuration
    initialize(config_path="configs", version_base=None)
    config = compose(config_name="default_config_bis")  # Store Hydra config in a variable
    args_dict = get_args2()
    task = args_dict['task']
    del args_dict["task"]
    config = OmegaConf.merge(config, args_dict)
    
    set_seeds(config.seed)

    world_size = torch.cuda.device_count()
    setup = Setup(config, world_size)
    experiment = BaseExperiment(setup)

    # if task == 'HPO':
    print('HPO', flush=True)
    experiment.hyperparameter_optimization()
    # elif task == 'train':
    
    print('train', flush=True)  
    mp.spawn(training_wrapper, args=(experiment, config), nprocs=world_size, join=True)
    
    # elif task == 'test':
    print('test', flush=True)
    experiment.launch_test()