from comet_ml import Experiment

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler,autocast

from torch.optim import AdamW

# from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import torch

import io


import math
from autoattack import AutoAttack

from torch.utils.data import DataLoader

import torch.distributed as dist

import sys
import hashlib
import json

from utils import Setup

from samplers import DistributedCustomSampler
from datasets import WeightedDataset, IndexedDataset, load_data
from architectures import load_architecture, CustomModel
from losses import get_loss, get_eval_loss
from utils import get_args, get_exp_name, set_seeds
from cosine import CosineLR
from torchvision.transforms import v2

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import io
import sys

def get_unique_id(args):

    args_string = json.dumps(args, sort_keys=True)
  
    unique_id = hashlib.md5(args_string.encode()).hexdigest()
    
    return unique_id


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
                                project_name="robust_training",
                                workspace="maxheuillet",
                                auto_metric_logging=False,
                                auto_output_logging=False)
        
        logger.log_parameter("run_id", os.getenv('SLURM_JOB_ID') )
        logger.log_parameter("global_process_rank", rank)
        logger.set_name( self.config_name )
        logger.log_parameters(self.args)

        return logger

    def initialize_loaders(self, rank):

        train_dataset, val_dataset, _, N, train_transform, transform = load_data(self.args) 
        
        train_dataset = WeightedDataset(self.args, train_dataset, train_transform, N, prune_ratio = self.args.pruning_ratio, )
        val_dataset = IndexedDataset(self.args, val_dataset, transform,  N,) 

        print('initialize sampler', rank,flush=True) 
        self.args.batch_size = self.setup.train_batch_size()
        train_sampler = DistributedCustomSampler(self.args, train_dataset, num_replicas=self.world_size, rank=rank, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=rank, drop_last=False)
        
        print('initialize dataoader', rank,flush=True) 
        trainloader = DataLoader(train_dataset, 
                                 batch_size=None, 
                                 sampler=train_sampler, 
                                 num_workers=3, 
                                 pin_memory=True) 
        
        valloader = DataLoader(val_dataset, 
                               batch_size=self.setup.train_batch_size(), #64
                               sampler=val_sampler, 
                               num_workers=3,
                               pin_memory=True)
        
        # original_data_size = len( train_dataset.global_indices )  # Size of original dataset
        # batch_size = self.args.batch_size             # Batch size for training
        # original_epochs = self.args.iterations        # Number of epochs for original dataset
        # pruned_data_size = train_sampler.pruner.N_tokeep #int(0.30 * original_data_size)
        # adjusted_epochs = adjust_epochs(original_data_size, pruned_data_size, batch_size, original_epochs)
        # experiment.log_parameter("adjusted_epochs", adjusted_epochs)
        
        return trainloader, valloader, train_sampler, val_sampler, N

    def training(self, rank):

        print('set up the distributed setup,', rank, flush=True)
        self.setup.distributed_setup(self.world_size, rank)
        
        logger = self.initialize_logger(rank)

        print('initialize dataset', rank,flush=True) 

        trainloader, valloader, train_sampler, val_sampler, N = self.initialize_loaders(rank)

        # print('N', self.args.N, flush=True)

        model = load_architecture(self.args, N, rank)
        model = CustomModel(self.args, model)
        model.set_fine_tuning_strategy()
        model.to(rank)
        model = DDP(model, device_ids=[rank])

        # torch.autograd.set_detect_anomaly(True)

        #self.validate(valloader, model, experiment, 0, rank)
        
        print('start the loop')
        self.fit(model, trainloader, train_sampler, N, logger, rank)

        dist.barrier() 

        if rank == 0:
            # Access the underlying model
            model_to_save = model.module
            # print(model_to_save)
            
            # Move the model to CPU
            model_to_save = model_to_save.cpu()
            # print(model_to_save)

            torch.save(model_to_save.state_dict(), './state_dicts/trained_model_{}.pt'.format(self.exp_id))
            print('Model saved by rank 0')

        dist.barrier()

        print('start the loop 3')

        logger.end()
        self.setup.cleanup() 

    def sync_value(self, value, nb_examples, rank):

        # Aggregate results across all processes
        value_tensor = torch.tensor([value], dtype=torch.float32, device=rank)
        nb_examples_tensor = torch.tensor([nb_examples], dtype=torch.float32, device=rank)

        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb_examples_tensor, op=dist.ReduceOp.SUM)

        # Compute global averages
        avg_value = value_tensor.item() / nb_examples_tensor.item()

        return avg_value, value_tensor.item(), nb_examples_tensor.item() 
        

    def fit(self, model, trainloader, train_sampler, N, logger, rank):

        # Gradient accumulation:
        effective_batch_size = 1024
        per_gpu_batch_size = self.args.batch_size  # Choose a batch size that fits in memory
        accumulation_steps = effective_batch_size // (self.world_size * per_gpu_batch_size)
        global_step = 0  # Track global iterations across accumulation steps
        print('effective batch size', effective_batch_size, 'per_gpu_batch_size', per_gpu_batch_size, 'accumulation steps', accumulation_steps)

        scaler = GradScaler()
        # print(self.args.init_lr, self.args.weight_decay, self.args.momentum) 
        # optimizer = torch.optim.SGD( model.parameters(),lr=self.args.init_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum, nesterov=True, )
        optimizer = torch.optim.AdamW( model.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay, )
        scheduler = CosineLR( optimizer, max_lr=self.args.init_lr, epochs=int(self.args.iterations) )

        cutmix = v2.CutMix(num_classes=N)
        mixup = v2.MixUp(num_classes=N)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        
        for iteration in range(self.args.iterations):

            model.train()
            train_sampler.set_epoch(iteration)
            optimizer.zero_grad()

            print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                # print(data.shape, target.shape)

                data, target_one_hot = cutmix_or_mixup(data, target)

                print(data.shape, target.shape)

                with torch.autocast(device_type='cuda'):
                    loss_values, logits = get_loss(self.args, model, data, target, optimizer)

                loss = loss_values.mean() #train_dataset.compute_loss(idxs, loss_values)
                loss = loss / accumulation_steps  # Scale the loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                if (batch_id + 1) % max(1,accumulation_steps) == 0 or (batch_id + 1) == len(trainloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # Clear gradients after optimizer step

                    # Log metrics
                    global_step += 1
                    
                    gradient_norm = compute_gradient_norms(model)
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.log_metric("global_step", global_step, epoch=iteration)
                    logger.log_metric("loss_value", loss.item() * accumulation_steps, epoch=iteration)
                    logger.log_metric("lr_schedule", current_lr, epoch=iteration)
                    logger.log_metric("gradient_norm", gradient_norm, epoch=iteration)

                # break

            model.module.update_fine_tuning_strategy(iteration)
                
            if self.args.sched == 'sched':
                scheduler.step()
  
            print(f'Rank {rank}, Iteration {iteration},', flush=True) 

    def evaluation(self, rank, result_queue):

        _, _, test_dataset, N, _, transform = load_data(self.args) 
        print('loaded data', flush=True) 

        test_dataset = IndexedDataset(self.args, test_dataset, transform, N,)
        print('indexed data', flush=True) 

        test_sampler = DistributedSampler(test_dataset, 
                                          num_replicas=self.world_size, 
                                          rank=rank, 
                                          shuffle=False,  # Disable shuffling
                                          drop_last=False)
        print('distributed sampled')
        
        testloader = DataLoader(test_dataset, 
                               batch_size=self.setup.test_batch_size(), #64
                               sampler=test_sampler, 
                               num_workers=3,
                               pin_memory=True)
        print('dataloader', flush=True) 
        
        # Check the number of samples assigned to this process
        num_samples = len(test_sampler)
        print(f"Rank {rank}: Number of samples = {num_samples}", flush=True) 
            
        # Re-instantiate the model
        model = load_architecture(self.args, N, rank)
        model = CustomModel(self.args, model)
        model.set_fine_tuning_strategy()
        trained_state_dict = torch.load('./state_dicts/trained_model_{}.pt'.format(self.exp_id), map_location='cpu')
        model.load_state_dict(trained_state_dict)
        model.to(rank)

        model.eval()

        print('start AA accuracy', flush=True) 
        stats, stats_nat, stats_adv = self.test(testloader, model, rank)
        print('end AA accuracy', flush=True) 

        statistics = { 'stats': stats, 'stats_nat': stats_nat, 'stats_adv': stats_adv,}
        
        # Put results into the queue
        result_queue.put((rank, statistics))
        print(f"Rank {rank}: Results sent to queue", flush=True)
        
    def test(self, testloader, model, rank):

        from utils import ActivationTracker, register_hooks, compute_stats

        tracker_nat = ActivationTracker()
        tracker_adv = ActivationTracker()
        model.current_tracker = 'nat'
        handles = register_hooks(model, tracker_nat, tracker_adv)

        def forward_pass(x):
            return model(x)
        
        adversary = AutoAttack(forward_pass, norm='Linf', eps=4/255, version='standard', verbose = False, device = rank)
        print('adversary instanciated', flush=True) 
        
        stats = {'nb_correct_nat': 0, 'nb_correct_adv': 0, 'nb_examples': 0}
        stats_nat = { "zero_count": 0, "dormant_count": 0, "overactive_count": 0, "total_neurons": 0 }
        stats_adv = { "zero_count": 0, "dormant_count": 0, "overactive_count": 0, "total_neurons": 0 }
        
        for _, batch in enumerate( testloader ):

            print('start batch iterations',rank, _, len(testloader), flush=True) 

            data, target, _ = batch

            data, target = data.to(rank), target.to(rank) 

            batch_size = data.size(0)

            x_adv = adversary.run_standard_evaluation(data, target, bs = batch_size )
            
            # print('Evaluate the model on natural data', rank, flush=True)
            model.current_tracker = 'nat'  # Set the tracker to natural data
            nat_outputs = model(data)
            _, preds_nat = torch.max(nat_outputs.data, 1)

            # print('Evaluate the model on adversarial examples', rank, flush=True)
            model.current_tracker = 'adv'  # Set the tracker to adversarial data
            adv_outputs = model(x_adv)
            _, preds_adv = torch.max(adv_outputs.data, 1)

            # print('Compute statitistics', rank, flush=True)

            stats['nb_correct_nat'] += (preds_nat == target).sum().item()
            stats['nb_correct_adv'] += (preds_adv == target).sum().item()
            stats['nb_examples'] += target.size(0)

            # Compute neuron statistics
            stats_nat = compute_stats(tracker_nat.activations)
            stats_adv = compute_stats(tracker_adv.activations)

            # Accumulate the statistics
            for key in ["zero_count", "dormant_count", "overactive_count", "total_neurons"]:
                stats_nat[key] += stats_nat[key]
                stats_adv[key] += stats_adv[key]

            # Clear activations for next batch
            tracker_nat.activations.clear()
            tracker_adv.activations.clear()

            # break

        # Remove hooks
        for handle in handles:
            handle.remove()

        return stats, stats_nat, stats_adv
    
    def launch_evaluation(self,):
        #Create a Queue to gather results
        result_queue = Queue()

        # Launch evaluation processes
        processes = []
        for rank in range(self.world_size):
            p = mp.Process(target=self.evaluation, args=(rank, result_queue) )
            p.start()
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
        statistics = self.setup.aggregate_results(all_statistics)

        self.setup.log_results(statistics)
        
        
if __name__ == "__main__":

    print('begining of the execution')

    import torch.multiprocessing as mp #import Queue
    from multiprocessing import Queue #, Process

    torch.multiprocessing.set_start_method("spawn", force=True)

    args = get_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)

    world_size = torch.cuda.device_count()
    
    set_seeds(args.seed)

    experiment = BaseExperiment(args, world_size)

    experiment.setup.pre_training_log()
    
    mp.spawn(experiment.training, nprocs=experiment.world_size, join=True)
    
    experiment.setup.post_training_log()
    
    experiment.launch_evaluation()