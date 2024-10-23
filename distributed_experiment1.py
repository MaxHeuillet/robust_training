from comet_ml import Experiment



import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler,autocast

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
from datasets import WeightedDataset, IndexedDataset, load_data
from architectures import load_architecture, add_lora, set_lora_gradients
from losses import get_loss, get_eval_loss
from utils import get_args, get_exp_name, set_seeds
from cosine import CosineLR

import numpy as np
import tqdm
import pandas as pd

def experiment_exists(df, current_experiment, key_columns, tol=1e-6):
    df_key = df[key_columns]
    current_experiment_key = {k: current_experiment[k] for k in key_columns}
    current_df_key = pd.DataFrame([current_experiment_key])

    matches = []
    for col in key_columns:
        if df_key[col].dtype.kind in 'fc':  # if column is float or complex
            match_series = pd.Series(np.isclose(df_key[col], current_df_key[col].iloc[0], atol=tol))
        else:
            match_series = df_key[col] == current_df_key[col].iloc[0]
        matches.append(match_series)

    exists = pd.concat(matches, axis=1).all(axis=1).any()

    return exists, pd.concat(matches, axis=1).all(axis=1)



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

def convert_syncbn_to_bn(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        if module.affine:
            module_output.weight = module.weight
            module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    else:
        for name, child in module.named_children():
            module_output.add_module(name, convert_syncbn_to_bn(child))
    return module_output

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

        train_dataset, val_dataset, test_dataset, N, transform = load_data(self.args) 
        
        train_dataset = WeightedDataset(self.args, train_dataset, transform, N, prune_ratio = self.args.pruning_ratio, )
        val_dataset = IndexedDataset(self.args, val_dataset, transform,  N,) 
        test_dataset = IndexedDataset(self.args, test_dataset, transform, N,)  

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
                               batch_size=64, 
                               sampler=val_sampler, 
                               num_workers=3,
                               pin_memory=True)


        original_data_size = len( train_dataset.global_indices )  # Size of original dataset
        batch_size = self.args.batch_size             # Batch size for training
        original_epochs = self.args.iterations        # Number of epochs for original dataset
        pruned_data_size = train_sampler.pruner.N_tokeep #int(0.30 * original_data_size)
        adjusted_epochs = adjust_epochs(original_data_size, pruned_data_size, batch_size, original_epochs)
        experiment.log_parameter("adjusted_epochs", adjusted_epochs)

        model = load_architecture(self.args)
        
        # if self.args.lora:
        #     add_lora(target_layers, model)
        #     set_lora_gradients(args, model, target_layers)

        model.to(rank)
        model = DDP(model, device_ids=[rank])

        # torch.autograd.set_detect_anomaly(True)

        scaler = GradScaler()
        print(self.args.init_lr, self.args.weight_decay, self.args.momentum) 
        optimizer = torch.optim.SGD( model.parameters(),lr=self.args.init_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum, nesterov=True, )
        scheduler = CosineLR( optimizer, max_lr=self.args.init_lr, epochs=int(self.args.iterations) )

        self.validate(valloader, model, experiment, 0, rank)
        print('start the loop')

        for iteration in range(adjusted_epochs):

            model.train()
            train_sampler.set_epoch(iteration)

            print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                # print('batch in gpu memory')

                optimizer.zero_grad()

                with torch.autocast(device_type='cuda'):
                    
                    loss_values, logits = get_loss(self.args, model, data, target, optimizer)

                # print('got losses')
                train_dataset.update_scores(rank, idxs, loss_values, logits)

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


            gradient_norm = compute_gradient_norms(model)
            current_lr = optimizer.param_groups[0]['lr']
            experiment.log_metric("iteration", iteration, epoch=iteration)
            experiment.log_metric("loss_value", loss, epoch=iteration)
            # experiment.log_metric("clean_value", clean_values.mean(), epoch=iteration)
            # experiment.log_metric("adv_value", robust_values.mean(), epoch=iteration)
            experiment.log_metric("lr_schedule", current_lr, epoch=iteration)
            experiment.log_metric("gradient_norm", gradient_norm, epoch=iteration)
            #experiment.log_metric("reward", loss_values.sum(), epoch=iteration)  

            if iteration % 5 == 0:
                print('start validation') 
                self.validate(valloader, model, experiment, iteration+1, rank)

            # if self.args.pruning_strategy in ['decay_based', 'decay_based_v2',  'decay_based_v3']:
            #     indices = train_sampler.process_indices
            #     train_dataset.decay_model.reset_counters()
            #     results = torch.tensor([ train_dataset.decay_model.fit_predict( train_dataset.global_scores2[idx], ) for idx in indices ])
            #     experiment.log_metric("solver_fails", train_dataset.decay_model.fail, epoch=iteration)
            #     experiment.log_metric("solver_total", train_dataset.decay_model.total, epoch=iteration)

            #     results = results.to(dtype=torch.float32)
            #     train_dataset.alphas[indices] = results[:,0]
            #     train_dataset.betas[indices] = results[:,1]
            #     train_dataset.cetas[indices] = results[:,2]
            #     train_dataset.pred_decay[indices] = results[:,4]
  
            print(f'Rank {rank}, Iteration {iteration},', flush=True) 

        dist.barrier() 

        # if rank == 0:
            # torch.save(model.state_dict(), "./state_dicts/{}.pt".format(self.config_name) )

        del trainloader, valloader
        del train_dataset, val_dataset
        del train_sampler, val_sampler

        print('final validation')
        self.final_validation(test_dataset, model, experiment, iteration, rank )
        
        ### log all the results of the experiment in a csv file:

        data_path = './results/results_{}.csv'.format(self.args.exp)
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            df = pd.DataFrame()

        current_experiment = vars(self.args)

        key_columns = list( current_experiment.keys() )

        # Check if the experiment exists and get the matching row index
        exists, match_mask = experiment_exists(df, current_experiment, key_columns, tol=1e-6)

        if not exists:
            # If the experiment doesn't exist, append it as a new row
            new_row = pd.DataFrame([current_experiment])
            new_row.to_csv(data_path, mode='a', header=not df.empty, index=False)
        else:
            # If the experiment exists, overwrite the existing row
            df.loc[match_mask, :] = pd.DataFrame([current_experiment])
            df.to_csv('results.csv', index=False)

        print(f"Experiment {'updated' if exists else 'added'} successfully.")


        experiment.end()

        print('clean up',flush=True)
        self.setup.cleanup()

    def final_validation(self, test_dataset, model, experiment, iteration, rank):

        if rank == 0:
            # Re-instantiate the model
            model_eval = load_architecture(self.args)
            model_eval.load_state_dict(model.module.state_dict())
            # Convert SyncBatchNorm to BatchNorm
            model_eval = convert_syncbn_to_bn(model_eval)
            model_eval.eval()
            model_eval = model_eval.to(rank)  # Ensure the model is on the correct device

            # Create DataLoader for the test_dataset
            testloader = DataLoader(
                test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
            )

            print('start AA accuracy')
            total_correct_nat, total_correct_adv, total_examples = self.compute_AA_accuracy(
                testloader, model_eval, rank
            )
            print('end AA accuracy')

            # Compute the metrics
            clean_accuracy = total_correct_nat / total_examples
            robust_accuracy = total_correct_adv / total_examples

            # Log the metrics
            experiment.log_metric("final_clean_accuracy", clean_accuracy, epoch=iteration)
            experiment.log_metric("final_robust_accuracy", robust_accuracy, epoch=iteration)
        else:
            # Other ranks do nothing
            pass

        # Do not use dist.barrier() here


    # def final_validation(self, test_dataset, model, experiment, iteration, rank ):

    #     model_eval = load_architecture(self.args).to(rank)
    #     model_eval.load_state_dict(model.module.state_dict())
    #     # Convert SyncBatchNorm to BatchNorm
    #     model_eval = convert_syncbn_to_bn(model_eval)

    #     model.eval()

    #     # Create a new testloader without DistributedSampler
    #     total_size = len(test_dataset)
    #     per_process_size = total_size // self.world_size

    #     # Calculate start and end indices for each process
    #     start_idx = rank * per_process_size
    #     # Ensure the last process gets any remaining data
    #     end_idx = (rank + 1) * per_process_size if rank != self.world_size - 1 else total_size

    #     # Subset the dataset
    #     subset_indices = list(range(start_idx, end_idx))
    #     subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
    #     print(rank, len(subset_indices) )

    #     # Create DataLoader for the subset
    #     testloader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    #     print('start AA accuracy')
    #     total_correct_nat, total_correct_adv, total_examples = self.compute_AA_accuracy(testloader, model, rank)
    #     print('end AA accuracy')

    #     dist.barrier() 
    #     clean_accuracy, robust_accuracy  = self.sync_final_result(total_correct_nat, total_correct_adv, total_examples, rank)
        
        
    #     experiment.log_metric("final_clean_accuracy", clean_accuracy, epoch=iteration)
    #     experiment.log_metric("final_robust_accuracy", robust_accuracy, epoch=iteration)

    # def sync_final_result(self, total_correct_nat, total_correct_adv, total_examples, rank):

    #     # Aggregate results across all processes
    #     total_correct_nat_tensor = torch.tensor([total_correct_nat], dtype=torch.float32, device=rank)
    #     total_correct_adv_tensor = torch.tensor([total_correct_adv], dtype=torch.float32, device=rank)
    #     total_examples_tensor = torch.tensor([total_examples], dtype=torch.float32, device=rank)

    #     dist.all_reduce(total_correct_nat_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_correct_adv_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)

    #     # Compute global averages
    #     clean_accuracy = total_correct_nat_tensor.item() / total_examples_tensor.item()
    #     robust_accuracy = total_correct_adv_tensor.item() / total_examples_tensor.item()

    #     return clean_accuracy, robust_accuracy

    def compute_AA_accuracy(self, testloader, model, rank):

        model.eval()
        model = model.to(rank)  # Ensure model is on the correct device

        def forward_pass(x):
            return model(x)
        
        norm = 'Linf'
        epsilon = 8/255
        adversary = AutoAttack(forward_pass, norm=norm, eps=epsilon, version='standard', device = rank)
        print('adversary instanciated')
        
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0
        
        for batch_id, batch in enumerate( testloader ):

            print('start batch iterations')

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
