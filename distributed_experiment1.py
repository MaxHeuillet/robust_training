from comet_ml import Experiment

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
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

        print('set up the distributed setup,', rank, flush=True)
        self.setup.distributed_setup(self.world_size, rank)

        # dist.barrier()

        experiment = Experiment(api_key="I5AiXfuD0TVuSz5UOtujrUM9i",
                                project_name="robust_training",
                                workspace="maxheuillet",
                                auto_output_logging=False)
        
        experiment.log_parameter("run_id", os.getenv('SLURM_JOB_ID') )

        experiment.log_parameter("global_process_rank", rank)

        experiment.set_name( self.config_name )
        experiment.log_parameters(self.args)

        print('initialize dataset', rank,flush=True) 
        train_dataset = WeightedDataset(rank, self.args, train=True, prune_ratio = self.args.pruning_ratio, )
        val_dataset = IndexedDataset(self.args, train=False)  

        print('initialize sampler', rank,flush=True) 
        train_sampler = DistributedCustomSampler(self.args, train_dataset, num_replicas=self.world_size, rank=rank, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=rank, drop_last=False)
        
        print('initialize dataoader', rank,flush=True) 
        trainloader = DataLoader(train_dataset, batch_size=None, sampler=train_sampler, num_workers=self.world_size) #
        valloader = DataLoader(val_dataset, batch_size=256, sampler=val_sampler, num_workers=self.world_size)

        print('load model', rank,flush=True) 
        model, target_layers = load_architecture(self.args)
        
        print('load statedict', rank,flush=True) 
        statedict = load_statedict(self.args)

        model.load_state_dict(statedict)

        add_lora(target_layers, model)

        model.to(rank)
        model = DDP(model, device_ids=[rank])
        
        scaler = GradScaler()
        optimizer = torch.optim.SGD( model.parameters(),lr=self.args.init_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum, nesterov=True, )
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        self.validate(valloader, model, experiment, 0, rank)

        for iteration in range(self.args.iterations):

            model.train()
            train_sampler.set_epoch(iteration)

            for batch_id, batch in enumerate( trainloader ):

                data, target, idxs = batch

                data, target = data.to(rank), target.to(rank) 

                optimizer.zero_grad()

                with autocast():
                    
                    loss_values, clean_values, robust_values, logits_nat, logits_adv = get_loss(self.args, model, data, target, optimizer)

                train_dataset.update_scores(idxs, clean_values, robust_values, loss_values, logits_nat, logits_adv)
                
                loss = train_dataset.compute_loss(idxs, loss_values)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if self.args.sched == 'sched':
                scheduler.step()

            #### synchronize the different quantities between the different processes

            self.sync_updated_values(train_dataset, rank)


            #### compute TS update with master and synchronize the update.


            gradient_norm = compute_gradient_norms(model)
            current_lr = optimizer.param_groups[0]['lr']
            experiment.log_metric("iteration", iteration, epoch=iteration)
            experiment.log_metric("loss_value", loss, epoch=iteration)
            experiment.log_metric("lr_schedule", current_lr, epoch=iteration)
            experiment.log_metric("gradient_norm", gradient_norm, epoch=iteration)  

            print('start validation') 
            self.validate(valloader, model, experiment, iteration+1, rank)
  
            print(f'Rank {rank}, Iteration {iteration},', flush=True) 

        dist.barrier() 

        # if rank == 0:
            # torch.save(model.state_dict(), "./state_dicts/{}.pt".format(self.config_name) )
        
        self.final_validation(valloader, model, experiment, iteration, rank )
        
        experiment.end()

        print('clean up',flush=True)
        self.setup.cleanup()

    def sync_updated_values(self, train_dataset, rank):
        
        clean_scores = torch.tensor(train_dataset.clean_scores, device=rank)
        robust_scores = torch.tensor(train_dataset.robust_scores, device=rank)
        global_scores = torch.tensor(train_dataset.global_scores, device=rank)
        clean_pred = torch.tensor(train_dataset.clean_pred, device=rank)
        robust_pred = torch.tensor(train_dataset.robust_pred, device=rank)
        pulls = torch.tensor(train_dataset.pulls, device=rank)

        dist.all_reduce(clean_scores, op=dist.ReduceOp.SUM)
        dist.all_reduce(robust_scores, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_scores, op=dist.ReduceOp.SUM)
        dist.all_reduce(clean_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(robust_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(pulls, op=dist.ReduceOp.SUM)

        train_dataset.clean_scores = clean_scores.cpu()
        train_dataset.robust_scores = robust_scores.cpu()
        train_dataset.global_scores = global_scores.cpu()
        train_dataset.clean_pred = clean_pred.cpu()
        train_dataset.robust_pred = robust_pred.cpu()
        train_dataset.pulls = pulls.cpu()


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











   # def evaluation(self, rank, args):

    #     state_dict, test_dataset, accuracy_tensor, type = args

    #     setup.distributed_setup(self.world_size, rank)

    #     sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        
    #     model = self.load_model()

    #     model.load_state_dict(state_dict)

    #     model.load_state_dict(state_dict)
    #     model.to('cuda')

    #     model.to(rank)
    #     model.eval()  # Ensure the model is in evaluation mode
    #     model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    #     if type=='clean_accuracy':
    #         loader = DataLoader(test_dataset, batch_size=self.batch_size_cleanacc, sampler=sampler, num_workers=self.world_size)
    #         correct, total = utils.compute_clean_accuracy(model, loader, rank)
    #     elif type == 'pgd_accuracy':
    #         loader = DataLoader(test_dataset, batch_size=self.batch_size_pgdacc, sampler=sampler, num_workers=self.world_size)
    #         correct, total = utils.compute_PGD_accuracy(model, loader, rank)
    #     else:
    #         print('error')

    #     # Tensorize the correct and total for reduction
    #     correct_tensor = torch.tensor(correct).cuda(rank)
    #     total_tensor = torch.tensor(total).cuda(rank)
    #     print(correct_tensor, total_tensor)

    #     # Use dist.all_reduce to sum the values across all processes
    #     dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    #     print(correct_tensor, total_tensor)

    #     # Compute accuracy and store it in the shared tensor
    #     accuracy_tensor[rank] = correct_tensor.item() / total_tensor.item()

    #     print(accuracy_tensor)

    # def launch_evaluation(self,  ):

    #     result = self.conf.copy()

    #     pool_dataset, test_dataset = self.load_data()

    #     model = self.load_model()
    #     state_dict = model.state_dict()
        
    #     accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
    #     accuracy_tensor.share_memory_()  
    #     arg = (state_dict, test_dataset, accuracy_tensor, 'clean_accuracy')
    #     torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
    #     clean_acc = accuracy_tensor[0]
    #     print('clean_acc', clean_acc)
    #     result['init_clean_accuracy'] = clean_acc.item()
        
    #     card =  math.ceil( len( pool_dataset ) * self.size/100 )
    #     result['card'] = card

    #     # try: 
    #     temp_state_dict = torch.load("./state_dicts/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(self.loss, self.lr, self.sched, self.data, self.model, self.active_strategy, self.n_rounds, self.size, self.epochs, self.seed) )
    #     state_dict = {}
    #     for key, value in temp_state_dict.items():
    #         if key.startswith("module."):
    #             state_dict[key[7:]] = value  # remove 'module.' prefix
    #         else:
    #             state_dict[key] = value
    #         # model.load_state_dict(new_state_dict)

    #     accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
    #     accuracy_tensor.share_memory_()  
    #     arg = (state_dict, test_dataset, accuracy_tensor, 'clean_accuracy')
    #     torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
    #     clean_acc = accuracy_tensor[0]
    #     print('clean_acc', clean_acc)
    #     result['final_clean_accuracy'] = clean_acc.item()

    #     accuracy_tensor = torch.zeros(world_size, dtype=torch.float)  # Placeholder for the accuracy, shared memory
    #     accuracy_tensor.share_memory_()  
    #     arg = (state_dict, test_dataset, accuracy_tensor, 'pgd_accuracy')
    #     torch.multiprocessing.spawn(self.evaluation, args=(arg,), nprocs=world_size, join=True)
    #     pgd_acc = accuracy_tensor[0]
    #     print('pgd_acc', pgd_acc)
    #     result['final_PGD_accuracy'] = pgd_acc.item()
        
    #     print(result)
    #     print('finished')
    #     with gzip.open( './results/{}.pkl.gz'.format(self.config_name) ,'wb') as f:
    #         pkl.dump(result,f)
    #     print('saved')


    

