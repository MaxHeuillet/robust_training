import math
import numpy as np
from typing import Iterator, Optional
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import torch.distributed as dist
import warnings
import random

from datasets.indexed_dataset import IndexedDataset



@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output


class WeightedDataset(IndexedDataset):
    """
    DataSchedule aims to achieve lossless training speed up by randomly prunes a portion of less informative samples
    based on the loss distribution and rescales the gradients of the remaining samples to approximate the original
    gradient. See https://arxiv.org/pdf/2303.04947.pdf

    .. note::.
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, args, train=True, prune_ratio: float = 0.5):

        super().__init__(args, train)

        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.K = len(self.dataset)

        
        # self.scores stores the loss value of each sample. Note that smaller value indicates the sample is better learned by the network.
        mean = 1.0
        std_dev = 0.1  
        self.clean_scores = torch.normal(mean, std_dev, size=(self.K,))
        self.robust_scores = torch.normal(mean, std_dev, size=(self.K,))
        self.global_scores = torch.normal(mean, std_dev, size=(self.K,))
        self.clean_pred = torch.ones( self.K, self.N ).half() * 3
        self.robust_pred = torch.ones( self.K, self.N ).half() * 3

        ### arguments relative to Thomspon pruning (non-contextual):
        self.mu0 = torch.zeros(self.K)
        self.kappa0 = torch.ones(self.K)
        self.alpha0 = torch.ones(self.K)
        self.beta0 = torch.ones(self.K)

        self.pulls = torch.zeros(self.K)  # number of pulls
        self.reward = torch.zeros(self.K)  # cumulative reward
        self.reward2 = torch.zeros(self.K)  # cumulative squared reward

        self.weights = torch.ones(len(self.dataset))
        self.num_pruned_samples = 0
        # self.cur_batch_index = None

        self.alpha = 1
        self.mu = torch.zeros(2048)
        self.Sigma_inv = torch.eye(2048) / self.alpha

    def define_latent_features(self,features):
        self.latent = features
        
    def update_scores(self, indices, clean_values, robust_values, global_values, clean_pred, robust_pred):

        #### update regarding score based pruning strategies:
        clean_loss_val = clean_values.detach().clone().cpu()
        self.clean_scores[indices.cpu().long()] = clean_loss_val

        robust_loss_val = robust_values.detach().clone().cpu()
        self.robust_scores[indices.cpu().long()] = robust_loss_val

        global_loss_val = global_values.detach().clone().cpu()
        self.global_scores[indices.cpu().long()] = global_loss_val

        clean_pred = clean_pred.detach().clone().cpu()
        self.clean_pred[indices.cpu().long()] = clean_pred

        robust_pred = robust_pred.detach().clone().cpu()
        self.robust_pred[indices.cpu().long()] = robust_pred

        self.pulls[indices.cpu().long()] += 1
        
        
    # def update_noncontextual_TS_parameters(self,):
    #### update regarding to TS (non contextual):
        # self.reward[indices.cpu().long()] += global_loss_val
        # self.reward2[indices.cpu().long()] += global_loss_val * global_loss_val
        # self.pulls[indices.cpu().long()] += 1

    #     pass
        

    # def update_contextual_TS_parameters(self,):

    #     #### update regarding TS (contextual):

    #     x = self.latent[indices].T

    #     # Compute the scalar factor for the Sherman-Morrison update
    #     Sigma_inv_x = torch.matmul(self.Sigma_inv, x)
    #     scaling_factor = 1 + torch.matmul(x.T, Sigma_inv_x)

    #     # Update the inverse covariance matrix using the Sherman-Morrison formula
    #     self.Sigma_inv = self.Sigma_inv - torch.matmul(Sigma_inv_x, Sigma_inv_x.T) / scaling_factor

    #     # Update the mean vector
    #     mu_dot_x = torch.matmul(self.mu, x.flatten())
    #     self.mu = self.mu + (global_loss_val - mu_dot_x) * Sigma_inv_x.flatten()


        
    def compute_loss(self, indices, loss_values):
        device = loss_values.device
        weights = self.weights[indices].to(device) #self.cur_batch_index
        # self.reset_cur_batch_index()
        loss_values.mul_(weights)
        return loss_values.mean() 
    
    # def reset_cur_batch_index(self):
    #     self.cur_batch_index = []
        
    # def __len__(self):
    #     return len(self.dataset)

    # def __getitem__(self, index):
    #     # self.cur_batch_index.append(index)
    #     return index, self.dataset[index] # , index
    #     # return self.dataset[index], index, self.scores[index]

    # def prune(self):
    #     # Prune samples that are well learned, rebalance the weight by scaling up remaining
    #     # well learned samples' learning rate to keep estimation about the same
    #     # for the next version, also consider new class balance

    #     well_learned_mask = ( (self.clean_scores < self.clean_scores.mean()) & (self.robust_scores < self.robust_scores.mean()) )
    #     well_learned_indices = np.where(well_learned_mask)[0]

    #     remained_indices = np.where(~well_learned_mask)[0].tolist()
    #     selected_indices = np.random.choice(well_learned_indices, int( self.keep_ratio * len(well_learned_indices)), replace=False )
    #     print('There are {} well learned samples and {} still to learn. We sampled {}'.format( sum(well_learned_mask), sum(~well_learned_mask), len(selected_indices) ) )

    #     # self.reset_weights()
    #     if len(selected_indices) > 0:
    #         self.update_weights(selected_indices)
    #         remained_indices.extend(selected_indices)
    #     self.num_pruned_samples += len(self.dataset) - len(remained_indices)
    #     np.random.shuffle(remained_indices)
    #     print('For next epoch, there will be {} samples. Total dataset size was {}'.format( len(remained_indices), len(self.dataset) ) )

    #     return remained_indices
    
    def update_weights(self, indices):
        self.weights[indices] = 1 / self.keep_ratio

    def reset_weights(self):
        self.weights[:] = 1

    def get_weights(self, indexes):
        return self.weights[indexes]

    # def get_pruned_count(self):
    #     return self.num_pruned_samples

    # def no_prune(self):
    #     samples_indices = list(range(len(self.dataset))) #list(range(len(self)))
    #     np.random.shuffle(samples_indices)
    #     return samples_indices

    # @property
    # def sampler(self):
    #     sampler = CustomSampler(self)
    #     # if dist.is_available() and dist.is_initialized():
    #     #     sampler = DistributedCustomSampler(sampler)
    #     return sampler

    # def mean_score(self):
    #     return self.scores.mean()

    # @property
    # def stop_prune(self):
    #     return self.num_epochs * self.delta

    