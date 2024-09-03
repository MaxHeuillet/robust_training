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
        self.clean_scores = torch.normal(mean, std_dev, size=(self.K,)).cpu()
        self.robust_scores = torch.normal(mean, std_dev, size=(self.K,)).cpu()
        self.global_scores = torch.normal(mean, std_dev, size=(self.K,)).cpu()
        self.clean_pred = torch.zeros( self.K, self.N ).cpu() #.half()
        self.robust_pred = torch.zeros( self.K, self.N ).cpu() #.half() 

        ### arguments relative to Thomspon pruning (non-contextual):
        self.mu0 = torch.zeros(self.K).cpu()
        self.kappa0 = torch.ones(self.K).cpu()
        self.alpha0 = torch.ones(self.K).cpu()
        self.beta0 = torch.ones(self.K).cpu()

        self.pulls = torch.zeros(self.K).cpu() # number of pulls
        self.reward = torch.zeros(self.K).cpu()  # cumulative reward

        self.weights = torch.ones(self.K).cpu()
        self.num_pruned_samples = 0
        # self.cur_batch_index = None

        self.alpha = 1
        self.mu = torch.zeros(2048).cpu()
        self.Sigma_inv = torch.eye(2048).cpu() / self.alpha

        self.global_scores2 = torch.zeros( (self.args.iterations, self.K) ).cpu()

    def define_latent_features(self,features):
        self.latent = features
        
    def update_scores(self, rank, indices, clean_values, robust_values, global_values, clean_pred, robust_pred):

        # Reshape 1D tensors to 2D to concatenate along columns
        n = indices.shape[0]
        indices = indices.to(rank).view(n, 1)
        clean_loss_val = clean_values.detach().clone().to(rank).view(n, 1)
        robust_loss_val = robust_values.detach().clone().to(rank).view(n, 1)
        global_loss_val = global_values.detach().clone().to(rank).view(n, 1)

        clean_pred = clean_pred.detach().clone().to(rank)
        robust_pred = robust_pred.detach().clone().to(rank)

        # Check and print shapes for debugging
        print(f"indices shape: {indices.shape}")
        print(f"clean_loss_val shape: {clean_loss_val.shape}")
        print(f"robust_loss_val shape: {robust_loss_val.shape}")
        print(f"global_loss_val shape: {global_loss_val.shape}")
        print(f"clean_pred shape: {clean_pred.shape}")
        print(f"robust_pred shape: {robust_pred.shape}")
        
        if dist.is_available() and dist.is_initialized():
            iv = torch.cat([indices, 
                                clean_loss_val,
                                robust_loss_val,
                                global_loss_val,
                                clean_pred,
                                robust_pred ], dim=1)
            
            print('iv', iv.shape)
            
            iv_whole_group = concat_all_gather(iv, 0)
            print(iv_whole_group.shape)
            # Extract values correctly based on the concatenated structure
            indices = iv_whole_group[:, 0].long()  # Extract the indices column
            clean_loss_val = iv_whole_group[:, 1] #.view(-1, 1)
            robust_loss_val = iv_whole_group[:, 2]#.view(-1, 1)
            global_loss_val = iv_whole_group[:, 3]#.view(-1, 1)

            # Correct extraction for clean_pred and robust_pred based on column positions
            # Since each pred should have 10 columns, adjust indices accordingly
            clean_pred = iv_whole_group[:, 4:14]  # Extract next 10 columns for clean_pred
            robust_pred = iv_whole_group[:, 14:24]  # Extract the next 10 columns for robust_pred

            
        # Check and print shapes for debugging
        print(f"indices shape: {indices.shape}")
        print(f"clean_loss_val shape: {clean_loss_val.shape}")
        print(f"robust_loss_val shape: {robust_loss_val.shape}")
        print(f"global_loss_val shape: {global_loss_val.shape}")
        print(f"clean_pred shape: {clean_pred.shape}")
        print(f"robust_pred shape: {robust_pred.shape}")

        # Ensure `clean_pred` and `robust_pred` have the correct shape by adding an extra dimension
        # clean_pred = clean_pred.unsqueeze(dim=1)  
        # robust_pred = robust_pred.unsqueeze(dim=1)  
        
        # Update scores and predictions
        self.pulls[indices.cpu().long()] += 1
        self.clean_scores[indices.cpu().long()] = clean_loss_val.cpu()
        self.robust_scores[indices.cpu().long()] = robust_loss_val.cpu()
        self.global_scores[indices.cpu().long()] = global_loss_val.cpu()
        self.clean_pred[indices.cpu().long()] = clean_pred.cpu()
        self.robust_pred[indices.cpu().long()] = robust_pred.cpu()

    # def update_scores2(self, iteration, indices, global_values,):

    #     global_loss_val = global_values.detach().clone().cpu()
    #     self.global_scores2[iteration][indices.cpu().long()] = global_loss_val

    # def update_contextual_TS_parameters(self,):

    #     #### update regarding TS (contextual):

    #     non_zero_indices = torch.nonzero(self.pulls, as_tuple=False)

    #     x = self.latent[non_zero_indices].T

    #     # Compute the scalar factor for the Sherman-Morrison update
    #     Sigma_inv_x = torch.matmul(self.Sigma_inv, x)
    #     scaling_factor = 1 + torch.matmul(x.T, Sigma_inv_x)
    #     self.Sigma_inv = self.Sigma_inv - torch.matmul(Sigma_inv_x, Sigma_inv_x.T) / scaling_factor

    #     # Update the mean vector
    #     mu_dot_x = torch.matmul(self.mu, x.flatten())
    #     self.mu = self.mu + (self.reward - mu_dot_x) * Sigma_inv_x.flatten()


        
    def compute_loss(self, indices, loss_values):
        device = loss_values.device
        weights = self.weights[indices].to(device) #self.cur_batch_index
        # self.reset_cur_batch_index()
        loss_values.mul_(weights)
        return loss_values.mean() 
        
    def update_weights(self, indices):
        self.weights[indices] = 1 / self.keep_ratio

    def reset_weights(self):
        self.weights[:] = 1

    def get_weights(self, indexes):
        return self.weights[indexes]
