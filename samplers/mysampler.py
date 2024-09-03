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

from scipy.stats import truncnorm

def truncated_normal(mean, std, lower_bound=0, upper_bound=np.inf):
    mean = mean.numpy() # Ensure mean is an array
    std = std.numpy()    # Ensure std is an array
    # Calculate the bounds for the truncated normal distribution
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
    # Sample from the truncated normal distribution
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=mean.shape)

def check_for_nans(tensors, tensor_names):
    for tensor, name in zip(tensors, tensor_names):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaNs!")

class Pruner:

    ##note: it's important that the random selection is done with numpy

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.global_indices = dataset.global_indices
        self.N_tokeep = int( self.dataset.keep_ratio * len(self.global_indices) )
        print(self.N_tokeep, len(self.global_indices) )

    def linear_homoskedastic_thomspon_pruning(self,):

        Sigma = np.linalg.inv(self.Sigma_inv)

        theta_sampled = np.random.multivariate_normal(self.dataset.mu, Sigma)

        excpected_rewards = self.dataset.latent.dot(theta_sampled)
        
        truncated_rewards = np.maximum(0, excpected_rewards)

        sampling_probas = truncated_rewards / np.sum(truncated_rewards)

        indices = np.random.choice(self.global_indices, size=self.N_tokeep, replace=False, p=sampling_probas).tolist()

        return indices


    def thompson_pruning(self,):

        # posterior distribution
        mu = (self.dataset.kappa0 * self.dataset.mu0 + self.dataset.reward) / (self.dataset.kappa0 + self.dataset.pulls)
        kappa = np.maximum(self.dataset.kappa0 + self.dataset.pulls, 1e-3)
        alpha = self.dataset.alpha0 + self.dataset.pulls / 2
        mean_reward = self.dataset.reward / np.maximum(self.dataset.pulls, 1)
        
        beta = self.dataset.beta0 + \
        0.5 * (self.dataset.reward**2 - 2 * self.dataset.reward * mean_reward + self.dataset.pulls * np.square(mean_reward)) + \
        self.dataset.kappa0 * self.dataset.pulls * np.square(mean_reward - self.dataset.mu0) / (2 * kappa)

        # posterior sampling
        Lambda = np.maximum( np.random.gamma(alpha, 1.0 / beta, size=self.dataset.K), 1e-3 )

        # self.mu = mu + np.random.randn(self.dataset.K) / np.sqrt(kappa * Lambda)
        self.mu = truncated_normal(mu, np.sqrt(kappa * Lambda) )

        # arm = np.argmax(self.mu)
        sampling_probas = self.mu / np.sum(self.mu)
        
        indices = np.random.choice(self.global_indices, size=self.N_tokeep, replace=False, p=sampling_probas).tolist()

        return indices

    def random_pruning(self, ):
        # Implement random pruning
        
        # Check if the requested number of instances is greater than the dataset size
        if self.N_tokeep > len(self.global_indices):
            print(f"Requested number of instances ({self.N_tokeep}) exceeds the dataset size ({len(indices)}). Returning all available indices.")
            return self.global_indices
        
        # Randomly select 'n_instances' indices
        indices = np.random.choice(self.global_indices, self.N_tokeep, replace=False).tolist()  
        # np.random.shuffle(indices)
        return indices

    def uncertainty_based(self, ):

        proba_predictions = torch.softmax(self.dataset.clean_pred, dim=1)
        uncertainty = 1 - torch.max(proba_predictions, dim=1)[0]
        uncertainty = uncertainty.cpu().numpy()

        sampling_probas = uncertainty / np.sum(uncertainty)

        # List of tensors and their names for easy reference in the check
        tensors = [proba_predictions, uncertainty, sampling_probas]
        tensor_names = ['proba_predictions', 'uncertainty', 'sampling_probas',]

        check_for_nans(tensors, tensor_names)

        # Count non-zero elements
        non_zero_count = np.count_nonzero(sampling_probas)
        print("Number of non-zero elements:", non_zero_count)
        
        indices = np.random.choice(self.global_indices, size=self.N_tokeep, replace=False, p=sampling_probas).tolist()
        # np.random.shuffle(indices)
        return indices
    
    def loss_score_based(self,):

        if self.args.pruning_strategy == 'score_v1':
            # print( self.dataset.global_scores )
            scores = (self.dataset.global_scores - self.dataset.global_scores.min()) / (self.dataset.global_scores.max() - self.dataset.global_scores.min())

        elif self.args.pruning_strategy == 'score_v2':
            clean_scores = (self.dataset.clean_scores - self.dataset.clean_scores.min()) / (self.dataset.clean_scores.max() - self.dataset.clean_scores.min())
            robust_scores = (self.dataset.robust_scores - self.dataset.robust_scores.min()) / (self.dataset.robust_scores.max() - self.dataset.robust_scores.min())

            alpha = 0.5
            scores = alpha * clean_scores + (1 - alpha) * robust_scores
        else:
            print('score not implemented')

        scores = scores.cpu().numpy()
        sampling_probas = scores / np.sum(scores)

        indices = np.random.choice(self.global_indices, size=self.N_tokeep, replace=False, p=sampling_probas).tolist()
        np.random.shuffle(indices)
        return indices

    def no_pruning(self):
        indices = self.global_indices.copy()
        # np.random.shuffle(indices)
        return indices
    
    def prune(self):

        if self.args.pruning_ratio > 0:
            if self.args.pruning_strategy == 'random':
                return self.random_pruning()
            elif self.args.pruning_strategy in [ 'score_v1', 'score_v2' ]:
                return self.loss_score_based()
            elif self.args.pruning_strategy == 'uncertainty':
                return self.uncertainty_based()
            elif self.args.pruning_strategy == 'TS_pruning':
                return self.thompson_pruning()
            elif self.args.pruning_strategy == 'TS_context':
                return self.linear_homoskedastic_thomspon_pruning()
            else:
                raise ValueError(f"Undefined pruning strategy: {self.args.pruning_strategy}")
        else:
            return self.global_indices






class CustomSampler(object):
    
    def __init__(self, args, dataset, ):

        self.args = args
        self.pruner = Pruner(args, dataset)
        self.dataset = dataset
        self.stop_prune = self.max_prune()

        # self.sample_indices = None
        self.iter_obj = None

        self.post_pruning_indices = None
        self.process_indices = None

        self.sample_size = args.sample_size
        self.batch_size = args.batch_size

    def max_prune(self):
        return self.args.iterations * self.args.delta

    def __getitem__(self, idx):
        return self.tosample_indices[idx]

    def reset(self, iteration ):

        if iteration > self.stop_prune or self.stop_prune==0:
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            if iteration == self.stop_prune + 1:
                self.dataset.reset_weights()
            print('no pruning')
            indices = self.pruner.no_pruning()
        else:
            print('pruning')
            indices = self.pruner.prune()
        
        return indices

        

    def __next__(self):
        
        if len(self.process_indices) == 0:
            raise StopIteration
        
        # Select the batch
        batch_set = self.process_indices[:self.batch_size]
        
        # Remove the selected indices from the list
        self.process_indices = self.process_indices[self.batch_size:]
        
        if not batch_set:
            raise StopIteration
        
        # print(batch_set)

        return batch_set
    
    def set_seed(self,iteration):
        np.random.seed(iteration)

    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        return self
    
    def get_process_indices(self,indices):
        return indices
    
    def set_epoch(self, iteration):

        self.set_seed(iteration)
        # print('global', self.dataset.global_indices)
        self.post_pruning_indices = self.reset(iteration)
        # print('post_pruning', self.post_pruning_indices)
        self.process_indices = self.get_process_indices(self.post_pruning_indices)
        # print('process', self.process_indices)


    
class DistributedCustomSampler(CustomSampler):

    def __init__(
        self,
        args,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,) -> None:

        super( ).__init__(args, dataset)
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError( f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]" )
        
        self.num_replicas = num_replicas
        self.rank = rank

        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
  
        # print('num replicas', self.num_replicas)
        # print('num samples', self.num_samples)
        
    def update_num_samples(self,indices):
        if self.drop_last and len(indices) % self.num_replicas != 0:  
            self.num_samples = math.ceil( (len(indices) - self.num_replicas) / self.num_replicas    )
        else:
            self.num_samples = math.ceil(len(indices) / self.num_replicas)
        
        self.total_size = self.num_samples * self.num_replicas


    def get_process_indices(self,indices):

        self.update_num_samples(indices)

        if not self.drop_last:
            print('add extra samples')
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[ :padding_size ]
        else:
            print('remove tail')
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return indices
    



        
        



        # if len(self.tosample_indices) == 0:
            
        #     raise StopIteration
        
        # # print('hey2')
        
        # # Ensure sample size does not exceed the available to-sample indices
        # current_sample_size = min(self.sample_size, len(self.tosample_indices))
        # sampled_set = random.sample(self.tosample_indices, current_sample_size)

        # # Ensure batch size does not exceed the sampled set size
        # current_batch_size = min(self.batch_size, len(sampled_set))
        # batch_set = random.sample(sampled_set, current_batch_size)

        # if not batch_set:
        #     # print('hey3')
        #     raise StopIteration

        # # Update sampled and tosample indices
        # # self.sampled_indices.update(batch_set)
        # # self.tosample_indices.difference_update(batch_set)
        # self.tosample_indices = [index for index in self.tosample_indices if index not in batch_set]

        # self.dataset.set_active_indices(indices)


        # return batch_set

# import torch
# import numpy as np
# import random

# class DistributedCustomSampler(CustomSampler):

#     def __init__(self, args, sampler, dataset: Dataset, num_replicas: Optional[int] = None, 
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = False):

#         super().__init__(args, dataset,)

#         self.args = args

#         self.sampler = sampler

#         self.batch_size = args.batch_size
#         self.indices = []

#         print('num samples', self.num_samples)
#         print('num replicas', self.num_replicas)
#         # print(self.indices)

#         def set_epoch(self, epoch: int):
#             self.epoch = epoch
#             self.sampler.set_seed()
#             self.sampler.reset()

#         def __iter__(self) -> Iterator[int]:
#             if self.shuffle:
#                 # deterministically shuffle based on epoch and seed
#                 g = torch.Generator()
#                 g.manual_seed(self.seed + self.epoch)
#                 indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
#             else:
#                 indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

#             if not self.drop_last:
#                 # add extra samples to make it evenly divisible
#                 padding_size = self.total_size - len(indices)
#                 if padding_size <= len(indices):
#                     indices += indices[:padding_size]
#                 else:
#                     indices += (indices * math.ceil(padding_size / len(indices)))[
#                         :padding_size
#                     ]
#             else:
#                 # remove tail of data to make it evenly divisible.
#                 indices = indices[: self.total_size]
#             assert len(indices) == self.total_size

#             # subsample
#             indices = indices[self.rank : self.total_size : self.num_replicas]
#             print('sampler indices', indices)
#             assert len(indices) == self.num_samples

#             return iter(indices)

            






# class DistributedCustomSampler(DistributedSampler):
#     """
#     Wrapper over `Sampler` for distributed training.
#     Allows you to use any sampler in distributed mode.
#     It is especially useful in conjunction with
#     `torch.nn.parallel.DistributedDataParallel`. In such case, each
#     process can pass a DistributedSamplerWrapper instance as a DataLoader
#     sampler, and load a subset of subsampled data of the original dataset
#     that is exclusive to it.
#     .. note::
#         Sampler can change size during training.
#     """
#     class DatasetFromSampler(Dataset):
#         def __init__(self, sampler: CustomSampler):
#             self.dataset = sampler
#             # self.indices = None
 
#         def reset(self, ):
#             self.indices = None
#             self.dataset.reset()

#         def __len__(self):
#             return len(self.dataset)

#         def __getitem__(self, index: int):
#             """Gets element of the dataset.
#             Args:
#                 index: index of the element in the dataset
#             Returns:
#                 Single element by index
#             """
#             # if self.indices is None:
#             #    self.indices = list(self.dataset)
#             return self.dataset[index]

#     def __init__(self, dataset: CustomSampler, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = True) -> None:
        
#         sampler = self.DatasetFromSampler(dataset)
#         super(DistributedCustomSampler, self).__init__(sampler, num_replicas, rank, shuffle, seed, drop_last)
#         self.sampler = sampler
#         self.dataset = sampler.dataset.dataset # the real dataset.
#         self.iter_obj = None

#     def __iter__(self) -> Iterator[int]:
#         """
#         Notes self.dataset is actually an instance of IBSampler rather than DataSchedule.
#         """
#         self.sampler.reset()
#         if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
#             # Split to nearest available length that is evenly divisible.
#             # This is to ensure each rank receives the same amount of data when
#             # using this Sampler.
#             self.num_samples = math.ceil((len(self.sampler) - self.num_replicas) /   self.num_replicas   ) # type: ignore[arg-type]
#         else:
#             self.num_samples = math.ceil(len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
#         self.total_size = self.num_samples * self.num_replicas

#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             # type: ignore[arg-type]
#             indices = torch.randperm(len(self.sampler), generator=g).tolist()
#         else:
#             indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size /
#                             len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         # print('distribute iter is called')
#         self.iter_obj = iter(itemgetter(*indices)(self.sampler))
#         return self.iter_obj