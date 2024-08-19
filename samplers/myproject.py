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


# def info_hack_indices(self):
#     print('heyhey')
#     with torch.autograd.profiler.record_function(self._profile_name):
#         if self._sampler_iter is None:
#             # TODO(https://github.com/pytorch/pytorch/issues/76750)
#             self._reset()  # type: ignore[call-arg]
#         if isinstance(self._dataset, DataSchedule):
#             indices, data = self._next_data()
#             print('there',len(data))
#         else:
#             data = self._next_data()
#             print('here',len(data))
#         self._num_yielded += 1
#         if self._dataset_kind == _DatasetKind.Iterable and \
#                 self._IterableDataset_len_called is not None and \
#                 self._num_yielded > self._IterableDataset_len_called:
#             warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
#                         "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
#                                                                 self._num_yielded)
#             if self._num_workers > 0:
#                 warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
#                                 "IterableDataset replica at each worker. Please see "
#                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
#             warnings.warn(warn_msg)
#         if isinstance(self._dataset, DataSchedule):
#             self._dataset.set_active_indices(indices)
#         return data


# _BaseDataLoaderIter.__next__ = info_hack_indices


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


class DataSchedule(Dataset):
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

    def __init__(self, dataset: Dataset, epochs: int, prune_ratio: float = 0.5, delta: float = 0.875):

        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = epochs
        self.delta = delta
        
        # self.scores stores the loss value of each sample. Note that smaller value indicates the sample is better learned by the network.
        self.clean_scores = torch.ones(len(self.dataset)) * 3 #to generate randomly instead
        self.robust_scores = torch.ones(len(self.dataset)) * 3 #to generate randomly instad

        self.weights = torch.ones(len(self.dataset))

        self.num_pruned_samples = 0
        self.cur_batch_index = None

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices

    def update_scores(self, clean_values, robust_values):

        assert isinstance(clean_values, torch.Tensor)
        assert isinstance(robust_values, torch.Tensor)

        batch_size = clean_values.shape[0]
        assert len(self.cur_batch_index) == batch_size, 'not enough index'
        device = clean_values.device

        indices = self.cur_batch_index.to(device)

        clean_loss_val = clean_values.detach().clone()
        robust_loss_val = robust_values.detach().clone()
        
        if dist.is_available() and dist.is_initialized():
            idx_val = torch.cat([ indices.view(1, -1), clean_loss_val.view(1, -1), robust_loss_val.view(1, -1)], dim=0)
            
            idx_val_whole_group = concat_all_gather(idx_val, 1)
            
            indices = idx_val_whole_group[0]
            
            clean_loss_val = idx_val_whole_group[1]

            robust_loss_val = idx_val_whole_group[2]

        self.clean_scores[indices.cpu().long()] = clean_loss_val.cpu()
        self.robust_scores[indices.cpu().long()] = robust_loss_val.cpu()

    
    def update_loss_weights(self, loss_values):
        device = loss_values.device
        weights = self.weights[self.cur_batch_index].to(device)
        self.reset_cur_batch_index()
        loss_values.mul_(weights)
        return loss_values.mean() 
    
    def reset_cur_batch_index(self):
        self.cur_batch_index = []
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # self.cur_batch_index.append(index)
        return index, self.dataset[index] # , index
        # return self.dataset[index], index, self.scores[index]

    def prune(self):
        # Prune samples that are well learned, rebalance the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        well_learned_mask = ( (self.clean_scores < self.clean_scores.mean()) & (self.robust_scores < self.robust_scores.mean()) )
        well_learned_indices = np.where(well_learned_mask)[0]

        remained_indices = np.where(~well_learned_mask)[0].tolist()
        selected_indices = np.random.choice(well_learned_indices, int( self.keep_ratio * len(well_learned_indices)), replace=False )
        print('There are {} well learned samples and {} still to learn. We sampled {}'.format( sum(well_learned_mask), sum(~well_learned_mask), len(selected_indices) ) )

        self.reset_weights()
        if len(selected_indices) > 0:
            self.weights[selected_indices] = 1 / self.keep_ratio
            remained_indices.extend(selected_indices)
        self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        np.random.shuffle(remained_indices)
        print('For next epoch, there will be {} samples. Total dataset size was {}'.format( len(remained_indices), len(self.dataset) ) )

        
        return remained_indices

    @property
    def sampler(self):
        sampler = IBSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedIBSampler(sampler)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def reset_weights(self):
        self.weights[:] = 1


class IBSampler(object):
    def __init__(self, dataset: DataSchedule):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            # print('we are going to continue pruning, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            self.sample_indices = self.dataset.prune()
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        
        return next(self.iter_obj) # may raise StopIteration
        
    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self


class DistributedIBSampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """
    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: IBSampler):
            self.dataset = sampler
            # self.indices = None
 
        def reset(self, ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(self, dataset: IBSampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedIBSampler, self).__init__(sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        """
        Notes self.dataset is actually an instance of IBSampler rather than DataSchedule.
        """
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # print('distribute iter is called')
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        return self.iter_obj