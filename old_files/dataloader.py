# from torch.utils.data import DataLoader, DistributedSampler
# import random

# class IndexedDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1, sampler=None):
        
#         super().__init__(dataset, batch_size=batch_size, sampler=sampler)
        
#         self.num_batches = (len(dataset) + batch_size - 1) // batch_size

#     def get_batch(self, index):
#         if index < 0 or index >= self.num_batches:
#             raise IndexError("Batch index out of range")
        
#         if self.sampler is not None:
#             # Handle batch indices according to the sampler
#             batch_indices = list(self.sampler)[index * self.batch_size: (index + 1) * self.batch_size]
#         else:
#             start_idx = index * self.batch_size
#             end_idx = min((index + 1) * self.batch_size, len(self.dataset))
#             batch_indices = range(start_idx, end_idx)
        
#         batch = [self.dataset[i] for i in batch_indices]
        
#         if self.collate_fn:
#             batch = self.collate_fn(batch)
        
#         return batch
    
#     def __iter__(self):
#         self.current_index = 0
#         return self
    
#     def __next__(self):
#         if self.current_index >= self.num_batches:
#             raise StopIteration
        
#         batch = self.get_batch(self.current_index)
#         self.current_index += 1
#         return batch


# # from torch.utils.data import DataLoader, Dataset
# # import random

# # class IndexedDataLoader(DataLoader):
# #     def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None):
# #         super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        
# #         self.num_batches = (len(dataset) + batch_size - 1) // batch_size
# #         self.shuffle = shuffle
# #         self.indices = list(range(len(dataset)))
# #         if self.shuffle:
# #             self.shuffle_data()

# #     def shuffle_data(self):
# #         # Shuffle the dataset at the beginning of each epoch
# #         random.shuffle(self.indices)

# #     def get_batch(self, index):

# #         if index < 0 or index >= self.num_batches:
# #             raise IndexError("Batch index out of range")
                
# #         start_idx = index * self.batch_size
# #         end_idx = min((index + 1) * self.batch_size, len(self.dataset))
# #         batch_indices = self.indices[start_idx:end_idx]
# #         batch = [self.dataset[i] for i in batch_indices]
        
# #         if self.collate_fn:
# #             batch = self.collate_fn(batch)
        
# #         return batch
    
# #     def __iter__(self):
# #         # Shuffle the data at the beginning of each epoch if shuffle is True
# #         if self.shuffle:
# #             self.shuffle_data()
        
# #         self.current_index = 0
# #         return self
    
# #     def __next__(self):
# #         if self.current_index >= self.num_batches:
# #             raise StopIteration
        
# #         batch = self.get_batch(self.current_index)
# #         self.current_index += 1
# #         return batch


