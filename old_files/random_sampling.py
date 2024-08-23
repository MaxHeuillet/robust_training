# import random

# def random_sampling(args):
#     pool_dataset, N = args

#     # List of all dataset indices
#     all_indices = list( range( len(pool_dataset) ) )
    
#     # Check if the requested number of instances is greater than the dataset size
#     if N > len(all_indices):
#         print(f"Requested number of instances ({N}) exceeds the dataset size ({len(all_indices)}). Returning all available indices.")
#         return all_indices
    
#     # Randomly select 'n_instances' indices
#     random_indices = random.sample(all_indices, N)
#     return random_indices