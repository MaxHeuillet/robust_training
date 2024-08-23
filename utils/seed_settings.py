import random
import numpy as np
import torch
import os



def set_seeds(seed):
    """
    Set the seed for reproducibility in all used libraries.
    
    Args:
    seed (int): The seed number to set.
    """
    # Setting the seed for various Python, Numpy and PyTorch operations
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # hf_set_seed(seed)
    
    # Setting the seed for the OS
    os.environ['PYTHONHASHSEED'] = str(seed)