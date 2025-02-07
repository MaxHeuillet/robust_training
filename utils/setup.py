
# from comet_ml import Experiment
# import cloudpickle as pickle


import os
import torch
import torch.distributed as dist
from filelock import FileLock
import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf
import hashlib
import datetime

def generate_timestamp():
    return datetime.now().strftime('%y/%m/%d/%H/%M/%S')

def check_unique_id(df1, df2, unique_id_col):

    unique_id = df2[unique_id_col].iloc[0]
    
    matching_indices = df1[df1[unique_id_col] == unique_id].index

    if not matching_indices.empty:
        iloc_indices = [df1.index.get_loc(idx) for idx in matching_indices]
        return True, iloc_indices
    else:
        return False, []
    
def get_config_id(cfg) -> str:

    # Convert the Hydra config to a dictionary and ensure ordering
    # config_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    
    # Get sorted list of values
    # sorted_items = sorted(config_dict.items())
    # values_list = [str(value) for key, value in sorted_items]
    
    # Join the values into a string
    serialized_values = cfg.backbone + '_' + cfg.dataset + '_' + cfg.loss_function
    print('serialized_values', serialized_values)
    
    return serialized_values

class Setup:

    def __init__(self, config, world_size):
        self.config = config
        self.exp_id = get_config_id(self.config)
        self.world_size = world_size
        self.hp_opt = False
        self.cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        self.project_name = config.project_name
        
    def distributed_setup(self, rank):

        # os.environ['NCCL_DEBUG'] = 'INFO'  # or 'TRACE' for more detailed logs
        # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        # os.environ['NCCL_BLOCKING_WAIT'] = '1'

        print('torch', torch.__version__)
        print('cuda', torch.version.cuda)
        print('cudnn', torch.backends.cudnn.version())
        
        #Initialize the distributed environment.
        print('set up the master adress and port')
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        #Set environment variables for offline usage of Hugging Face libraries
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
        
        #Set up the local GPU for this process
        # dist.init_process_group("nccl", )
        print('init process group ok')

        return rank
        
            
    def cleanup(self,):
        dist.destroy_process_group()

    def sync_value(self, value, nb_examples, rank):

        # Aggregate results across all processes
        value_tensor = torch.tensor([value], dtype=torch.float32, device=rank)
        nb_examples_tensor = torch.tensor([nb_examples], dtype=torch.float32, device=rank)

        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb_examples_tensor, op=dist.ReduceOp.SUM)

        # Compute global averages
        avg_value = value_tensor.item() / nb_examples_tensor.item()

        return avg_value, value_tensor.item(), nb_examples_tensor.item() 
    

    def train_batch_size(self): #(arch: str, dataset: str, loss_fn: str) -> int

        # -------------------------
        # 1) BASELINES PER ARCH
        # -------------------------

        if 'convnext_tiny' in self.config.backbone:
            base_bs = 64 #124
        elif 'convnext_base' in self.config.backbone:
            base_bs = 22 #40
        elif 'deit_small' in self.config.backbone:
            base_bs = 88 #212
        elif 'vit_base' in self.config.backbone:
            base_bs = 40 #96

        # -------------------------
        # 2) DATASET → #CLASSES
        # -------------------------

        dataset_nclasses = {
            'stanford_cars':         196,
            'caltech101':            101,
            'dtd':                   47,
            'eurosat':               10,
            'fgvc-aircraft-2013b':   100,
            'flowers-102':           102,
            'oxford-iiit-pet':       37
        }
        n_classes = dataset_nclasses.get(self.config.dataset, 100)  # fallback if unknown

        # -------------------------
        # 3) SCALE BY #CLASSES
        # -------------------------

        if n_classes <= 10:
            class_scale = 1.00
        elif n_classes <= 35:
            class_scale = 0.85
        elif n_classes <= 105:
            class_scale = 0.75
        elif n_classes <= 200:
            class_scale = 0.65
        else:
            print('undefined')

        # -------------------------
        # 5) FINAL BATCH SIZE
        # -------------------------
        
        batch_size = 2 * int(base_bs * class_scale * 3/4 ) 

        return batch_size

        
    def test_batch_size(self,):
        
        batch_size = self.train_batch_size() / 2
        
        return int(batch_size)
    
    def aggregate_results(self, results, corruption_type):

        total_correct = 0
        total_examples = 0

        # Sum up values from each process
        for process_id, process_data in results.items():
            total_correct += process_data['nb_correct']
            total_examples += process_data['nb_examples']

        # Calculate percentages
        accuracy = total_correct / total_examples
        
        statistic = { corruption_type+'_acc': accuracy  }

        return statistic

    def log_results(self, hpo_results=None, statistic=None):

        data_path = './results/results_{}_{}.pkl'.format(self.project_name, self.exp_id)

        # Load the current experiment configuration only once when first saving
        current_experiment_config = OmegaConf.load("./configs/HPO_{}_{}.yaml".format(self.project_name, self.exp_id))

        # Use a file lock to prevent concurrent access
        lock = FileLock(data_path + '.lock')

        with lock:
            # Load existing results if the file exists
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    results_dict = pickle.load(f)
            else:
                results_dict = {}

            # Ensure the current experiment's structure
            if self.exp_id not in results_dict:
                results_dict[self.exp_id] = {
                    "config": current_experiment_config,  # Set config once when first initializing
                    "statistics": {'clean_acc': None, 'Linf_acc': None, 'L2_acc': None, 'L1_acc': None},
                    "hpo_results": {}
                }

            if hpo_results:
                results_dict[self.exp_id]["hpo_results"] = hpo_results

            if statistic:
                results_dict[self.exp_id]["statistics"].update(statistic)

            # Save the updated dictionary back to the file
            with open(data_path, 'wb') as f:
                pickle.dump(results_dict, f)