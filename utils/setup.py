
# from comet_ml import Experiment

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
        # These are "safe but reasonably large" total batch sizes for 4 GPUs on 224x224 images.

        if 'convnext_tiny' in self.config.backbone:
            base_bs = 64 #124
        elif 'convnext_base' in self.config.backbone:
            base_bs = 32 #40
        elif 'deit_small' in self.config.backbone:
            base_bs = 128 #212
        elif 'vit_base' in self.config.backbone:
            base_bs = 64 #96

        # -------------------------
        # 2) DATASET → #CLASSES
        # -------------------------
        # Approximate #classes for each dataset
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
        # More classes => slightly larger last layer => slightly higher memory usage.
        # You can tune these multipliers further if you see OOM or leftover memory.
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
        # 4) SCALE FOR TRADES
        # -------------------------
        # TRADES effectively does ~2 forward passes => ~2x memory usage.

        # -------------------------
        # 5) FINAL BATCH SIZE
        # -------------------------
        batch_size = int(base_bs * class_scale * 3/4 ) 

        return batch_size

        
    def test_batch_size(self,):
        
        batch_size = self.train_batch_size() / 2
        
        return int(batch_size)
    
    def aggregate_results(self,results):

        print(results)
        # Initialize sums
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0

        # total_neurons_nat = 0
        # total_zero_nat = 0
        # total_dormant_nat = 0
        # total_overactive_nat = 0

        # total_neurons_adv = 0
        # total_zero_adv = 0
        # total_dormant_adv = 0
        # total_overactive_adv = 0

        # Sum up values from each process
        for process_id, process_data in results.items():
            total_correct_nat += process_data['stats']['nb_correct_nat']
            total_correct_adv += process_data['stats']['nb_correct_adv']
            total_examples += process_data['stats']['nb_examples']

            # total_zero_nat += process_data['stats_nat']['zero_count']
            # total_dormant_nat += process_data['stats_nat']['dormant_count']
            # total_overactive_nat += process_data['stats_nat']['overactive_count']
            # total_neurons_nat += process_data['stats_nat']['total_neurons']

            # total_zero_adv += process_data['stats_adv']['zero_count']
            # total_dormant_adv += process_data['stats_adv']['dormant_count']
            # total_overactive_adv += process_data['stats_adv']['overactive_count']
            # total_neurons_adv += process_data['stats_adv']['total_neurons']

        # Calculate percentages
        clean_accuracy = total_correct_nat / total_examples
        robust_accuracy = total_correct_adv / total_examples
        
        # nat_zero_mean = total_zero_nat / total_neurons_nat
        # nat_dormant_mean = total_dormant_nat / total_neurons_nat
        # nat_overactive_mean = total_overactive_nat / total_neurons_nat

        # adv_zero_mean = total_zero_adv / total_neurons_adv
        # adv_dormant_mean = total_dormant_adv / total_neurons_adv
        # adv_overactive_mean = total_overactive_adv / total_neurons_adv

        statistics = {  'clean_acc':clean_accuracy, 
                        'robust_acc':robust_accuracy,
                        
                        # 'zero_nat_test':nat_zero_mean,
                        # 'dormant_nat_test':nat_dormant_mean,
                        # 'overactive_nat_test':nat_overactive_mean,
                        
                        # 'zero_adv_test':adv_zero_mean,
                        # 'dormant_adv_test':adv_dormant_mean,
                        # 'overactive_adv_test':adv_overactive_mean,
                        
                        }

        return statistics
    
    def log_results(self, hpo_results=None, statistics=None, ):
        import cloudpickle as pickle

        data_path = './results/results_{}_{}.pkl'.format( self.project_name, self.exp_id  )

        # Load the current experiment configuration
        current_experiment_config = OmegaConf.load("./configs/HPO_{}_{}.yaml".format(self.project_name, self.exp_id) )

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
                    "config": {},
                    "statistics": {},
                    "hpo_results": {}
                }

            # Update the dictionary with the respective sections
            results_dict[self.exp_id]["config"].update(current_experiment_config)
            
            if hpo_results:
                results_dict[self.exp_id]["hpo_results"] = hpo_results

            if statistics:
                results_dict[self.exp_id]["statistics"].update(statistics)
            
            # Save the updated dictionary back to the file
            with open(data_path, 'wb') as f:
                pickle.dump(results_dict, f)



# EXAMPLE USAGE
# -------------
# Suppose we're training convnext_tiny on 'stanford_cars' with TRADES:
#   arch      = 'convnext_tiny'
#   dataset   = 'stanford_cars'
#   loss_fn   = 'trades'
#   batchsize = get_safe_batch_size(arch, dataset, loss_fn)
#
# This gives us the total batch size across 4 GPUs (16GB each). 
# If you do DDP, you'll set per-GPU batch size to batchsize // 4 (rounded).

        
    # def train_batch_size(self):

    #     if self.cluster_name == 'narval':
    #         base = 9/4
    #     elif self.cluster_name == 'beluga':
    #         base = 3/4
    #     else:
    #         return 8
        
    #     if self.config.loss_function == 'TRADES_v2':
    #         loss_base = 0.5
    #     else:
    #         loss_base = 1

    #     # Batch size recommendations based on the backbone
    #     # if self.config.backbone in ['robust_wideresnet_28_10', 'wideresnet_28_10']:
    #     #     batch_size = 8 * base  # 8 = OK, 16 = NOT OK, 12 NOT OK
    #     if 'deit_small_patch16_224' in self.config.backbone:
    #         batch_size = 128 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = OK, 256 = NOT OK
    #     elif 'vit_base_patch16_224' in self.config.backbone:
    #         batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
    #     elif 'convnext_base' in self.config.backbone:
    #         batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK
    #     elif 'convnext_tiny' in self.config.backbone:
    #         batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
  
    #     return int(batch_size)