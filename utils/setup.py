
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
        
    def train_batch_size(self):

        if self.cluster_name == 'narval':
            base = 9/4
        elif self.cluster_name == 'beluga':
            base = 1
        else:
            return 8

        # Batch size recommendations based on the backbone
        if self.config.backbone in ['robust_wideresnet_28_10', 'wideresnet_28_10']:
            batch_size = 8 * base  # 8 = OK, 16 = NOT OK, 12 NOT OK
        elif self.config.backbone in ['deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224', 'random_deit_small_patch16_224']:
            batch_size = 128 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = OK, 256 = NOT OK
        elif self.config.backbone in ['vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224', 'random_vit_base_patch16_224']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
        elif self.config.backbone in ['convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base', 'random_convnext_base']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK
        elif self.config.backbone in ['convnext_tiny_random', 'convnext_tiny', 'convnext_tiny.fb_in22k', 'robust_convnext_tiny', 'random_convnext_tiny']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
  
        return int(batch_size)
        
    def test_batch_size(self,):
        
        if self.cluster_name == 'narval':
            base = 9/4
        elif self.cluster_name == 'beluga':
            base = 1
        else:
            return 8

        # Batch size recommendations based on the backbone
        if self.config.backbone in ['robust_wideresnet_28_10', 'wideresnet_28_10']:
            batch_size = 4 * base  # 8 = NOT OK,
        elif self.config.backbone in ['deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224', 'random_deit_small_patch16_224']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK, 
        elif self.config.backbone in ['vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224', 'random_vit_base_patch16_224']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK,
        elif self.config.backbone in ['convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base', 'random_convnext_base']:
            batch_size = 16 * base  # 16 = OK, 32 = NOT OK,
        elif self.config.backbone in ['convnext_tiny_random', 'convnext_tiny', 'convnext_tiny.fb_in22k', 'robust_convnext_tiny', 'random_convnext_tiny']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK,
        
        return int(batch_size)
    
    def aggregate_results(self,results):
        # Initialize sums
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0

        total_neurons_nat = 0
        total_zero_nat = 0
        total_dormant_nat = 0
        total_overactive_nat = 0

        # Sum up values from each process
        for process_id, process_data in results.items():
            total_correct_nat += process_data['stats']['nb_correct_nat']
            total_correct_adv += process_data['stats']['nb_correct_adv']
            total_examples += process_data['stats']['nb_examples']

            total_zero_nat += process_data['stats_nat']['zero_count']
            total_dormant_nat += process_data['stats_nat']['dormant_count']
            total_overactive_nat += process_data['stats_nat']['overactive_count']
            total_neurons_nat += process_data['stats_nat']['total_neurons']

        # Calculate percentages
        clean_accuracy = total_correct_nat / total_examples
        robust_accuracy = total_correct_adv / total_examples
        nat_zero_mean = total_zero_nat / total_neurons_nat
        nat_dormant_mean = total_dormant_nat / total_neurons_nat
        nat_overactive_mean = total_overactive_nat / total_neurons_nat

        statistics = { 'clean_acc':clean_accuracy, 
                       'robust_acc':robust_accuracy,
                        'nat_zero_mean':nat_zero_mean,
                        'nat_dormant_mean':nat_dormant_mean,
                        'nat_overactive_mean':nat_overactive_mean,
                        }

        return statistics
    
    def aggregate_results2(self,results):
        
        total_neurons_nat = 0
        total_zero_nat = 0
        total_dormant_nat = 0
        total_overactive_nat = 0

        # Sum up values from each process
        for process_id, process_data in results.items():

            total_zero_nat += process_data['zero_count']
            total_dormant_nat += process_data['dormant_count']
            total_overactive_nat += process_data['overactive_count']
            total_neurons_nat += process_data['total_neurons']

        nat_zero_mean = total_zero_nat / total_neurons_nat
        nat_dormant_mean = total_dormant_nat / total_neurons_nat
        nat_overactive_mean = total_overactive_nat / total_neurons_nat

        statistics = { 'zero_mean':nat_zero_mean,
                        'dormant_mean':nat_dormant_mean,
                        'overactive_mean':nat_overactive_mean,
                        }

        return statistics
    


    def log_results(self, hpo_results=None, statistics=None, dormants = None):
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
                print(hpo_results)

            if statistics:
                results_dict[self.exp_id]["statistics"].update(statistics)
            
            if dormants:
                results_dict[self.exp_id]["dormants"].update(dormants)
            

            # Save the updated dictionary back to the file
            with open(data_path, 'wb') as f:
                pickle.dump(results_dict, f)



    # def pre_training_log(self, ):

    #     statistics = { 'clean_acc':-2, 
    #                    'robust_acc':-2,
    #                     'nat_zero_mean':-2,
    #                     'nat_dormant_mean':-2,
    #                     'nat_overactive_mean':-2,
    #                     'adv_zero_mean':-2,
    #                     'adv_dormant_mean':-2,
    #                     'adv_overactive_mean':-2
    #                     }
        
    #     self.log_results(statistics)

    # def post_training_log(self, ):

    #     statistics = { 'clean_acc':-1, 
    #                    'robust_acc':-1,
    #                     'nat_zero_mean':-1,
    #                     'nat_dormant_mean':-1,
    #                     'nat_overactive_mean':-1,
    #                     'adv_zero_mean':-1,
    #                     'adv_dormant_mean':-1,
    #                     'adv_overactive_mean':-1
    #                     }
        
    #     self.log_results(statistics)

    
    # def log_results(self, statistics):

    #     cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
    #     data_path = './results/results_{}.csv'.format( cluster_name )
    #     current_experiment = OmegaConf.to_container(self.config, resolve=True)
        
    #     lock = FileLock(data_path + '.lock')

    #     with lock:
            
    #         columns = list(self.config.keys())
    #         key_columns = columns.copy()

    #         try:
    #             df = pd.read_csv(data_path)
    #         except FileNotFoundError:
    #             df = pd.DataFrame(columns=key_columns + ['timestamp'] + list( statistics.keys() ) )
      
    #         current_experiment['timestamp'] = generate_timestamp()
    #         current_experiment.update(statistics)
            
    #         new_row = pd.DataFrame([current_experiment], columns=current_experiment.keys() )
            
    #         if df.empty:
    #             df = pd.concat([df, new_row]) 
                
    #         else:
    #             exists, match_mask = check_unique_id(df, new_row, 'exp_id')
    #             print('exists', exists, match_mask)

    #             if not exists:
    #                 print('experiment does not exist in the database')
    #                 df = pd.concat([df, new_row])

    #             else:
    #                 print('experiment already exists in the database')
    #                 # If the experiment exists, overwrite the existing row
    #                 new_row = new_row[df.columns]
    #                 df.loc[match_mask, :] = new_row.values

    #         df.to_csv(data_path, header=True, index=False)