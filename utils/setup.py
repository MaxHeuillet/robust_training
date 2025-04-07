
# from comet_ml import Experiment
# import cloudpickle as pickle

import os
import torch
import torch.distributed as dist
from filelock import FileLock
from omegaconf import OmegaConf
import pickle




class Setup:

    def __init__(self, world_size): #config
        # self.config = config
        # self.exp_id = get_config_id(self.config)
        self.world_size = world_size
        self.hp_opt = False
        self.cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        # self.project_name = config.project_name
        
    def distributed_setup(self, rank):

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
        if nb_examples == 0:
            print(f"⚠️ sync_value: No examples provided on rank {rank}. Returning inf.")
            return float("inf"), value, nb_examples

        value_tensor = torch.tensor([value], dtype=torch.float32, device=rank)
        nb_examples_tensor = torch.tensor([nb_examples], dtype=torch.float32, device=rank)

        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb_examples_tensor, op=dist.ReduceOp.SUM)

        # Check again after sync in case only one rank had data
        if nb_examples_tensor.item() == 0:
            print(f"⚠️ sync_value: Global nb_examples is 0. Returning inf.")
            return float("inf"), value_tensor.item(), 0

        avg_value = value_tensor.item() / nb_examples_tensor.item()
        return avg_value, value_tensor.item(), nb_examples_tensor.item()

    # def sync_value(self, value, nb_examples, rank):

    #     # Aggregate results across all processes
    #     value_tensor = torch.tensor([value], dtype=torch.float32, device=rank)
    #     nb_examples_tensor = torch.tensor([nb_examples], dtype=torch.float32, device=rank)

    #     dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(nb_examples_tensor, op=dist.ReduceOp.SUM)

    #     # Compute global averages
    #     avg_value = value_tensor.item() / nb_examples_tensor.item()

    #     return avg_value, value_tensor.item(), nb_examples_tensor.item() 
    

    def train_batch_size(self, config): #(arch: str, dataset: str, loss_fn: str) -> int

        # -------------------------
        # 1) BASELINES PER ARCH
        # -------------------------

        arch_lower = config.backbone.lower()
        base_bs = 30  # default fallback

        # 1) Match known architectures by substring
        if 'convnext_tiny' in arch_lower:
            base_bs = 55
        elif 'coatnet_2' in arch_lower:
            base_bs = 15
        elif any(x in arch_lower for x in ['convnext_base', ]):
            base_bs = 20
        elif any(x in arch_lower for x in ['deit_small', 'eva02_tiny', 'swin_tiny', 'coatnet_0', 'vit_small']):
            base_bs = 80
        elif any(x in arch_lower for x in ['vit_base', 'swin_base', 'eva02_base', ]):
            base_bs = 35
        elif 'resnet50' in arch_lower:
            base_bs = 60
        else:
            print(f"WARNING: unrecognized backbone '{config.backbone}', using fallback base_bs={base_bs}.")


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
            'oxford-iiit-pet':       37,
            'uc-merced-land-use-dataset': 21,
            'kvasir-dataset': 8
        }
        n_classes = dataset_nclasses.get(config.dataset, 100)  # fallback if unknown

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
            class_scale = 0.7
        else:
            print('undefined')

        # -------------------------
        # 5) FINAL BATCH SIZE
        # -------------------------
        
        cluster_keywords = ["calculquebec", "calcul.quebec"]
        nodename = os.uname().nodename.lower()
        # Check if the node is part of the Calcul Québec cluster
        if any(keyword in nodename for keyword in cluster_keywords):
            batch_size = int(base_bs * class_scale * 3/4 ) # * 2
        else:
            # Define the default data directory for non-cluster environments
            batch_size = 2

        return batch_size

        
    def test_batch_size(self, config):
        
        batch_size = self.train_batch_size(config) / 2
        
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

    def log_results(self, config, statistic):

        save_path = os.path.join(config.results_path, config.project_name, f"{config.exp_id}.pkl")

        # Use a file lock to prevent concurrent access
        lock = FileLock(save_path + '.lock')

        with lock:
            # Load existing results if the file exists
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    results = pickle.load(f)
            else:
                results = {'clean_acc': None, 'Linf_acc': None, 'L2_acc': None, 'L1_acc': None }

            results.update( statistic )

            # Save the updated dictionary back to the file
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)