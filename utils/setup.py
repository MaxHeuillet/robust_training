
# from comet_ml import Experiment

import os
import torch
import torch.distributed as dist
from filelock import FileLock
import pandas as pd
from datetime import datetime





def generate_timestamp():
    return datetime.now().strftime('%y/%m/%d/%H/%M/%S')

def check_unique_id(df1, df2, unique_id_col='unique_id'):

    unique_id = df2[unique_id_col].iloc[0]
    
    matching_indices = df1[df1[unique_id_col] == unique_id].index

    if not matching_indices.empty:
        iloc_indices = [df1.index.get_loc(idx) for idx in matching_indices]
        return True, iloc_indices
    else:
        return False, []

class Setup:

    def __init__(self, args, config_name, exp_id, current_experiment):
        self.config_name = config_name
        self.exp_id = exp_id
        self.args = args
        self.current_experiment = current_experiment
        
    def distributed_setup(self, world_size, rank):

        # os.environ['NCCL_DEBUG'] = 'INFO'  # or 'TRACE' for more detailed logs
        # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        # os.environ['NCCL_BLOCKING_WAIT'] = '1'

        print('torch', torch.__version__)
        print('cuda', torch.version.cuda)
        print('cudnn', torch.backends.cudnn.version())
        
        #Initialize the distributed environment.
        print( ' world size {}, rank {}'.format(world_size,rank) )
        print('set up the master adress and port')
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'

        #Set environment variables for offline usage of Hugging Face libraries
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        #Set up the local GPU for this process
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print('init process group ok')

    def cleanup(self,):
        dist.destroy_process_group()

        
    def train_batch_size(self):

        cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        
        if cluster_name == 'narval':
            base = 2.5
        elif cluster_name == 'beluga':
            base = 1

        # Batch size recommendations based on the backbone
        if self.args.backbone in ['robust_wideresnet_28_10', 'wideresnet_28_10']:
            batch_size = 8 * base  # 8 = OK, 16 = NOT OK, 12 NOT OK
        elif self.args.backbone in ['deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224']:
            batch_size = 128 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = OK, 256 = NOT OK
        elif self.args.backbone in ['vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
        elif self.args.backbone in ['convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK
        elif self.args.backbone in ['convnext_tiny', 'convnext_tiny.fb_in22k', 'robust_convnext_tiny']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK

        return int(batch_size)
        
    def test_batch_size(self,):

        cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        
        if cluster_name == 'narval':
            base = 2.5
        elif cluster_name == 'beluga':
            base = 1

        # Batch size recommendations based on the backbone
        if self.args.backbone in ['robust_wideresnet_28_10', 'wideresnet_28_10']:
            batch_size = 4 * base  # 8 = NOT OK,
        elif self.args.backbone in ['deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK, 
        elif self.args.backbone in ['vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK,
        elif self.args.backbone in ['convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base']:
            batch_size = 16 * base  # 16 = OK, 32 = NOT OK,
        elif self.args.backbone in ['convnext_tiny', 'convnext_tiny.fb_in22k', 'robust_convnext_tiny']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK,
        
        return int(batch_size)
    
    def log_results(self, statistics):

        cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        data_path = './results/results_{}_{}.csv'.format(cluster_name, self.args.exp)
        
        lock = FileLock(data_path + '.lock')

        with lock:
            
            columns = list(self.current_experiment.keys())
            key_columns = columns.copy()

            try:
                df = pd.read_csv(data_path)
            except FileNotFoundError:
                df = pd.DataFrame(columns=key_columns + ['timestamp', 'clean_acc', 'robust_acc'])

            self.current_experiment['id'] = self.epx_id        
            self.current_experiment['timestamp'] = generate_timestamp()
            self.current_experiment = self.current_experiment | statistics
            
            new_row = pd.DataFrame([self.current_experiment], columns=self.current_experiment.keys() )
            
            if df.empty:
                df = pd.concat([df, new_row]) 
                
            else:
                exists, match_mask = check_unique_id(df, new_row, 'id')
                print('exists', exists, match_mask)

                if not exists:
                    print('experiment does not exist in the database')
                    df = pd.concat([df, new_row])

                else:
                    print('experiment already exists in the database')
                    # If the experiment exists, overwrite the existing row
                    new_row = new_row[df.columns]
                    df.loc[match_mask, :] = new_row.values

            df.to_csv(data_path, header=True, index=False)