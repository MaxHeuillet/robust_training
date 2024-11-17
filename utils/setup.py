
# from comet_ml import Experiment

import os
import torch
import torch.distributed as dist


class Setup:

    def __init__(self, args, config_name):
        self.config_name = config_name
        self.args = args
        
    def distributed_setup(self, world_size, rank):
        
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
        if self.args.backbone == 'robust_wideresnet_28_10':
            batch_size = 8 * base  # 8 = OK, 16 = NOT OK, 12 NOT OK
        elif self.args.backbone in ['deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224']:
            batch_size = 128 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = OK, 256 = NOT OK
        elif self.args.backbone in ['vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
        elif self.args.backbone in ['convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK
        elif self.args.backbone in ['convnext_tiny', 'convnext_tiny.fb_in22k', 'robust_convnext_tiny']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK

        return batch_size
        
    def test_batch_size(self,):

        cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        
        if cluster_name == 'narval':
            base = 2.5
        elif cluster_name == 'beluga':
            base = 1

        # Batch size recommendations based on the backbone
        if self.args.backbone == 'robust_wideresnet_28_10':
            batch_size = 8 * base  # 8 = OK, 16 = NOT OK, 12 NOT OK
        elif self.args.backbone in ['deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224']:
            batch_size = 128 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = OK, 256 = NOT OK
        elif self.args.backbone in ['vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
        elif self.args.backbone in ['convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base']:
            batch_size = 32 * base  # 16 = OK, 32 = OK, 64 = NOT OK
        elif self.args.backbone in ['convnext_tiny', 'convnext_tiny.fb_in22k', 'robust_convnext_tiny']:
            batch_size = 64 * base  # 16 = OK, 32 = OK, 64 = OK, 128 = NOT OK
        
        # if os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'beluga' and self.args.backbone == 'robust_wideresnet_28_10':
        #     batch_size = 16
        # elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'beluga' and self.args.dataset in ['EuroSAT']:
        #     batch_size = 128
        # elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval' and self.args.dataset in ['EuroSAT']:
        #     batch_size = 256
        # elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval' and self.args.dataset in ['Aircraft']:
        #     batch_size = 64
        # elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval' and self.args.dataset in ['CIFAR10', 'CIFAR100']:
        #     batch_size = 512
        # else:
        #     print('error')

        return batch_size