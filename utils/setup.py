
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

    def test_batch_size(self,):

        if os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval' and self.args.backbone == 'robust_wideresnet_28_10':
            batch_size = 8
        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval' and self.args.backbone != 'robust_wideresnet_28_10':
            batch_size = 64
        else:
            batch_size = 16
        
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