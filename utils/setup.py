
# from comet_ml import Experiment

import os
import torch
import torch.distributed as dist


class Setup:

    def __init__(self, args, config_name):
        self.config_name = config_name
        self.args = args

    # def initialize_monitor(self, ):
        
    #     print('instantiate experiment')
        

    # def update_monitor(self, iteration, optimizer, loss, gradient_norm):
    #     # Compute gradient norms
        
    #     current_lr = optimizer.param_groups[0]['lr']

    #     # Log each metric for the current epoch
    #     self.exp_logger.log_metric("iteration", iteration, epoch=iteration)
    #     self.exp_logger.log_metric("loss_value", loss, epoch=iteration)
    #     self.exp_logger.log_metric("lr_schedule", current_lr, epoch=iteration)
    #     self.exp_logger.log_metric("gradient_norm", gradient_norm, epoch=iteration)

    # def end_monitor(self,):
    #     self.exp_logger.end()
        
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

    def batch_sizes(self,):
        if os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'beluga' and self.args.loss == 'TRADES':
            batch_size_uncertainty = 512
            batch_size_update = 64
            batch_size_pgdacc = 64
            batch_size_cleanacc = 512
        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval' and self.args.loss == 'TRADES':
            batch_size_uncertainty = 1024
            batch_size_update = 256
            batch_size_pgdacc = 128
            batch_size_cleanacc = 512
        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'cedar' and self.args.loss == 'TRADES':
            batch_size_uncertainty = 1024
            batch_size_update = 64
            batch_size_pgdacc = 128
            batch_size_cleanacc = 1024

        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'beluga' and self.args.loss == 'Madry':
            batch_size_uncertainty = 512
            batch_size_update = 512
            batch_size_pgdacc = 512
            batch_size_cleanacc = 512
        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'cedar' and self.args.loss == 'Madry':
            batch_size_uncertainty = 512
            batch_size_update = 512
            batch_size_pgdacc = 128
            batch_size_cleanacc = 1024
        elif os.environ.get('SLURM_CLUSTER_NAME', 'Unknown') == 'narval' and self.loss == 'Madry':
            batch_size_uncertainty = 1024
            batch_size_update = 1024
            batch_size_pgdacc = 1024
            batch_size_cleanacc = 1024
        else:
            print('error')

        return batch_size_uncertainty, batch_size_update, batch_size_pgdacc, batch_size_cleanacc