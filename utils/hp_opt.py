
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer, get_device, get_devices
from ray import tune
from ray.air.config import RunConfig
from ray.tune.tuner import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from omegaconf import OmegaConf

import torch
import numpy as np

import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import os
from ray.tune.search.basic_variant import BasicVariantGenerator

class Hp_opt:

    def __init__(self,setup):
        self.setup = setup
        cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        if cluster_name == 'narval' or cluster_name == 'beluga':
            self.minutes = 150
            self.trials = 1000
        else:
            self.minutes = 5
            self.trials = 5

    def get_config(self, ):

        # Ray Tune hyperparameter search space
        tune_config = {
            "lr1": tune.loguniform(1e-5, 1e-1),
            "lr2": tune.loguniform(1e-5, 1e-1),
            "weight_decay1": tune.loguniform(1e-6, 1e-2),
            "weight_decay2": tune.loguniform(1e-6, 1e-2),
            #"use_scheduler": tune.choice([True, False])
        }

        # add the choice of the learning rate, add the choice of scheduler

        return tune_config

    def get_scheduler(self, epochs):

        # Configure the scheduler WITHOUT metric and mode
        scheduler = ASHAScheduler(
            max_t=epochs,
            grace_period=1,
            reduction_factor=2
        )

        return scheduler
    
    def get_trainer(self, training_func):

        # Determine the number of workers and GPU usage
        num_gpus = torch.cuda.device_count()
        print(num_gpus)

        # Initialize the TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=training_func,

            scaling_config=ScalingConfig(
                # num_workers=4,  # Number of workers
                use_gpu=True,  # Use GPUs
                resources_per_worker={"CPU": 6, "GPU": 1},  # Resources per worker
            ),
        )

        return trainer
    
    def get_tuner(self, epochs, training_func):

        update_config = self.get_config()
        scheduler = self.get_scheduler(epochs)
        trainer = self.get_trainer( training_func )

        # Define maximum runtime in seconds
        from datetime import timedelta
        max_runtime_seconds = timedelta(minutes=self.minutes).total_seconds() 

        # Set up the Tuner with metric and mode specified

        # full_path = os.path.abspath("./hpo_results")

        tuner = Tuner(
            trainer,
            param_space={"train_loop_config": update_config},
            tune_config=tune.TuneConfig(
                metric="loss",  # Specify the metric to optimize
                mode="min",     # Specify the optimization direction
                scheduler=scheduler,
                num_samples=self.trials, #
                time_budget_s=max_runtime_seconds,
                ),
            run_config=RunConfig(
                name="{}_{}".format(self.setup.project_name, self.setup.exp_id),
                # storage_path=f"file://{full_path}",
                  ),
        )

        return tuner