
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

class Hp_opt:

    def __init__(self,):
        pass

    def get_config(self, ):

        # Ray Tune hyperparameter search space
        tune_config = {
            "lr1": tune.loguniform(1e-5, 1e-1),
            "lr2": tune.loguniform(1e-5, 1e-1),
            "weight_decay": tune.loguniform(1e-6, 1e-2)
        }

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
        max_runtime_seconds = timedelta(minutes=150).total_seconds() #150

        # Set up the Tuner with metric and mode specified
        tuner = Tuner(
            trainer,
            param_space={"train_loop_config": update_config},
            tune_config=tune.TuneConfig(
                metric="loss",  # Specify the metric to optimize
                mode="min",     # Specify the optimization direction
                scheduler=scheduler,
                num_samples=15,
                time_budget_s=max_runtime_seconds,
                ),
            run_config=RunConfig(
                name="hpo_experiment",
                  ),
        )

        return tuner