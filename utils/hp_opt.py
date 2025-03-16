from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer, get_device, get_devices
from ray import tune
from ray.air.config import RunConfig
from ray.tune.tuner import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

import torch
import os
from datetime import timedelta
import subprocess

class Hp_opt:

    def __init__(self, config):
        self.config = config
        cluster_name = os.environ.get('SLURM_CLUSTER_NAME', 'Unknown')
        if cluster_name in ['narval', 'beluga']:
            self.minutes = 5 #150
            self.trials = 1 #1000
        else:
            self.minutes = 2
            self.trials = 2

    def get_config(self):
        # Ray Tune hyperparameter search space

        tune_config = {
            "lr1": tune.loguniform(1e-5, 1e-1),
            "lr2": tune.loguniform(1e-5, 1e-1),
            "weight_decay1": tune.loguniform(1e-6, 1e-2),
            "weight_decay2": tune.loguniform(1e-6, 1e-2),
            "scheduler": tune.choice([True, False])   }
        
        return tune_config

    def get_scheduler(self, ):
        # Configure the scheduler WITHOUT metric and mode
        scheduler = ASHAScheduler(
            max_t=self.config.epochs,
            grace_period=1,
            reduction_factor=2
        )
        return scheduler
    
    def get_trainer(self, training_func):
        # Determine the number of workers and GPU usage
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        # Initialize the TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=training_func,
            scaling_config=ScalingConfig(
                use_gpu=True,  # Use GPUs
                resources_per_worker={"CPU": 6, "GPU": 1},  # Resources per worker
            ),
        )
        return trainer
    
    def get_tuner(self, training_func):

        update_config = self.get_config()
        scheduler = self.get_scheduler()
        trainer = self.get_trainer(training_func)

        # Define maximum runtime in seconds
        max_runtime_seconds = timedelta(minutes=self.minutes).total_seconds()

        # Set up storage path
        full_path = os.path.expanduser(self.config.hpo_path)
        experiment_path = os.path.join(full_path, self.config.project_name, self.config.exp_id )

        # Check if experiment path exists and delete it before starting a new run
        if os.path.exists(experiment_path):
            print(f"Deleting existing experiment directory: {experiment_path}")
            subprocess.run(["rm", "-rf", experiment_path], check=True)

        os.makedirs(experiment_path, exist_ok=True)

        # Set up the Tuner
        tuner = Tuner(
            trainer,
            param_space={"train_loop_config": update_config},
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=self.trials,
                time_budget_s=max_runtime_seconds,
            ),
            run_config=RunConfig(
                name=f"{self.config.project_name}_{self.config.exp_id}",
                storage_path=f"file://{full_path}",
            ),
        )

        return tuner