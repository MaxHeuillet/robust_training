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
from pathlib import Path  # In case config.dataset_path is a string


class Hp_opt:

    def __init__(self, config, ):
        self.config = config

        cluster_keywords = ["calculquebec", "calcul.quebec", "computecanada.ca"]
        nodename = os.uname().nodename.lower()
        # Check if the node is part of the Calcul Québec cluster
        if any(keyword in nodename for keyword in cluster_keywords):
            self.trials = 2 
            self.minutes = 5#120
        else:
            self.trials = 2
            self.minutes = 5

    def get_config(self):
        # Ray Tune hyperparameter search space

        if 'linearprobe' in self.config.project_name:
            tune_config = {
                "lr2": tune.loguniform(1e-6, 1e-1),  # Search over a range for the classification head
                "weight_decay2": tune.loguniform(1e-6, 1e-2),
                "scheduler": tune.choice([True, False])
            }
                    
        elif 'full_fine_tuning' in self.config.project_name:
            tune_config = {
                "lr1": tune.loguniform(1e-6, 1e-1),
                "lr2": tune.loguniform(1e-6, 1e-1),
                "weight_decay1": tune.loguniform(1e-6, 1e-2),
                "weight_decay2": tune.loguniform(1e-6, 1e-2),
                "scheduler": tune.choice([True, False])   }

        else:
            print('not implemented error')

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
                resources_per_worker={"CPU": 7, "GPU": 1},  # Resources per worker
            ),
        )
        return trainer
    
    def get_tuner(self, training_func):

        update_config = self.get_config()
        scheduler = self.get_scheduler()
        trainer = self.get_trainer(training_func)

        # Define maximum runtime in seconds
        max_runtime_seconds = timedelta(minutes=self.minutes).total_seconds()
        path = Path(os.path.expandvars(self.config.work_path)).expanduser().resolve()

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
                reuse_actors=True
            ),
            run_config=RunConfig(
                name=f"{self.config.exp_id}",
                storage_path=f"file://{path}",
            ),
        )

        return tuner