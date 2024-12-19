import os

from ray import train, tune
from ray.tune.examples.mnist_pytorch import train_mnist
from ray.tune import ResultGrid

from hydra import initialize, compose

from omegaconf import OmegaConf
from datasets import load_data
from architectures import load_architecture
from utils import load_optimizer, get_args2, set_seeds, Hp_opt

import torch

# initialize(config_path="configs", version_base=None)

from utils import Setup
from distributed_experiment2 import BaseExperiment

world_size = torch.cuda.device_count()


full_path = os.path.abspath("./hpo_results")

project_name = 'test30'

dataset = 'Aircraft'
loss = 'CLASSIC_AT'
backbone = 'convnext_tiny.fb_in22k'
            
config = OmegaConf.load("./configs/default_config.yaml")

config = compose(config_name="default_config")  # Store Hydra config in a variable

config.dataset = dataset
config.backbone = backbone
config.loss_function = loss
config.project_name = 'test30'

set_seeds(config.seed)

print(config)

setup = Setup(config, world_size)
experiment = BaseExperiment(setup)
 
hp_search = Hp_opt(setup)
trainer = hp_search.get_trainer(experiment.training)


# tune.Tuner.can_restore( '{}/{}_{}_{}_{}'.format(full_path, project_name, backbone, dataset, loss) )

# tune.Tuner.

tuner = tune.Tuner.restore('{}/{}_{}_{}_{}'.format(full_path, project_name, backbone, dataset, loss), trainable=trainer)


# tuner = tune.Tuner.restore('{}/0001_deit_small_patch16_224.fb_in1k_Flowers_CLASSIC_AT'.format(full_path, project_name, backbone, dataset, loss), trainable=trainer)

result_grid = tuner.get_results()