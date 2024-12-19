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

initialize(config_path="configs", version_base=None)

from utils import Setup
from distributed_experiment2 import BaseExperiment

world_size = torch.cuda.device_count()


full_path = os.path.abspath("./hpo_results")

project_name = 'test30'

for dataset in ['Aircraft', 'Flowers', 'Imagenette' ]:

    for loss in ['CLASSIC_AT', 'TRADES_v2']:

        for backbone in [ 'convnext_tiny.fb_in22k', 'convnext_tiny', 'robust_convnext_tiny', ]:
            
            config = OmegaConf.load("./configs/default_config.yaml")

            config = compose(config_name="default_config")  # Store Hydra config in a variable

            config.dataset = dataset
            config.backbone = backbone
            config.loss_function = loss
            config.project_name = project_name

            set_seeds(config.seed)

            setup = Setup(config, world_size)
            experiment = BaseExperiment(setup)
            
            hp_search = Hp_opt(setup)
            trainer = hp_search.get_trainer(experiment.training)

            tuner = tune.Tuner.restore('{}/{}_{}_{}_{}'.format(full_path, project_name, backbone, dataset, loss), trainable=trainer)

            result_grid = tuner.get_results()

            print()

            print( result_grid.get_best_result() )

            print(result_grid)

            print()

            ax = None
            for result in result_grid:
                print(result.metrics_dataframe)
                res = result.config['train_loop_config']
                # print(res)
                print()
                label = f"lr1={res['lr1']:.3f}, lr2={res['lr2']}"
                if ax is None:
                    ax = result.metrics_dataframe.plot("training_iteration", "loss", label=label)
                else:
                    result.metrics_dataframe.plot("training_iteration", "loss", ax=ax, label=label)
                    
            ax.set_title("Loss vs. Training Iteration for All Trials")
            ax.set_ylabel("Loss")

            ax.figure.savefig("./results/{}_{}_{}.png".format(project_name, backbone, dataset, loss), dpi=300)