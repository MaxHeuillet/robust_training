import os
from ray import train, tune
from ray.tune import ResultGrid
from utils import Setup
from distributed_experiment2 import BaseExperiment
from hydra import initialize, compose
from omegaconf import OmegaConf
from utils import  set_seeds, Hp_opt
import torch

import logging
from ray.tune.logger import Logger

# Set the logging level for ray.tune
logging.getLogger("ray").setLevel(logging.ERROR)

initialize(config_path="configs", version_base=None)

# Compute pairwise Euclidean distances
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

def result_grid_analysis(result_grid):

    data = []

    for result in result_grid:

        res = {}
        
        if len(result.metrics.keys() ) > 1:

            res['lr1'] = result.config['train_loop_config']['lr1']
            res['lr2'] = result.config['train_loop_config']['lr2']
            res['weight_decay1'] = result.config['train_loop_config']['weight_decay1']
            res['weight_decay2'] = result.config['train_loop_config']['weight_decay2']

            label = f"lr1={res['lr1']:.3f}, lr2={res['lr2']}, weight_decay1={res['weight_decay1']:.3f}, weight_decay2={res['weight_decay2']} "

            loss_list = result.metrics_dataframe.sort_values("training_iteration")["loss"].tolist()
            
            res['loss_list'] = loss_list

            data.append(res)

    max_len = max(len(d['loss_list']) for d in data)

    # 2) Pad each loss_list to ensure they all have the same length
    for d in data:
        d['loss_list'] += [None] * (max_len - len(d['loss_list']))

    # 3) Expand each entry in loss_list into separate columns (loss_0, loss_1, etc.)
    for d in data:
        for i, val in enumerate(d['loss_list']):
            d[f'loss_{i}'] = val
        # Optionally delete the original loss_list if you don’t need it anymore
        del d['loss_list']

    # 4) Create the DataFrame
    df = pd.DataFrame(data)
    
    # 1. Inner Variability: Standard Deviation and Mean Absolute Change (per time series)
    loss_columns = [col for col in df.columns if col.startswith('loss_')]

    df = df.fillna(0)
    # Compute standard deviation and mean absolute change per time series
    df['std_dev'] = df[loss_columns].std(axis=1, skipna=True)
    df['mean_abs_change'] = df[loss_columns].diff(axis=1).abs().mean(axis=1, skipna=True)

    filled_data = df[loss_columns].fillna(0).to_numpy()

    pairwise_distances = pdist(filled_data, metric='euclidean')
    mean_pairwise_distance = np.mean(pairwise_distances)
    variance_pairwise_distance = np.var(pairwise_distances)

    return { "MeanPairwiseDistance": mean_pairwise_distance,
             "VariancePairwiseDistance": variance_pairwise_distance,
             "MeanSTD": np.mean(df['std_dev']), 
             "VarianceSTD":np.std(df['std_dev']),
             "MeanAbsChange": np.mean(df['mean_abs_change']), 
             "VarianceAbsChange":np.std(df['mean_abs_change'])  }

world_size = torch.cuda.device_count()

full_path = os.path.abspath("~/scratch/hpo_results")

project_name = 'full_fine_tuning_5epochs_final1'

datas=( 'stanford_cars', 'caltech101', 'fgvc-aircraft-2013b', 'dtd', 'flowers-102', 'oxford-iiit-pet'  ) #'imagenette2' 'eurosat' 

losses=( 'TRADES_v2', 'CLASSIC_AT' ) #  

backbones=(
  'convnext_tiny', 'robust_convnext_tiny', 'convnext_tiny.fb_in22k', 
  'deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224',
  'convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base', 
  'convnext_base.clip_laion2b', 'convnext_base.clip_laion2b_augreg',
  'vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k',
  'vit_base_patch16_224.dino', 'vit_base_patch16_224.mae', 'vit_base_patch16_224.orig_in21k',
  'vit_base_patch16_224.sam_in1k', 'vit_base_patch16_224_miil.in21k'  ) 

config = OmegaConf.load("./configs/default_config.yaml")
config = compose(config_name="default_config")  # Store Hydra config in a variable


import pickle
import numpy as np

final_data = []

for backbone in backbones:
    for loss in losses:
        for dataset in datas:

            statistics = {}
            statistics["dataset"] = dataset
            statistics["backbone"] = backbone
            statistics["loss_function"] = loss


            config.dataset = dataset
            config.backbone = backbone
            config.loss_function = loss
            config.project_name = project_name

            set_seeds(config.seed)
            setup = Setup(config, world_size)
            experiment = BaseExperiment(setup)
            hp_search = Hp_opt(setup)
            trainer = hp_search.get_trainer(experiment.training)

            experiment_path = "~/scratch/hpo_results/{}_{}".format(setup.project_name, setup.exp_id)

            try:

                restored_tuner = tune.Tuner.restore(experiment_path, trainable=trainer)
                result_grid = restored_tuner.get_results()

                stats = result_grid_analysis(result_grid)

                statistics.update(stats)
                final_data.append(statistics)
                
                
            except:
                pass
                # stats  = { "MeanPairwiseDistance": np.nan,
                #             "VariancePairwiseDistance": np.nan,
                #             "MeanSTD": np.nan,
                #             "VarianceSTD": np.nan,
                #             "MeanAbsChange": np.nan,
                #             "VarianceAbsChange": np.nan,  }


final_data = pd.DataFrame(final_data)
final_data.to_csv( "~/projects/def-adurand/mheuill/robust_training/RQ4_final_data_{}.csv".format(project_name) )