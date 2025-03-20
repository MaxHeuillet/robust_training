import logging
import pandas as pd
import numpy as np
import ray
from ray import tune, init, remote
from scipy.spatial.distance import pdist
from hydra import initialize, compose
from omegaconf import OmegaConf
import torch

from utils import Setup, set_seeds, Hp_opt
from distributed_experiment_final import BaseExperiment

# Set the logging level for ray.tune
logging.getLogger("ray").setLevel(logging.ERROR)

initialize(config_path="configs", version_base=None)

# Initialize Ray
init()

def result_grid_analysis(result_grid):
    data = []

    for result in result_grid:
        if len(result.metrics.keys()) > 1:
            res = {
                'lr1': result.config['train_loop_config']['lr1'],
                'lr2': result.config['train_loop_config']['lr2'],
                'weight_decay1': result.config['train_loop_config']['weight_decay1'],
                'weight_decay2': result.config['train_loop_config']['weight_decay2'],
            }

            loss_list = result.metrics_dataframe["loss"].values
            res['loss_list'] = loss_list
            data.append(res)

    if not data:
        raise ValueError("No valid results in result_grid.")

    # Pre-allocate DataFrame
    max_len = max(len(d['loss_list']) for d in data)
    df = pd.DataFrame([{**d, **{f'loss_{i}': d['loss_list'][i] if i < len(d['loss_list']) else None
                                for i in range(max_len)}} for d in data])
    del df['loss_list']

    # Fill NaN with column means
    loss_columns = [col for col in df.columns if col.startswith('loss_')]
    # loss_means = df[loss_columns].mean(axis=0, skipna=True)
    df[loss_columns] = df[loss_columns].fillna(0)

    # Compute std_dev
    df['std_dev'] = df[loss_columns].std(axis=1, skipna=True)
    df['mean_abs_change'] = df[loss_columns].diff(axis=1).abs().mean(axis=1, skipna=True)

    # Compute statistics
    pairwise_distances = pdist(df[loss_columns].values, metric='euclidean')
    stats = {
        "PairwiseDistance_Mean": np.mean(pairwise_distances),
        "PairwiseDistance_Median": np.median(pairwise_distances),
        "PairwiseDistance_25%": np.percentile(pairwise_distances, 25),
        "PairwiseDistance_75%": np.percentile(pairwise_distances, 75),
        "PairwiseDistance_Std": np.std(pairwise_distances),
    }

    # Add std_dev and mean_abs_change statistics
    for metric in ['std_dev', 'mean_abs_change']:
        stats.update({
            f"{metric}_25%": np.percentile(df[metric], 25),
            f"{metric}_Mean": np.mean(df[metric]),
            f"{metric}_Median": np.median(df[metric]),
            f"{metric}_75%": np.percentile(df[metric], 75),
            f"{metric}_Std": np.std(df[metric]),
        })

    return df, stats

# Parallelizable function for processing a single experiment
@remote
def process_experiment(backbone, loss, dataset, config, world_size, project_name):
    statistics = {"dataset": dataset, "backbone": backbone, "loss_function": loss}

    config.dataset = dataset
    config.backbone = backbone
    config.loss_function = loss
    config.project_name = project_name

    set_seeds(config.seed)
    setup = Setup(config, world_size)
    experiment = BaseExperiment(setup)
    hp_search = Hp_opt(setup)
    trainer = hp_search.get_trainer(experiment.training)

    experiment_path = f"~/scratch/hpo_results/{setup.project_name}_{setup.exp_id}"

    try:
        restored_tuner = tune.Tuner.restore(experiment_path, trainable=trainer)
        result_grid = restored_tuner.get_results()

        df, stats = result_grid_analysis(result_grid)
        statistics.update(stats)
        return statistics
    except Exception as e:
        logging.error(f"Failed for {dataset}, {loss}, {backbone}: {e}")
        return None


# Main parallel execution
def main():
    world_size = torch.cuda.device_count()
    project_name = 'full_fine_tuning_5epochs_final1'

    datas = ('stanford_cars', 'caltech101', 'fgvc-aircraft-2013b', 'dtd', 'flowers-102', 'oxford-iiit-pet')
    losses = ('TRADES_v2', 'CLASSIC_AT')
    backbones = (
        'convnext_tiny', 'robust_convnext_tiny', 'convnext_tiny.fb_in22k',
        'deit_small_patch16_224.fb_in1k', 'robust_deit_small_patch16_224',
        'convnext_base', 'convnext_base.fb_in22k', 'robust_convnext_base',
        'convnext_base.clip_laion2b', 'convnext_base.clip_laion2b_augreg',
        'vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k',
        'vit_base_patch16_224.dino', 'vit_base_patch16_224.mae', 'vit_base_patch16_224.orig_in21k',
        'vit_base_patch16_224.sam_in1k', 'vit_base_patch16_224_miil.in21k'
    )

    config = OmegaConf.load("./configs/default_config.yaml")
    config = compose(config_name="default_config")  # Store Hydra config in a variable

    # Generate combinations of backbone, loss, and dataset
    combinations = [(b, l, d) for b in backbones for l in losses for d in datas]

    # Submit parallel tasks
    futures = [
        process_experiment.remote(backbone, loss, dataset, config, world_size, project_name)
        for backbone, loss, dataset in combinations
    ]

    # Gather results
    results = ray.get(futures)
    final_data = [res for res in results if res is not None]

    # Save results
    final_data_df = pd.DataFrame(final_data)
    final_data_df.to_csv(f"~/projects/def-adurand/mheuill/robust_training/RQ4_final_data_{project_name}.csv")


if __name__ == "__main__":
    main()