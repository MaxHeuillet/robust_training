

import pickle
# import omegaconf
import warnings
import pickle
import numpy as np
import math
from typing import Mapping, Tuple
from hydra import initialize, compose
from omegaconf import OmegaConf
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

datas=( 'uc-merced-land-use-dataset',
        'stanford_cars', 
        'caltech101', 
        'fgvc-aircraft-2013b', 
        'flowers-102',
            'oxford-iiit-pet'  ) #'dtd',

losses=( 'TRADES_v2', 'CLASSIC_AT' ) # 

model_parameters = {
        'convnext_base': 86.0,
        'convnext_tiny': 28.0,
        'deit_small': 22.0,
        'vit_base': 86.0,
        'vit_small': 22.0,
        'resnet50': 25.0,
        'eva02_base': 78.0,
        'eva02_tiny': 24.0,
        'swin_base': 88.0,
        'swin_tiny': 28.0,
        'coatnet_0': 33.0,
        'coatnet_2': 77.0,
        'regnetx_004': 8.0,
        'efficientnet-b0': 5.3, 
        'deit_tiny': 5.0,
        'mobilevit-small': 7.0,
        'mobilenetv3': 5.4,  
        'edgenext_small': 8.0,  
        'coat_tiny': 12.0, }

model_type = {
        'convnext_base': "fully convolutional",
        'convnext_tiny': "fully convolutional",
        'deit_small': "fully attention",
        'vit_base': "fully attention",
        'vit_small': "fully attention",
        'resnet50': "fully convolutional",
        'eva02_base': "fully attention",
        'eva02_tiny': "fully attention",
        'swin_base': "fully attention",
        'swin_tiny': "fully attention",
        'coatnet_0': "hybrid",
        'coatnet_2': "hybrid",
        'regnetx_004': "fully convolutional",
        'efficientnet-b0': "fully convolutional", 
        'deit_tiny': "fully attention",
        'mobilevit-small': "hybrid",
        'mobilenetv3': "fully convolutional",
        'edgenext_small': "fully convolutional",
        'coat_tiny': "hybrid", }

backbones=(
    'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
    'CLIP-convnext_base_w-laion2B-s13B-b82K',
    'deit_small_patch16_224.fb_in1k',
    'robust_resnet50',
    'vit_small_patch16_224.augreg_in21k',
    'convnext_base.fb_in1k',
    'resnet50.a1_in1k',
    'robust_vit_base_patch16_224',
    'vit_base_patch16_224.mae',
    'vit_small_patch16_224.dino',
    'convnext_base.fb_in22k',

    'robust_convnext_base',
    'vit_base_patch16_224.augreg_in1k',
    'vit_base_patch16_224.augreg_in21k',
    'vit_base_patch16_clip_224.laion2b',
    'convnext_tiny.fb_in1k',
    'robust_convnext_tiny',
    'robust_deit_small_patch16_224',
    'vit_small_patch16_224.augreg_in1k',
    'convnext_tiny.fb_in22k',
    'vit_base_patch16_clip_224.laion2b_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',

    'vit_small_patch16_224.augreg_in21k_ft_in1k',
    'eva02_base_patch14_224.mim_in22k',
    'eva02_tiny_patch14_224.mim_in22k',
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_tiny_patch4_window7_224.ms_in1k',
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',
    'convnext_base.fb_in22k_ft_in1k',
    'convnext_tiny.fb_in22k_ft_in1k',
    'coatnet_0_rw_224.sw_in1k',
    'coatnet_2_rw_224.sw_in12k_ft_in1k',
    'coatnet_2_rw_224.sw_in12k',

    "regnetx_004.pycls_in1k",
    'efficientnet-b0', 
    'deit_tiny_patch16_224.fb_in1k',
    'mobilevit-small',
    'mobilenetv3_large_100.ra_in1k',
    'edgenext_small.usi_in1k',
    'coat_tiny.in1k', )


yann_backbones = (
    "regnetx_004.pycls_in1k",
    'efficientnet-b0', 
    'deit_tiny_patch16_224.fb_in1k',
    'mobilevit-small',
    'mobilenetv3_large_100.ra_in1k',
    'edgenext_small.usi_in1k',
    'coat_tiny.in1k', )

def sums_from_dict(scores):

    values = []
    for v in scores.values():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return math.nan, math.nan
        values.append(float(v))

    arith_sum = sum(values)
    geom_sum  = math.prod(values)
    return arith_sum, geom_sum

def get_config_id(cfg) -> str:
    # Join the values into a string
    serialized_values = cfg.backbone + '_' + cfg.dataset + '_' + cfg.loss_function
    print('serialized_values', serialized_values)
    return serialized_values

initialize(config_path="../configs", version_base=None)


def load_hpo_result_dataset(pn1, pn2,):

    final_data = []

    for loss in losses:
        for data in datas:
            for backbone in backbones:

                if 'linearprobe_50epochs' in pn2:
                    config_base = compose(config_name="default_config_linearprobe50")
                elif 'full_fine_tuning_5epochs' in pn2:
                    config_base = compose(config_name="default_config_fullfinetuning5")
                elif 'full_fine_tuning_50epochs' in pn2:
                    config_base = compose(config_name="default_config_fullfinetuning50")
                else:
                    print('error in the experiment name', flush=True)

                config_base.dataset = data
                config_base.backbone = backbone
                config_base.loss_function = loss

                if backbone in yann_backbones:
                    project_name = pn1
                    config_base.project_name = pn1
                else:
                    project_name = pn2
                    config_base.project_name = pn2

                config_base.exp_id = get_config_id(config_base)

                print(config_base.configs_path, config_base.project_name, config_base.exp_id)

                path = os.path.join("/Users/maximeheuillet/Desktop/robust_training/configs",
                                            "HPO_results",
                                            config_base.project_name, 
                                            f"{config_base.exp_id}.yaml")
                print(path)

                try:        
                    config_optimal = OmegaConf.load(path)
                except:
                    print('file not found')

                result = {      'nb_trials': config_optimal.nb_completed_trials,
                                'lr1': config_optimal.lr1,
                                'lr2': config_optimal.lr2,
                                'wd1': config_optimal.weight_decay1,
                                'wd2': config_optimal.weight_decay2,
                                'sched': config_optimal.scheduler }

                    
                for key, value in model_parameters.items():
                    if key in backbone:  # Match the model name in the backbone string
                        if value < 20:
                            result['model_size'] = 0 
                        elif value < 50:
                            result['model_size'] = 1
                        else:
                            result['model_size'] = 2

                        break

                result['backbone'] = backbone
                result['dataset'] = data
                result['loss_function'] = loss
                
                # ── set model_type ────────────────────────────────────────────────────────────
                for key, mtype in model_type.items():
                    if key in backbone:         # e.g. "convnext_base" in "convnext_base.fb_in22k"
                        result['model_type'] = mtype
                        break
                else:
                    # executed only if the loop ends without 'break'
                    result['model_type'] = "unknown"

                final_data.append(result)

    return final_data



# import os
# from ray import tune
# from ray.tune.examples.mnist_pytorch import train_mnist
# from hydra import initialize, compose
# from omegaconf import OmegaConf
# from databases import load_data
# from architectures import load_architecture
# from utils import load_optimizer, get_args2, set_seeds, Hp_opt
# import torch
# initialize(config_path="configs", version_base=None)
# from utils import Setup
# from distributed_experiment_final import BaseExperiment
# world_size = torch.cuda.device_count()
# full_path = os.path.abspath("./hpo_results")
# project_name = 'test30'
# for dataset in ['Aircraft', 'Flowers', 'Imagenette' ]:

#     for loss in ['CLASSIC_AT', 'TRADES_v2']:

#         for backbone in [ 'convnext_tiny.fb_in22k', 'convnext_tiny', 'robust_convnext_tiny', ]:
            
#             config = OmegaConf.load("./configs/default_config.yaml")
#             config = compose(config_name="default_config")  

#             config.dataset = dataset
#             config.backbone = backbone
#             config.loss_function = loss
#             config.project_name = project_name

#             set_seeds(config.seed)

#             setup = Setup(config, world_size)
#             experiment = BaseExperiment(setup)
            
#             hp_search = Hp_opt(setup)
#             trainer = hp_search.get_trainer(experiment.training)

#             tuner = tune.Tuner.restore('{}/{}_{}_{}_{}'.format(full_path, project_name, backbone, dataset, loss), trainable=trainer)

#             result_grid = tuner.get_results()

#             print()

#             print( result_grid.get_best_result() )

#             print(result_grid)

#             print()

#             ax = None
#             for result in result_grid:

#                 if not result.metrics_dataframe.empty:
#                     print(result.metrics_dataframe)
#                     res = result.config['train_loop_config']
#                     # print(res)
#                     print()
#                     label = f"lr1={res['lr1']:.3f}, lr2={res['lr2']}"
#                     if ax is None:
#                         ax = result.metrics_dataframe.plot("training_iteration", "loss", label=label)
#                     else:
#                         result.metrics_dataframe.plot("training_iteration", "loss", ax=ax, label=label)
                        
#             ax.set_title("Loss vs. Training Iteration for All Trials")
#             ax.set_ylabel("Loss")

#             ax.figure.savefig("./results/{}_{}_{}_{}.png".format(project_name, backbone, dataset, loss), dpi=300)