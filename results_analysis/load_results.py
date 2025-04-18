
import pickle
# import omegaconf
import warnings
import pickle
import numpy as np
import math
from typing import Mapping, Tuple
warnings.filterwarnings("ignore", category=DeprecationWarning)

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


    # Two backbone groups
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

def load_result_dataset(pn1, pn2,):

    final_data = []

    for loss in losses:
        for data in datas:
            for backbone in backbones:

                if backbone in yann_backbones:
                    project_name = pn1#'full_fine_tuning_50epochs_edge_paper_final2'
                else:
                    project_name = pn2#'full_fine_tuning_50epochs_paper_final2'

                try:
                    name ='{}_{}_{}'.format(backbone, data, loss)
                    print( '../results/{}/{}.pkl'.format(project_name, name) )
                    with open('../results/{}/{}.pkl'.format(project_name, name), 'rb') as f:
                        result = pickle.load(f)
                                        # result = saved_data[name]["statistics"]

                    arith_sum, geom_sum = sums_from_dict(result)
                    result['sum'] = arith_sum
                    result['geom'] = geom_sum

                except:
                    
                    result = {'clean_acc': math.nan, 'Linf_acc': math.nan, 'L2_acc': math.nan, 'L1_acc': math.nan, 'common_acc': math.nan, 
                            'sum':math.nan, 'geom':math.nan, }
                    
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
                final_data.append(result)

    return final_data
