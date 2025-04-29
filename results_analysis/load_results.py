
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
        'edgenext_small': "hybrid",
        'coat_tiny': "hybrid", }


backbone_name={
    'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K':'convnext_b,clip,laiona',
    'CLIP-convnext_base_w-laion2B-s13B-b82K':'convnext_b,clip,laion2b',
    'deit_small_patch16_224.fb_in1k':'deit_s,sup,in1k',
    'robust_resnet50':'resnet50,rob-sup,in1k',
    'vit_small_patch16_224.augreg_in21k':'vit_s,sup,in22k',
    'convnext_base.fb_in1k':'convnext_b,sup,in1k',
    'resnet50.a1_in1k':'resnet50,sup,in1k',
    'robust_vit_base_patch16_224':'vit_b,rob-sup,in1k',
    'vit_base_patch16_224.mae':'vit_b,mae,in1k',
    'vit_small_patch16_224.dino':'vit_b,dino,in1k',
    'convnext_base.fb_in22k':'convnext_b,sup,in22k',

    'robust_convnext_base':'convnext_b,rob-sup,in1k',
    'vit_base_patch16_224.augreg_in1k':'vit_b,sup,in1k',
    'vit_base_patch16_224.augreg_in21k':'vit_b,sup,in22k',
    'vit_base_patch16_clip_224.laion2b':'vit_b,clip,laion2b',
    'convnext_tiny.fb_in1k':'convnext_t,sup,in1k',
    'robust_convnext_tiny':'convnext_t,rob-sup,in1k',
    'robust_deit_small_patch16_224':'deit_s,rob-sup,in1k',
    'vit_small_patch16_224.augreg_in1k':'vit_s,sup,in1k',
    'convnext_tiny.fb_in22k':'convnext_t,sup,in22k',
    'vit_base_patch16_clip_224.laion2b_ft_in1k':'vit_b,clip,laion2b',
    'vit_base_patch16_224.augreg_in21k_ft_in1k':'vit_b,sup,ink22k-in1k',

    'vit_small_patch16_224.augreg_in21k_ft_in1k':'vit_s,sup,ink22k-in1k',
    'eva02_base_patch14_224.mim_in22k':'eva02_b,mim,ink22k',
    'eva02_tiny_patch14_224.mim_in22k':'eva02_b,mim,ink22k',
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k':'swin_b,sup,ink22k-in1k',
    'swin_tiny_patch4_window7_224.ms_in1k':'swin_t,sup,in1k',
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k':'convnext_b,hybrid,laion2b-in12k-in1k',
    'convnext_base.fb_in22k_ft_in1k':'convnext_b,sup,in22k-in1k',
    'convnext_tiny.fb_in22k_ft_in1k':'convnext_t,sup,in22k-in1k',
    'coatnet_0_rw_224.sw_in1k':'coatnet_0,sup,in1k',
    'coatnet_2_rw_224.sw_in12k_ft_in1k':'coatnet_2,sup,in12k-in1k',
    'coatnet_2_rw_224.sw_in12k':'coatnet_2,sup,in12k',

    "regnetx_004.pycls_in1k":'regnetx_004,sup,in1k',
    'efficientnet-b0':'efficientnet_b0,sup,in1k',
    'deit_tiny_patch16_224.fb_in1k':'deit_t,sup,in1k',
    'mobilevit-small':'mobilevit_s,sup,in1k',
    'mobilenetv3_large_100.ra_in1k':'mobilenet_v3,sup,in1k',
    'edgenext_small.usi_in1k':'edgenetx_s,sup,in1k',
    'coat_tiny.in1k':'coat_t,sup,in1k', }


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

volume_pre_training_data = {
    "laion_aesthetic":900_000_000 ,
    "laion2B":2_320_000_000,
    "in1k":1_281_167,
    "in22k":14_197_122,
    "laion2b+in1k":2_320_000_000+1_281_167,
    "in22k+in1k":14_197_122+1_281_167,
    "laion2b+in12k+in1k":2_320_000_000+9_000_000+1_281_167,
    "in12k+in1k":9_000_000+1_281_167,
    "in12k":9_000_000,
}

pre_training_dataset = {
    'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K': "laion_aesthetic",
    'CLIP-convnext_base_w-laion2B-s13B-b82K':"laion2B",
    'deit_small_patch16_224.fb_in1k':"in1k",
    'robust_resnet50':"in1k",
    'vit_small_patch16_224.augreg_in21k':"in22k",
    'convnext_base.fb_in1k':"in1k",
    'resnet50.a1_in1k':"in1k",
    'robust_vit_base_patch16_224':"in1k",
    'vit_base_patch16_224.mae':"in1k",
    'vit_small_patch16_224.dino':"in1k",
    'convnext_base.fb_in22k':"in22k",
    'robust_convnext_base': "in1k",
    'vit_base_patch16_224.augreg_in1k':"in1k",
    'vit_base_patch16_224.augreg_in21k':"in22k",
    'vit_base_patch16_clip_224.laion2b':"laion2B",
    'convnext_tiny.fb_in1k':"in1k",
    'robust_convnext_tiny':"in1k",
    'robust_deit_small_patch16_224':"in1k",
    'vit_small_patch16_224.augreg_in1k':"in1k",
    'convnext_tiny.fb_in22k':"in22k",
    'vit_base_patch16_clip_224.laion2b_ft_in1k':"laion2b+in1k",
    'vit_base_patch16_224.augreg_in21k_ft_in1k':"in22k+in1k",
    'vit_small_patch16_224.augreg_in21k_ft_in1k':"in22k+in1k",
    'eva02_base_patch14_224.mim_in22k':"in22k",
    'eva02_tiny_patch14_224.mim_in22k':"in22k",
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k':"in22k+in1k",
    'swin_tiny_patch4_window7_224.ms_in1k':"in1k",
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k':"laion2b+in12k+in1k",
    'convnext_base.fb_in22k_ft_in1k':'in22k+in1k',
    'convnext_tiny.fb_in22k_ft_in1k':'in22k+in1k',
    'coatnet_0_rw_224.sw_in1k':"in1k",
    'coatnet_2_rw_224.sw_in12k_ft_in1k':"in12k+in1k",
    'coatnet_2_rw_224.sw_in12k':"in12k",
    "regnetx_004.pycls_in1k":"in1k",
    'efficientnet-b0':"in1k", 
    'deit_tiny_patch16_224.fb_in1k':"in1k",
    'mobilevit-small':"in1k",
    'mobilenetv3_large_100.ra_in1k':"in1k",
    'edgenext_small.usi_in1k':"in1k",
    'coat_tiny.in1k':"in1k", 
    "regnetx_004.pycls_in1k":"in1k",
    'efficientnet-b0':"in1k", 
    'deit_tiny_patch16_224.fb_in1k':"in1k",
    'mobilevit-small':"in1k",
    'mobilenetv3_large_100.ra_in1k':"in1k",
    'edgenext_small.usi_in1k':"in1k",
    'coat_tiny.in1k':"in1k", 
}


pre_training_strategy = {
    'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K': 'self-supervised (multimodal)',
    'CLIP-convnext_base_w-laion2B-s13B-b82K':'self-supervised (multimodal)',
    'deit_small_patch16_224.fb_in1k':'supervised',
    'robust_resnet50':'supervised (robust)',
    'vit_small_patch16_224.augreg_in21k':'supervised',
    'convnext_base.fb_in1k':'supervised',
    'resnet50.a1_in1k':'supervised',
    'robust_vit_base_patch16_224':'supervised (robust)',
    'vit_base_patch16_224.mae':'self-supervised',
    'vit_small_patch16_224.dino':'self-supervised',
    'convnext_base.fb_in22k':'supervised',
    'robust_convnext_base': 'supervised (robust)',
    'vit_base_patch16_224.augreg_in1k':'supervised',
    'vit_base_patch16_224.augreg_in21k':'supervised',
    'vit_base_patch16_clip_224.laion2b':'self-supervised (multimodal)',
    'convnext_tiny.fb_in1k':'supervised',
    'robust_convnext_tiny':'supervised (robust)',
    'robust_deit_small_patch16_224':'supervised (robust)',
    'vit_small_patch16_224.augreg_in1k':'supervised',
    'convnext_tiny.fb_in22k':'supervised',
    'vit_base_patch16_clip_224.laion2b_ft_in1k':'hybrid',
    'vit_base_patch16_224.augreg_in21k_ft_in1k':'supervised',
    'vit_small_patch16_224.augreg_in21k_ft_in1k':'supervised',
    'eva02_base_patch14_224.mim_in22k':'self-supervised',
    'eva02_tiny_patch14_224.mim_in22k':'self-supervised',
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k':'supervised',
    'swin_tiny_patch4_window7_224.ms_in1k':'supervised',
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k':'hybrid',
    'convnext_base.fb_in22k_ft_in1k':'supervised',
    'convnext_tiny.fb_in22k_ft_in1k':'supervised',
    'coatnet_0_rw_224.sw_in1k':'supervised',
    'coatnet_2_rw_224.sw_in12k_ft_in1k':'supervised',
    'coatnet_2_rw_224.sw_in12k':'supervised',
    "regnetx_004.pycls_in1k":'supervised',
    'efficientnet-b0':'supervised',
    'deit_tiny_patch16_224.fb_in1k':'supervised',
    'mobilevit-small':'supervised',
    'mobilenetv3_large_100.ra_in1k':'supervised',
    'edgenext_small.usi_in1k':'supervised',
    'coat_tiny.in1k':'supervised',
    "regnetx_004.pycls_in1k":'supervised',
    'efficientnet-b0':'supervised',
    'deit_tiny_patch16_224.fb_in1k':'supervised',
    'mobilevit-small':'supervised',
    'mobilenetv3_large_100.ra_in1k':'supervised',
    'edgenext_small.usi_in1k':'supervised',
    'coat_tiny.in1k':'supervised',
}




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
                    project_name = pn1
                else:
                    project_name = pn2

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

                    try: 
                        project_name = pn2

                        name ='{}_{}_{}'.format(backbone, data, loss)
                        print( '../results/{}/{}.pkl'.format(project_name, name) )
                        with open('../results/{}/{}.pkl'.format(project_name, name), 'rb') as f:
                            result = pickle.load(f)

                        arith_sum, geom_sum = sums_from_dict(result)
                        result['sum'] = arith_sum
                        result['geom'] = geom_sum

                    except:
                                    
                        result = {'clean_acc': math.nan, 'Linf_acc': math.nan, 'L2_acc': math.nan, 'L1_acc': math.nan, 'common_acc': math.nan, 
                                'sum':math.nan, 'geom':math.nan, }
                    
                for key, value in model_parameters.items():
                    if key in backbone:  # Match the model name in the backbone string
                        if value < 20:
                            result['model_size'] = 'small'
                        elif value < 50:
                            result['model_size'] = 'medium'
                        else:
                            result['model_size'] = 'large'

                        break

                result['backbone'] = backbone
                result['dataset'] = data
                result['loss_function'] = loss
                result['pre_training_dataset'] = pre_training_dataset[backbone]
                result['pre_training_strategy'] = pre_training_strategy[backbone]
                result['volume_pre_training_data'] = volume_pre_training_data[ pre_training_dataset[backbone] ]
                result["backbone_name"] = backbone_name[backbone]
                
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
