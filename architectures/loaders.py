import torch

import timm
import torch.nn as nn
import os
import open_clip

# from utils import get_state_dict_dir
from architectures.clip_wrapper import CLIPConvNeXtClassifier


def load_architecture(config, N):
    backbone = config.backbone
    statedict_dir = os.path.abspath(os.path.expanduser(config.statedicts_path)) #get_state_dict_dir(config)

    # Determine if it's an OpenCLIP model
    openclip_models = {
        'CLIP-convnext_base_w-laion2B-s13B-b82K': 'convnext_base_w',
        'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K': 'convnext_base_w',
    }

    if backbone in openclip_models:
        
        model_name = openclip_models[backbone]
        print('Loading OpenCLIP model:', model_name)

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None  # you're loading your own weights
        )

        checkpoint_path = os.path.join(statedict_dir, f'{backbone}.pt')
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']  # Handle wrapped checkpoints

        model.load_state_dict(state_dict, strict=False)

    else:
        # Use timm model
        equivalencies = {
            'robust_convnext_base': 'convnext_base',
            'robust_convnext_tiny': 'convnext_tiny',
            'robust_resnet50': 'resnet50',
            'robust_deit_small_patch16_224': 'deit_small_patch16_224',
            'robust_vit_base_patch16_224': 'vit_base_patch16_224',
        }

        model_name = equivalencies.get(backbone, backbone)
        print('Loading timm model:', model_name)

        model = timm.create_model(model_name, pretrained=False)

        checkpoint_path = os.path.join(statedict_dir, f'{backbone}.pt')
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        model.load_state_dict(state_dict)

    # Replace classification head if needed
    model = change_head(backbone, model, N)

    return model



def change_head(backbone, model, N):

    if 'CLIP-convnext' in backbone:
        model = CLIPConvNeXtClassifier(model, num_classes=N)

    elif "convnext" in backbone or 'swinv2' in backbone: 
        num_features = model.head.fc.in_features
        model.head.fc = nn.Linear(num_features, N)

    elif "resnet" in backbone:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N)

    elif "deit" in backbone:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, N)

    elif "vit" in backbone: 

        if isinstance(model.head, nn.Identity):
            num_features = 768
            model.head = nn.Linear(num_features, N)

        num_features = model.head.in_features
        model.head = nn.Linear(num_features, N)


    return model



