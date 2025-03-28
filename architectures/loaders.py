import torch

import timm
import torch.nn as nn
import os
import open_clip
from transformers import AutoModel
# from utils import get_state_dict_dir
from architectures.clip_wrapper import CLIPConvNeXtClassifier

class HFModelWrapper(nn.Module):
    def __init__(self, hf_model, num_classes, backbone):
        super().__init__()
        self.hf_model = hf_model
        # Suppose you want your final classification layer to be called `fc`
        if "efficientnet" in backbone:
            in_features = 1280
        elif "mobilevit" in backbone:
            in_features = 640
        # else: maybe other backbones

        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 1) Run the original HF model
        outputs = self.hf_model(x)  
        # 2) For EfficientNet, use outputs.pooler_output or last_hidden_state[-1], etc.
        #    We'll pick the pooler_output for demonstration:
        pooled = outputs.pooler_output
        # 3) Apply your classification head
        logits = self.head(pooled)
        return logits


def load_architecture(config, N):
    backbone = config.backbone
    print('BACKBONE NAME', backbone)
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
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True, )

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']  # Handle wrapped checkpoints

        model.load_state_dict(state_dict, strict=False)

        # New: Hugging Face Models (Google, Apple, and others)
    elif backbone == 'efficientnet-b0' or backbone == 'mobilevit-small':

            # Determine if it's an OpenCLIP model
        hf_models = {
        'efficientnet-b0': 'google/efficientnet-b0',
        'mobilevit-small': 'apple/mobilevit-small',
            }

        print(f"Loading Hugging Face model: {backbone}")

        try:
            model = AutoModel.from_pretrained(hf_models[backbone], cache_dir=statedict_dir)  # Load model from HF
        except Exception as e:
            print(f"❌ Failed to load {backbone} from Hugging Face: {e}")
            return None
        checkpoint_path = os.path.join(statedict_dir, f'{backbone}.pt')
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True, )
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

    # Replace classification head 
    model = change_head(backbone, model, N)
    # Ensure all models have a `.head` attribute for classification
    rename_head_layer(model)

    return model

def rename_head_layer(model):
    """ Rename classification layers like `.fc` or `.classifier` to `.head` """
    if hasattr(model, "fc"):
        model.head = model.fc  # Rename `.fc` to `.head`
        del model.fc  # Remove original reference
    elif hasattr(model, "classifier"):
        model.head = model.classifier  # Rename `.classifier` to `.head`
        del model.classifier  # Remove original reference


def change_head(backbone, model, N):

    if 'CLIP-convnext' in backbone:
        model = CLIPConvNeXtClassifier(model, num_classes=N)

    elif "convnext" in backbone or 'swin' in backbone or 'regnetx' in backbone or 'edgenext' in backbone: 
        num_features = model.head.fc.in_features
        model.head.fc = nn.Linear(num_features, N)
    
    elif "efficientnet" in backbone:
        model = HFModelWrapper(model, N, backbone)
    
    elif "mobilevit" in backbone:
        model = HFModelWrapper(model, N, backbone)

    elif 'mobilenetv3' in backbone:
        model.classifier = nn.Linear(model.classifier.in_features, N)

    elif "resnet" in backbone:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N)

    elif "deit" in backbone:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, N)

    elif "vit" in backbone: 

        if isinstance(model.head, nn.Identity):
            num_features = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_features, N)
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, N)

    elif "eva02" in backbone: 
        print("hey")
        if isinstance(model.head, nn.Identity):
            num_features = model.fc_norm.normalized_shape[0]
            print(num_features)
            model.head = nn.Linear(num_features, N)
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, N)

    else:
        print("not implemented error")


    return model



