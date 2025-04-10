import torch

import timm
import torch.nn as nn
import os
import open_clip
from transformers import AutoModel
# from utils import get_state_dict_dir
from architectures.clip_wrapper import CLIPConvNeXtClassifier
from transformers import AutoConfig, AutoModel
from pathlib import Path  # In case config.dataset_path is a string


class HFModelWrapper(nn.Module):
    def __init__(self, hf_model, num_classes, backbone):
        super().__init__()
        self.hf_model = hf_model
        # Suppose you want your final classification layer to be called `fc`
        if "efficientnet" in backbone:
            in_features = hf_model.pooler.kernel_size
        elif "mobilevit" in backbone:
            # in_features = 640
            in_features = hf_model.conv_1x1_exp.convolution.out_channels

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
    statedict_dir =  Path(os.path.expandvars(config.work_path)).expanduser().resolve()
    
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
    elif backbone == 'efficientnet-b0' or backbone == 'mobilevit-small' or backbone == 'cvt-21':

            # Determine if it's an OpenCLIP model
        hf_models = {
        'efficientnet-b0': 'google/efficientnet-b0',
        'mobilevit-small': 'apple/mobilevit-small',  }

        print(f"Loading Hugging Face model: {backbone}")

        config = AutoConfig.from_pretrained( os.path.join(statedict_dir, backbone) )
        model = AutoModel.from_config(config)

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
    # model = change_head(backbone, model, N)
    # model = rename_head_layer(model)
    model = replace_classifier(backbone, model, N)

    return model



class StandardClassifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.1):
        super().__init__()  # Call the parent constructor
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def replace_classifier(backbone, model, N):
    """ Replaces the model's classification head with a standardized classifier. """

    if 'CLIP-convnext' in backbone:
        in_features = model.visual.head.proj.in_features
        model = model.visual.trunk
        model.head.fc = nn.Linear(in_features, N, bias=True) 
    
    elif 'resnet' in backbone:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, N, bias=True)

    elif 'regnetx' in backbone:
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, N, bias=True)

    elif 'mobilevit' in backbone or 'efficientnet' in backbone:
        model = HFModelWrapper(model, N, backbone)

    elif 'mobilenetv3' in backbone:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, N, bias=True) 

    elif 'edgenext' in backbone:
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, N, bias=True)

    elif 'coatnet' in backbone:
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, N, bias=True)
    
    elif 'coat' in backbone:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, N, bias=True)

    elif 'deit' in backbone or 'vit' in backbone or 'eva02' in backbone:
        if isinstance(model.head, nn.Identity) and 'eva02' in backbone:
            num_features = model.fc_norm.normalized_shape[0]
            model.head = nn.Linear(num_features, N)
        elif isinstance(model.head, nn.Identity):
            num_features = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_features, N)
        else:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, N, bias=True) #StandardClassifier(in_features, N)


    elif 'swin' in backbone:
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, N, bias=True)

    elif 'convnext' in backbone:
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, N, bias=True) 

    return model


# import torch.nn as nn

# def replace_fc_layer(model, fc_path, in_features, N):
#     """Helper function to replace a classifier layer at the specified path."""
#     fc_parent, fc_attr = fc_path.rsplit(".", 1)  # Split the path
#     parent_module = eval(f"model.{fc_parent}")  # Get the parent module
#     setattr(parent_module, fc_attr, nn.Linear(in_features, N, bias=True))  # Replace FC layer

# def replace_classifier(backbone, model, N):
#     """Replaces the model's classification head with a standardized classifier."""
    
#     if 'CLIP-convnext' in backbone:
#         in_features = model.visual.head.proj.in_features
#         model = model.visual.trunk  # Extract the trunk for modification
#         replace_fc_layer(model, "head.fc", in_features, N)

#     elif 'resnet' in backbone:
#         replace_fc_layer(model, "fc", model.fc.in_features, N)

#     elif 'regnetx' in backbone or 'swin' in backbone or 'convnext' in backbone:
#         replace_fc_layer(model, "head.fc", model.head.fc.in_features, N)

#     elif 'mobilevit' in backbone or 'efficientnet' in backbone:
#         model = HFModelWrapper(model, N, backbone)  # Use a wrapper for these architectures

#     elif 'mobilenetv3' in backbone:
#         replace_fc_layer(model, "classifier", model.classifier.in_features, N)

#     elif 'edgenext' in backbone:
#         replace_fc_layer(model, "head.fc", model.head.fc.in_features, N)

#     elif 'deit' in backbone or 'vit' in backbone or 'eva02' in backbone:
#         if isinstance(model.head, nn.Identity):
#             num_features = model.fc_norm.normalized_shape[0] if 'eva02' in backbone else model.norm.normalized_shape[0]
#             model.head = nn.Linear(num_features, N)
#         else:
#             replace_fc_layer(model, "head", model.head.in_features, N)

#     return model



# def replace_classifier(backbone, model, N):
#     """ Replaces the model's classification head with a standardized classifier. """

#     if 'CLIP-convnext' in backbone:

#         in_features = model.visual.head.proj.in_features
#         model = model = model.visual.trunk

#         # pool = model.head.global_pool
#         # norm = model.head.norm
#         # flatten = model.head.flatten

#         # model.global_pool = pool
#         # model.norm = norm
#         # model.flatten = flatten
#         model.head.fc = nn.Linear(in_features, N, bias=True) #StandardClassifier(in_features, N)

        
    
#     elif 'resnet' in backbone:

#         in_features = model.fc.in_features
#         model.fc = nn.Linear(in_features, N, bias=True)#StandardClassifier(in_features,N)
#         # del model.fc
    
#     elif 'regnetx' in backbone:

#         in_features = model.head.fc.in_features
#         # pool = model.head.global_pool
#         # model.global_pool = pool
#         model.head.fc = nn.Linear(in_features, N, bias=True)#StandardClassifier(in_features, N)
#         # del model.head

#     # elif 'efficientnet' in backbone:
#     #     model = HFModelWrapper(model, N, backbone)
#         # model.fc = nn.Linear(in_features, N, bias=True)#StandardClassifier(in_features, N)

#     elif 'mobilevit' in backbone or 'efficientnet' in backbone:
#         model = HFModelWrapper(model, N, backbone)
#         # in_features = model.conv_1x1_exp.convolution.out_channels
#         # model.classifier = StandardClassifier(in_features, N)
    
#     elif 'mobilenetv3' in backbone:
#         in_features = model.classifier.in_features
#         model.classifier = StandardClassifier(in_features, N)

#     elif 'edgenext' in backbone:

#         in_features = model.head.fc.in_features
#         model.head.fc = nn.Linear(in_features, N, bias=True)


#     elif 'deit' in backbone or 'vit' in backbone or 'eva02' in backbone:
#         if isinstance(model.head, nn.Identity) and 'eva02' in backbone:
#             num_features = model.fc_norm.normalized_shape[0]
#             model.head = nn.Linear(num_features, N)
#         elif isinstance(model.head, nn.Identity):
#             num_features = model.norm.normalized_shape[0]
#             model.head = nn.Linear(num_features, N)
#         else:
#             in_features = model.head.in_features
#             model.head = nn.Linear(in_features, N, bias=True) #StandardClassifier(in_features, N)

#     elif 'swin' in backbone:
#         in_features = model.head.fc.in_features
#         model.head.fc = nn.Linear(in_features, N, bias=True) #StandardClassifier(in_features, N)

#     elif 'convnext' in backbone:
#         in_features = model.head.fc.in_features
#         model.head.fc = nn.Linear(in_features, N, bias=True) #StandardClassifier(in_features, N)


#     return model
    



# def change_head(backbone, model, N):

#     if 'CLIP-convnext' in backbone:
#         model = CLIPConvNeXtClassifier(model, num_classes=N)

#     elif "convnext" in backbone or 'swin' in backbone or 'regnetx' in backbone or 'edgenext' in backbone: 
#         num_features = model.head.fc.in_features
#         model.head.fc = nn.Linear(num_features, N)
    
#     elif "efficientnet" in backbone:
#         model = HFModelWrapper(model, N, backbone)
    
#     elif "mobilevit" in backbone:
#         model = HFModelWrapper(model, N, backbone)

#     elif 'mobilenetv3' in backbone:
#         model.classifier = nn.Linear(model.classifier.in_features, N)

#     elif "resnet" in backbone:
#         num_features = model.fc.in_features
#         model.fc = nn.Linear(num_features, N)

#     elif "deit" in backbone:
#         num_features = model.head.in_features
#         model.head = nn.Linear(num_features, N)

#     elif "vit" in backbone: 

#         if isinstance(model.head, nn.Identity):
#             num_features = model.norm.normalized_shape[0]
#             model.head = nn.Linear(num_features, N)
#         num_features = model.head.in_features
#         model.head = nn.Linear(num_features, N)

#     elif "eva02" in backbone: 
#         print("hey")
#         if isinstance(model.head, nn.Identity):
#             num_features = model.fc_norm.normalized_shape[0]
#             print(num_features)
#             model.head = nn.Linear(num_features, N)
#         num_features = model.head.in_features
#         model.head = nn.Linear(num_features, N)

#     else:
#         print("not implemented error")


#     return model