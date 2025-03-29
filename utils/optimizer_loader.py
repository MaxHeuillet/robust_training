from torch.optim import AdamW
import torch.nn as nn
from architectures import CustomModel

def load_optimizer(config, model):
    backbone = config.backbone
    lr1 = config.lr1
    lr2 = config.lr2
    weight_decay1 = config.weight_decay1
    weight_decay2 = config.weight_decay2

    if isinstance(model, CustomModel):
        model = model.base_model

    def get_param_groups(model, head_module=None):
        decay, no_decay, head_decay, head_no_decay = [], [], [], []

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Check if this param belongs to the head module
                is_head = head_module is not None and any(param is p for p in head_module.parameters())

                if is_head:
                    if "bias" in name or "norm" in name or "bn" in name:
                        head_no_decay.append(param)
                    else:
                        head_decay.append(param)
                else:
                    if "bias" in name or "norm" in name or "bn" in name:
                        no_decay.append(param)
                    else:
                        decay.append(param)

        return decay, no_decay, head_decay, head_no_decay
    
    # Handling ViTs separately because they may not have an explicit "classifier" module
    if 'deit' in backbone or 'vit' in backbone or 'eva02' in backbone:
        head_module = model.head #if isinstance(model.head, nn.Linear) else None  # Only assign if head is a classifier
    elif hasattr(model, 'classifier'):
        head_module = model.classifier  # Most CNN models
    elif hasattr(model, 'fc'):
        head_module = model.fc  # ResNets
    elif hasattr(model, 'head') and hasattr(model.head, 'fc'):
        head_module = model.head.fc  # ConvNeXt, Swin, RegNetX
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        head_module = model.head
    else:
        head_module = None  # Fallback case (should not happen in supported architectures)

    # Now correctly get parameter groups
    decay, no_decay, head_decay, head_no_decay = get_param_groups(model, head_module)

    print(len(decay), len(no_decay), len(head_decay), len(head_no_decay))   

    optimizer = AdamW([
        # Backbone parameters with weight decay
        {
            'params': decay,
            'lr': lr1,
            'weight_decay': weight_decay1,
            'betas': (0.9, 0.95),
            'name': 'backbone_decay'
        },
        # Backbone parameters without weight decay
        {
            'params': no_decay,
            'lr': lr1,
            'weight_decay': 0.0,
            'betas': (0.9, 0.95),
            'name': 'backbone_no_decay'
        },
        # Head parameters with weight decay
        {
            'params': head_decay,
            'lr': lr2,
            'weight_decay': weight_decay2,
            'betas': (0.9, 0.95),
            'name': 'head_decay'
        },
        # Head parameters without weight decay
        {
            'params': head_no_decay,
            'lr': lr2,
            'weight_decay': 0.0,
            'betas': (0.9, 0.95),
            'name': 'head_no_decay'
        },
    ])

    return optimizer


# def load_optimizer(config, model):

#     backbone = config.backbone
#     lr1 = config.lr1
#     lr2 = config.lr2
#     weight_decay1 = config.weight_decay1
#     weight_decay2 = config.weight_decay2

#     def get_param_groups(model, head_module=None):
#         decay, no_decay, head_decay, head_no_decay = [], [], [], []

#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue

#             # Check if this param belongs to the head module
#             is_head = head_module is not None and any(param is p for p in head_module.parameters())

#             if is_head:
#                 if "bias" in name or "norm" in name or "bn" in name:
#                     head_no_decay.append(param)
#                 else:
#                     head_decay.append(param)
#             else:
#                 if "bias" in name or "norm" in name or "bn" in name:
#                     no_decay.append(param)
#                 else:
#                     decay.append(param)

#         return decay, no_decay, head_decay, head_no_decay
    
#     # if 'resnet' in backbone:
#     #     decay, no_decay, head_decay, head_no_decay = get_param_groups(model, model.fc)
#     # else:
#     ### TODO
#     decay, no_decay, head_decay, head_no_decay = get_param_groups(model, model.classifier)

#     print(len(decay), len(no_decay), len(head_decay), len(head_no_decay))   

#     optimizer = AdamW([
#         # Backbone parameters with weight decay
#         {
#             'params': decay,
#             'lr': lr1,
#             'weight_decay': weight_decay1,
#             'betas': (0.9, 0.95),
#             'name': 'backbone_decay'
#         },
#         # Backbone parameters without weight decay
#         {
#             'params': no_decay,
#             'lr': lr1,
#             'weight_decay': 0.0,
#             'betas': (0.9, 0.95),
#             'name': 'backbone_no_decay'
#         },
#         # Head parameters with weight decay
#         {
#             'params': head_decay,
#             'lr': lr2,
#             'weight_decay': weight_decay2,
#             'betas': (0.9, 0.95),
#             'name': 'head_decay'
#         },
#         # Head parameters without weight decay
#         {
#             'params': head_no_decay,
#             'lr': lr2,
#             'weight_decay': 0.0,
#             'betas': (0.9, 0.95),
#             'name': 'head_no_decay'
#         },
#     ])

#     return optimizer






