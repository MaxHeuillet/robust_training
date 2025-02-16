from torch.optim import AdamW
import torch.nn as nn

def load_optimizer(config, model):

    backbone = config.backbone
    lr1 = config.lr1
    lr2 = config.lr2
    weight_decay1 = config.weight_decay1
    weight_decay2 = config.weight_decay2

    def get_param_groups(model, head_module_name):
        decay = []
        no_decay = []
        head_decay = []
        head_no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            # Identify if the parameter is part of the head
            if head_module_name in name:
                # Parameters in the head
                if name.endswith(".bias") or isinstance(getattr(model, name.split('.')[0]), nn.BatchNorm2d) or "bn" in name or "norm" in name:
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)
            else:
                # Parameters in the backbone
                if name.endswith(".bias") or isinstance(getattr(model, name.split('.')[0]), nn.BatchNorm2d) or "bn" in name or "norm" in name:
                    no_decay.append(param)
                else:
                    decay.append(param)

        return decay, no_decay, head_decay, head_no_decay

    # Determine the head module name based on backbone
    if "convnext" in backbone:
        head_module_name = "head.fc"
    elif "wideresnet" in backbone:
        head_module_name = "logits"
    elif "deit" in backbone or "vit" in backbone:
        head_module_name = "head"
    # else:
    #     head_module_name = "head"

    # Get parameter groups
    decay, no_decay, head_decay, head_no_decay = get_param_groups(model, head_module_name)

    print(len(decay), len(no_decay), len(head_decay), len(head_no_decay))   

    optimizer = AdamW([
        # Backbone parameters with weight decay
        {
            'params': decay,
            'lr': lr1,
            'weight_decay': weight_decay1,
            'betas': (0.9, 0.95)
        },
        # Backbone parameters without weight decay
        {
            'params': no_decay,
            'lr': lr1,
            'weight_decay': 0.0,
            'betas': (0.9, 0.95)
        },
        # Head parameters with weight decay
        {
            'params': head_decay,
            'lr': lr2,
            'weight_decay': weight_decay2,
            'betas': (0.9, 0.95)
        },
        # Head parameters without weight decay
        {
            'params': head_no_decay,
            'lr': lr2,
            'weight_decay': 0.0,
            'betas': (0.9, 0.95)
        },
    ])

    return optimizer






