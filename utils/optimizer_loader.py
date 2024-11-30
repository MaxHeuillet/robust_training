from torch.optim import AdamW

def load_optimizer(config, model):

    backbone = config.backbone
    lr1 = config.lr1
    lr2 = config.lr2
    weight_decay = config.weight_decay

    if "convnext" in backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "head.fc" not in n],
                'lr': lr1,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)

            },
            {
                'params': model.head.fc.parameters(),
                'lr': lr2,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)
            },
        ])
    elif "wideresnet" in backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "logits" not in n],
                'lr': lr1,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)
            },
            {
                'params': model.logits.parameters(),
                'lr': lr2,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)
            },
        ])
    elif "deit" in backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "head" not in n],
                'lr': lr1,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)
            },
            {
                'params': model.head.parameters(),
                'lr': lr2,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)
            },
        ])
    elif "vit" in backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "head" not in n],
                'lr': lr1,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)
            },
            {
                'params': model.head.parameters(),
                'lr': lr2,
                'weight_decay': weight_decay,
                'betas':(0.9, 0.95)
            },
        ])
    return optimizer
