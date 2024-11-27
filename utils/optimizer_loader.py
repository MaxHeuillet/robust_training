from torch.optim import AdamW

def load_optimizer(args, config, model):
    if "convnext" in args.backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "head.fc" not in n],
                'lr': config["lr1"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)

            },
            {
                'params': model.head.fc.parameters(),
                'lr': config["lr2"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)
            },
        ])
    elif "wideresnet" in args.backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "logits" not in n],
                'lr': config["lr1"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)
            },
            {
                'params': model.logits.parameters(),
                'lr': config["lr2"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)
            },
        ])
    elif "deit" in args.backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "head" not in n],
                'lr': config["lr1"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)
            },
            {
                'params': model.head.parameters(),
                'lr': config["lr2"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)
            },
        ])
    elif "vit" in args.backbone:
        optimizer = AdamW([
            {
                'params': [p for n, p in model.named_parameters() if "head" not in n],
                'lr': config["lr1"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)
            },
            {
                'params': model.head.parameters(),
                'lr': config["lr2"],
                'weight_decay': config["weight_decay"],
                'betas':(0.9, 0.95)
            },
        ])
    return optimizer
