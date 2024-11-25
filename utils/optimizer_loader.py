from torch.optim import AdamW



def load_optimizer(args, model, ):

    if "convnext" in args.backbone:

        optimizer = AdamW( [
            {'params': model.parameters(), 'lr': args.init_lr, 'weigh_decay':args.weight_decay,} ,
            {'params': model.head.fc.parameters(), 'lr': 0.001, 'weigh_decay':args.weight_decay,},  
                    ])

    elif "wideresnet" in args.backbone:

        optimizer = AdamW( [
            {'params': model.parameters(), 'lr': args.init_lr, 'weigh_decay':args.weight_decay,}, 
            {'params': model.logits.parameters(), 'lr': 0.001, 'weigh_decay':args.weight_decay,},  
                    ])

    elif "deit" in args.backbone:

        optimizer = AdamW( [
            {'params': model.parameters(), 'lr': args.init_lr, 'weigh_decay':args.weight_decay,} ,
            {'params': model.head.parameters(), 'lr': 0.001, 'weigh_decay':args.weight_decay,}, 
                    ])

    elif "vit" in args.backbone:

        optimizer = AdamW( [
            {'params': model.parameters(), 'lr': args.init_lr, 'weigh_decay':args.weight_decay,},
            {'params': model.head.parameters(), 'lr': 0.001, 'weigh_decay':args.weight_decay,}, 
                    ])

    return optimizer