import torch.nn as nn
import torch.nn.functional as F
from attacks import pgd_attack, apgd_attack


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def trades_loss_v2(setup, model, x_nat, y, ):
    
    model.eval()

    # x_adv = pgd_attack(args, model, x_nat, y)
    x_adv = apgd_attack(setup, model, x_nat, y)

    model.train()

    logits_nat, logits_adv = model(x_nat, x_adv)
        
    clean_values = F.cross_entropy(logits_nat, y, reduction='none')
        
    robust_values = nn.KLDivLoss(reduction='none')( F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1) ).sum(dim=1)
        
    loss_values = clean_values + setup.config.beta * robust_values

    return loss_values, logits_adv



    # logits_nat = model(x_natural)
    # x_adv = x_natural.detach() #+ 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()

            # print(f' GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes')
            # print(f' GPU Memory Cached: {torch.cuda.memory_reserved()} bytes')

            # print( )
            # print('gradient compute')

    #if args.arch == 'resnet50':
    # logits_nat, logits_adv = model(x_natural, x_adv)
    # else:
    #     logits_nat = model(x_natural)    
    #     logits_adv = model(x_adv)
    # logits_nat = model(x_natural)
    # logits_adv = model(x_adv)


    # else:
        # logits_nat, logits_adv = model(x_natural, x_adv)
        
        # clean_values = F.cross_entropy(logits_nat, y, reduction='none')
        # robust_values = nn.KLDivLoss(reduction='none')(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)).sum(dim=1)
        # loss_values = clean_values + args.beta * robust_values