import torch.nn as nn
import torch.nn.functional as F

from corruptions import apgd_attack


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss_eval(config,model, x_nat, y,):
    
    model.eval()
    
    x_adv = apgd_attack(config, model, x_nat, y)

    # model.module.current_task = 'val_infer'
    logits_nat, logits_adv = model(x_nat, x_adv)
    # model.module.current_task = None

    clean_values = F.cross_entropy(logits_nat, y, reduction='none')
    robust_values = nn.KLDivLoss(reduction='none')(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)).sum(dim=1)
    loss_values = clean_values + config.beta * robust_values

    return loss_values, logits_nat, logits_adv
