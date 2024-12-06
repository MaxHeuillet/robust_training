import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from attacks import pgd_attack, apgd_attack


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss_eval(setup,
                model,
                x_nat,
                y,):
    
    model.eval()
    
    # print('generate attack')
    x_adv = apgd_attack(setup, model, x_nat, y)

    model.module.current_task = 'infer'
    # print('infer')
    logits_nat, logits_adv = model(x_nat, x_adv)
    model.module.current_task = None

    # logits_nat = model(x_natural)    
    # logits_adv = model(x_adv)
    # logits_nat, logits_adv = model(x_natural, x_adv)
    clean_values = F.cross_entropy(logits_nat, y, reduction='none')
    robust_values = nn.KLDivLoss(reduction='none')(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)).sum(dim=1)
    loss_values = clean_values + setup.config.beta * robust_values

    return loss_values, clean_values, robust_values, logits_nat, logits_adv
