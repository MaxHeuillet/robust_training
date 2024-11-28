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


def trades_loss_eval(args,
                model,
                x_nat,
                y,):
    
    step_size = args.epsilon / 4
    model.eval()
    
    x_adv = apgd_attack(args, model, x_nat, y)

    logits_nat, logits_adv = model(x_nat, x_adv)
    # logits_nat = model(x_natural)    
    # logits_adv = model(x_adv)
    # logits_nat, logits_adv = model(x_natural, x_adv)
    clean_values = F.cross_entropy(logits_nat, y, reduction='none')
    robust_values = nn.KLDivLoss(reduction='none')(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)).sum(dim=1)
    loss_values = clean_values + args.beta * robust_values

    return loss_values, clean_values, robust_values, logits_nat, logits_adv
