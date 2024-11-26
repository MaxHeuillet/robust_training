import time
import torch
import torch.nn.functional as F
import math
from attacks import apgd_attack, pgd_attack

def classic_at_loss(args, model, x_nat, y):

    model.eval()
    
    # x_adv = pgd_attack(args, model, x_nat, y)
    x_adv = apgd_attack(args, model, x_nat, y)

    # optimizer.zero_grad()
    
    model.train()
    logits_adv = model(x_adv)
    loss_values = F.cross_entropy(logits_adv, y, reduction='none')

    return loss_values, logits_adv

# y = torch.zeros(logits_adv.size(0), dtype=torch.long, device=logits_adv.device)
# y[0] = 1