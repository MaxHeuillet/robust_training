
import torch.nn.functional as F

def ce_loss(model, x_nat, y):

    logits_nat, _ = model(x_1 = x_nat, x_2 = None)
    loss_values = F.cross_entropy(logits_nat, y, reduction='none')

    return loss_values, logits_nat

