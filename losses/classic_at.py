
import torch.nn.functional as F

from corruptions import apgd_attack

def classic_at_loss(args, model, x_nat, y):

    model.eval()
    
    x_adv = apgd_attack(args, model, x_nat, y)
    
    model.train()

    model.module.current_task = 'infer'
    _, logits_adv = model(x_1 = None, x_2 = x_adv)
    loss_values = F.cross_entropy(logits_adv, y, reduction='none')

    return loss_values, logits_adv

# y = torch.zeros(logits_adv.size(0), dtype=torch.long, device=logits_adv.device)
# y[0] = 1