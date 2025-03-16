
import torch.nn.functional as F

from corruptions import apgd_attack

def classic_at_loss_eval(config, model, x_nat, y):

    model.eval()
    
    x_adv = apgd_attack(config, model, x_nat, y)
    
    model.train()

    model.module.current_task = 'val_infer'
    logits_nat, logits_adv = model(x_1 = x_nat, x_2 = x_adv)
    model.module.current_task = None

    loss_values = F.cross_entropy(logits_adv, y, reduction='none')

    return loss_values, logits_nat, logits_adv
