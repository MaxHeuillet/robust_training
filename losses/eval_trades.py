import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss_eval(args,
                model,
                x_natural,
                y,):
    
    step_size = args.epsilon / 4
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()

    if args.distance == 'l_inf':
        # print('init x_adv')
        for _ in range(args.perturb_steps):

            x_adv = x_adv.requires_grad_()
            with torch.enable_grad():
                #print('infer')
                logits_adv = model(x_adv)
                loss = F.cross_entropy( logits_adv, y)

            grad = torch.autograd.grad(loss, [x_adv])[0]
            # print('other operations')
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon).detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    else:
        print('attack distance misspecified')
        # x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    if args.arch == 'resnet50':
        logits_nat, logits_adv = model(x_natural, logits_adv)
    else:
        logits_nat = model(x_natural)    
        logits_adv = model(x_adv)
    clean_values = F.cross_entropy(logits_nat, y, reduction='none')
    robust_values = nn.KLDivLoss(reduction='none')(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)).sum(dim=1)
    loss_values = clean_values + args.beta * robust_values

    return loss_values, clean_values, robust_values, logits_nat, logits_adv
