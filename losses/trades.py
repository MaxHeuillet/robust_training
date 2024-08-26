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


def trades_loss(args,
                model,
                x_natural,
                y,
                optimizer,):
    
    # define KL-loss
    # criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()

    # print(x_natural)
    # print(x_adv)

    if args.distance == 'l_inf':
        # print('init x_adv')
        for _ in range(args.perturb_steps):
            # print(f' GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes')
            # print(f' GPU Memory Cached: {torch.cuda.memory_reserved()} bytes')

            x_adv = x_adv.requires_grad_()
            with torch.enable_grad():
                #print('infer')
                logits_nat, logits_adv = model(x_natural, x_adv)
                #print('kl loss')
                loss_kl = nn.KLDivLoss(reduction='sum')( F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1) )

            # print(f' GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes')
            # print(f' GPU Memory Cached: {torch.cuda.memory_reserved()} bytes')

            # print( )
            # print('gradient compute')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            # print('other operations')
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon).detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()


    if args.task == 'train':
        model.train()

        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits_nat, logits_adv = model(x_natural, x_adv)
        # print(logits_nat.shape, logits_adv.shape)
        
        clean_values = F.cross_entropy(logits_nat, y, reduction='none')
        # print(loss_natural.shape)  # Should be [batch_size]

        robust_values = nn.KLDivLoss(reduction='none')( F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1) ).sum(dim=1)
        # print(loss_robust.shape)  # Should be [batch_size]
        
        loss_values = clean_values + args.beta * robust_values
        # print(loss_individual.shape)  # Should be [batch_size]
    else:
        logits_nat, logits_adv = model(x_natural, x_adv)
        
        clean_values = F.cross_entropy(logits_nat, y, reduction='none')
        robust_values = nn.KLDivLoss(reduction='none')(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)).sum(dim=1)
        loss_values = clean_values + args.beta * robust_values

    return loss_values, clean_values, robust_values, logits_nat, logits_adv
