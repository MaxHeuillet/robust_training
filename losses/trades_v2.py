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


def trades_loss_v2(args,
                model,
                x_natural,
                y,
                optimizer,):
    
    model.eval()

    use_rs = False
    # logits_nat = model(x_natural)
    # x_adv = x_natural.detach() #+ 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()

    if not use_rs:
        x_adv = x_natural.clone()
    else:
        raise NotImplemented
    
    x_adv = x_adv.clamp(0., 1.)

    if args.distance == 'l_inf':
        # print('init x_adv')
        for _ in range(args.perturb_steps):

            x_adv = x_adv.requires_grad_()
            with torch.enable_grad():
                #print('infer')
                logits_adv = model(x_adv)
                #print('kl loss')
                # loss_kl = nn.KLDivLoss(reduction='sum')( F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1) )
                loss = F.cross_entropy( logits_adv, y)

            # print(f' GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes')
            # print(f' GPU Memory Cached: {torch.cuda.memory_reserved()} bytes')

            # print( )
            # print('gradient compute')
            grad = torch.autograd.grad(loss, [x_adv])[0]
            # print('other operations')
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon).detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    else:
        print('attack distance misspecified')
        # x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()


    model.train()

    optimizer.zero_grad()

    #if args.arch == 'resnet50':
    logits_nat, logits_adv = model(x_natural, x_adv)
    # else:
    #     logits_nat = model(x_natural)    
    #     logits_adv = model(x_adv)
        
    clean_values = F.cross_entropy(logits_nat, y, reduction='none')
        
    robust_values = nn.KLDivLoss(reduction='none')( F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1) ).sum(dim=1)
        
    loss_values = clean_values + args.beta * robust_values

    # else:
        # logits_nat, logits_adv = model(x_natural, x_adv)
        
        # clean_values = F.cross_entropy(logits_nat, y, reduction='none')
        # robust_values = nn.KLDivLoss(reduction='none')(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)).sum(dim=1)
        # loss_values = clean_values + args.beta * robust_values

    return loss_values, clean_values, robust_values, logits_nat, logits_adv
