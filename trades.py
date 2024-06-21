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



# def eval_test_nat(model, test_loader, device, advFlag=None,natural_mode=None):
#     # torch.manual_seed(1)
#     model.eval()
#     acc = 0.
#     print(natural_mode)
#     for images, labels in test_loader:
#          images = images.to(device)
#          labels = labels.to(device)
#          with torch.no_grad():
#             if natural_mode is not None:
#                 output = model.eval()(images,bn_name=natural_mode,thread=advFlag)
#             else:
#                 output = model.eval()(images)
#             acc += (output.max(1)[1] == labels).float().sum()
#     print(acc, len(test_loader.dataset))
#     return acc.item() / len(test_loader.dataset)

# def trades_loss_autolora(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0, distance='l_inf', LAMBDA1=0, LAMBDA2=0):
#     batch_size = len(x_natural)
#     # define KL-loss
#     criterion_kl = nn.KLDivLoss(size_average=False)
#     model.eval()

#     # generate adversarial example
#     x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
#     if distance == 'l_inf':
#         for _ in range(perturb_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 model.eval()
#                 loss_kl = F.cross_entropy(model(x_adv, thread=None), y)
#             grad = torch.autograd.grad(loss_kl, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     else:
#         assert False

#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

#     # zero gradient
#     model.zero_grad()
#     optimizer.zero_grad()
#     model.train()
#     # calculate robust loss
#     for n,p in model.named_parameters():
#         if 'fc' in n:
#             p.requires_grad = True
#         elif 'lora' in n:
#             p.requires_grad = True
#         else:
#             p.requires_grad = False
#     nat_output = model(x_natural, thread='nat')

#     for n,p in model.named_parameters():
#         if 'lora' in n:
#             p.requires_grad = False
#         else:
#             p.requires_grad = True
#     adv_output= model(x_adv, thread=None)

#     for n,p in model.named_parameters():
#         p.requires_grad = True

#     loss = LAMBDA1 * F.cross_entropy(nat_output, y) \
#         + (1 - LAMBDA1) * F.cross_entropy(adv_output, y) \
#         + LAMBDA2 *  (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1), F.softmax(nat_output, dim=1))
    
#     return loss


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    
    # define KL-loss
    # criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()

    # print(x_natural)
    # print(x_adv)

    if distance == 'l_inf':
        print('init x_adv')
        for _ in range(perturb_steps):
            print(f' GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes')
            print(f' GPU Memory Cached: {torch.cuda.memory_reserved()} bytes')

            x_adv = x_adv.requires_grad_()
            with torch.enable_grad():
                print('infer')
                logits_nat, logits_adv = model(x_natural, x_adv)
                print('kl loss')
                loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

            print(f' GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes')
            print(f' GPU Memory Cached: {torch.cuda.memory_reserved()} bytes')

            print( )
            print('gradient compute')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            print('other operations')
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon).detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits_nat, logits_adv = model(x_natural, x_adv)
    loss_natural = F.cross_entropy(logits_nat, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))
    loss = loss_natural + beta * loss_robust

    return logits_nat, loss



    # elif distance == 'l_2':
    #     delta = 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()
    #     delta = Variable(delta.data, requires_grad=True)

    #     # Setup optimizers
    #     optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

    #     for _ in range(perturb_steps):
    #         adv = x_natural + delta

    #         # optimize
    #         optimizer_delta.zero_grad()
    #         with torch.enable_grad():
    #             loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),  F.softmax(model(x_natural), dim=1))
    #         loss.backward()
    #         # renorming gradient
    #         grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
    #         delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
    #         # avoid nan or inf if gradient is 0
    #         if (grad_norms == 0).any():
    #             delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
    #         optimizer_delta.step()

    #         # projection
    #         delta.data.add_(x_natural)
    #         delta.data.clamp_(0, 1).sub_(x_natural)
    #         delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
    #     x_adv = Variable(x_natural + delta, requires_grad=False)