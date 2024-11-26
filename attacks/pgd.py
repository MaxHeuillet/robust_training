
import torch
import torch.nn.functional as F


def pgd_attack(args, model, x_nat, y):

    use_rs = False

    if not use_rs:
        x_adv = x_nat.clone()
    else:
        raise NotImplemented
    
    x_adv = x_adv.clamp(0., 1.)

    if args.distance == 'Linf':
        # print('init x_adv')
        for _ in range(args.perturb_steps):

            x_adv = x_adv.requires_grad_()
            with torch.enable_grad():
                #print('infer')
                logits_adv = model(x_adv)
                #print('kl loss')
                # loss_kl = nn.KLDivLoss(reduction='sum')( F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1) )
                loss = F.cross_entropy( logits_adv, y)

            # grad = torch.autograd.grad(loss, [x_adv])[0]
            loss.backward()
            grad = x_adv.grad.detach()

            # print('other operations')
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_nat - args.epsilon), x_nat + args.epsilon).detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    else:
        print('attack distance misspecified')
        # x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv.detach()