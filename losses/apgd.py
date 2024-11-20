import time
import torch
import torch.nn.functional as F
import math

### parts of this code are from: https://github.com/nmndeep/revisiting-at/tree/main

def check_oscillation(x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()
        return (t <= k * k3 * torch.ones_like(t)).float()


#args, model, x, y, optimizer)

def apgd_attack(args, model, x, y, optimizer):
        
    # is_train=True
    # mixup=None
    use_rs=False
    n_iter= args.perturb_steps #10
    eps = args.epsilon #4/255
    norm = args.distance #'Linf'

    # y = y.reshape( (-1,1) )
    # print(y.shape)

    assert not model.training
    device = x.device
    ndims = len(x.shape) - 1

    if not use_rs:
        x_adv = x.clone()
    else:
        raise NotImplemented
        # if norm == 'Linf':
        #     t = torch.rand_like(x)
    
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone().detach()
    x_best_adv = x_adv.clone().detach()
    
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)
    
    # set params
    # n_fts = math.prod( x.shape[1:] )
    if norm in ['l_inf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.
    
    step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims], device=device, dtype=x.dtype)
    counter3 = 0

    x_adv.requires_grad_()
    #grad = torch.zeros_like(x)
    #for _ in range(self.eot_iter)
    #with torch.enable_grad()
    logits = model(x_adv)
    loss_indiv = F.cross_entropy(logits, y, reduction='none') 
    loss = loss_indiv.sum()
    #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    #grad /= float(self.eot_iter)
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()
    
    # if mixup is not None:
    #     acc = logits.detach().max(1)[1] == y.max(1)[1]
    # else:
    acc = logits.detach().max(1)[1] == y
    
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    # n_reduced = 0
    
    # u = torch.arange(x.shape[0], device=device)
    x_adv_old = x_adv.clone().detach()
    
    for i in range(n_iter):
        ### gradient step
        if True: #with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            # loss_curr = loss.detach().mean()
            
            a = 0.75 if i > 0 else 1.0

            if norm == 'l_inf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - eps), x + eps), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
            #return x_adv
            

        ### get gradient
        if i < n_iter - 1:
            x_adv.requires_grad_()
        #grad = torch.zeros_like(x)
        #for _ in range(self.eot_iter)
        #with torch.enable_grad()
        logits = model(x_adv)
        loss_indiv = F.cross_entropy(logits, y, reduction='none') 
        loss = loss_indiv.sum()
        
        #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        if i < n_iter - 1:
            # save one backward pass
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        #grad /= float(self.eot_iter)
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()
        
        # if mixup is not None:
        #     pred = logits.detach().max(1)[1] == y.max(1)[1]
        # else:
        pred = logits.detach().max(1)[1] == y
        
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = ~pred
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        # if verbose:
        #     str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
        #         step_size.mean(), topk.mean() * n_fts) if norm in ['L1'] else ' - step size: {:.5f}'.format(
        #         step_size.mean())
        #     print('iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
        #         i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats))
            #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
        
        ### check step size
        if True: #with torch.no_grad()
          y1 = loss_indiv.detach().clone()
          loss_steps[i] = y1 + 0
          ind = (y1 > loss_best).nonzero().squeeze()
          x_best[ind] = x_adv[ind].clone()
          grad_best[ind] = grad[ind].clone()
          loss_best[ind] = y1[ind] + 0
          loss_best_steps[i + 1] = loss_best + 0

          counter3 += 1

          if counter3 == k:
              if norm in ['l_inf', 'L2']:
                  fl_oscillation = check_oscillation(loss_steps, i, k, loss_best, k3=thr_decr)
                  fl_reduce_no_impr = (1. - reduced_last_check) * (
                      loss_best_last_check >= loss_best).float()
                  fl_oscillation = torch.max(fl_oscillation,
                      fl_reduce_no_impr)
                  reduced_last_check = fl_oscillation.clone()
                  loss_best_last_check = loss_best.clone()

                  if fl_oscillation.sum() > 0:
                      ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                      step_size[ind_fl_osc] /= 2.0
                    #   n_reduced = fl_oscillation.sum()

                      x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                      grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                  
                  counter3 = 0
                  k = max(k - size_decr, n_iter_min)
                          
    return x_best, acc, loss_best, x_best_adv


def apgd_loss(args, model, x_natural, y, optimizer):
    model.eval()
    x_adv, acc, loss, best_adv = apgd_attack(args, model, x_natural, y, optimizer)

    model.train()
    optimizer.zero_grad()
    logits_adv = model(x_adv)
    loss_values = F.cross_entropy(logits_adv, y, reduction='none')

    return loss_values, logits_adv