import torch
import torch.nn as nn
import torch.optim as optim

from models import resnet
import dill


from autoattack import AutoAttack

import torch
import torch.nn.functional as F
import time

import random
import numpy as np
import os


def add_data(query_indices, pool_indices, pool_dataset, adapt_dataset):
    global_query_indices = [ pool_indices[idx] for idx in query_indices]

    for idx in global_query_indices:
        image, label = pool_dataset[idx]
        adapt_dataset.add_data( [image], [label] )
    
    return [idx for idx in pool_indices if idx not in global_query_indices]
    


def set_seeds(seed):
    """
    Set the seed for reproducibility in all used libraries.
    
    Args:
    seed (int): The seed number to set.
    """
    # Setting the seed for various Python, Numpy and PyTorch operations
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setting the seed for the OS
    os.environ['PYTHONHASHSEED'] = str(seed)

# Attack types

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=10, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def epoch_base(loader, model,device, opt=None,):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model( X )
        loss = nn.CrossEntropyLoss()( yp, y )
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_AT_vanilla(loader, model,device, opt=None,):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = pgd_linf(model, X, y)
        yp = model( X + delta )
        loss = nn.CrossEntropyLoss()( yp, y )
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_fast_AT(loader, model, device, opt=None,):
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = fgsm(model, X, y, epsilon=0.1) #pgd_linf(model, X, y)
        yp = model( X + delta )
        loss = nn.CrossEntropyLoss()( yp, y )
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_free_AT(loader, model, device, opt=None,):

    num_repeats=10
    epsilon=0.1
    alpha=0.01

    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = torch.zeros_like(X, requires_grad=True)  # Initialize perturbation

        for _ in range(num_repeats):  # Update the adversarial example in-place
            yp = model(X + delta)  # Prediction on perturbed data
            loss = nn.CrossEntropyLoss()(yp, y)
            opt.zero_grad()
            loss.backward()  # Gradients w.r.t. delta and model parameters

            # Update delta within its allowable range and clamp
            delta.data = (delta + X.shape[0] * alpha * delta.grad.data).clamp(-epsilon, epsilon)
            delta.grad.zero_()

            # Update model parameters
            opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Training loop

def launch_experiment(model, device, train_loader, test_loader, opt, epochs, epoch_fn):

    print(*("{}".format(i) for i in ("Train Err", "Test Err", "Adv Err")), sep="\t")

    for _ in range(epochs):

        train_err, train_loss = epoch_fn(train_loader, model, device, opt)
        test_err, test_loss = epoch_base(test_loader, model,device)
        adv_err, adv_loss = epoch_AT_vanilla(test_loader, model,device, opt)
        print(*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")



def compute_clean_accuracy(model, test_loader, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign

def clamp(X, l, u, cuda=True):
    device = 'cuda' if cuda else 'cpu'
    if not isinstance(l, torch.Tensor):
        l = torch.tensor([l], device=device, dtype=torch.float32)
    if not isinstance(u, torch.Tensor):
        u = torch.tensor([u], device=device, dtype=torch.float32)
    return torch.max(torch.min(X, u), l)

def attack_pgd_restart(model, X, y, eps, alpha, attack_iters, n_restarts, rs=True, verbose=False,
                       linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    device = 'cuda' if cuda else 'cpu'
    max_loss = torch.zeros(y.shape[0], device=device)
    max_delta = torch.zeros_like(X, device=device)
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X, device=device)
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            if not l2_grad_update:
                delta.data = delta + alpha * sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad.pow(2).sum([1, 2, 3], keepdim=True).sqrt())

            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = delta.data.pow(2).sum([1, 2, 3], keepdim=True).sqrt()
                delta.data = eps * delta.data / torch.max(eps * torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()

        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
            if verbose:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
                
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta

def compute_AA_accuracy(model, test_loader, device='cuda'):

    model.eval()

    def forward_pass(x):
        return model(x)
    
    norm = 'Linf'
    epsilon = 8/255
    
    adversary = AutoAttack(forward_pass, norm=norm, eps=epsilon, version='standard')
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:

        start_time = time.time()

        images, labels = images.to(device), labels.to(device)

        x_adv = adversary.run_standard_evaluation(images, labels, bs= test_loader.batch_size )
        
        # Evaluate the model on adversarial examples
        outputs = model(x_adv)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        end_time = time.time()
        # print(f"Time taken for batch: {end_time - start_time:.4f} seconds")
    
    robust_accuracy = 100 * correct / total
    return robust_accuracy


def compute_PGD_accuracy(model, test_loader, device='cuda'):

    model.eval()
    
    correct = 0
    total = 0

    epsilon = 8/255
    n_restarts = 10
    attack_iters = 50
    alpha = epsilon/4
    
    for images, labels in test_loader:

        start_time = time.time()

        images, labels = images.to(device), labels.to(device)

        delta = attack_pgd_restart(model, images, labels, epsilon, alpha, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True)
        
        x_adv = images + delta
        
        # Evaluate the model on adversarial examples
        outputs = model(x_adv)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        end_time = time.time()
        # print(f"Time taken for batch: {end_time - start_time:.4f} seconds")
    
    robust_accuracy = 100 * correct / total
    return robust_accuracy


def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
