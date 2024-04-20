import torch
import torch.nn as nn
import torch.optim as optim

from models import resnet
import dill

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

from autoattack import AutoAttack

def compute_robust_accuracy(model, test_loader, epsilon, norm='Linf', device='cuda'):

    model.eval()

    def forward_pass(x):
        return model(x)
    
    adversary = AutoAttack(forward_pass, norm=norm, eps=epsilon, version='standard')
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        x_adv = adversary.run_standard_evaluation(images, labels, bs=images.size(0))
        
        # Evaluate the model on adversarial examples
        outputs = model(x_adv)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    robust_accuracy = 100 * correct / total
    return robust_accuracy