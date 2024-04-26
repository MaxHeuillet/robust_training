import torch
import numpy as np
from torch.utils.data import Dataset
import random
import utils


# Define an Empty Dataset
class EmptyDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        self.targets = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        
        return image, target

    def add_data(self, new_data, new_targets):
        self.data.extend(new_data)
        self.targets.extend(new_targets)


# Uncertainty sampling function
def uncertainty_sampling(model, loader, n_instances=10):
    device = 'cuda'
    model.eval()
    all_indices = list( range(len(loader.dataset)) )
    uncertainties = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = torch.softmax( model(images), dim=1)
            uncertainty = 1 - torch.max(outputs, dim=1)[0]
            uncertainties.extend( uncertainty.tolist() )
    
    # Select the indices of the top uncertain instances
    uncertainty_indices = np.argsort(uncertainties)[-n_instances:]

    return [all_indices[i] for i in uncertainty_indices]


def margin_sampling(model, loader, n_instances=10):
    device = 'cuda'
    model.eval()
    all_indices = list(range(len(loader.dataset)))
    uncertainties = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            # Sort the outputs to get the top two
            top_probs, _ = outputs.topk(2, dim=1, largest=True, sorted=True)
            # Calculate the margin (difference between the top two probabilities)
            margin = top_probs[:, 0] - top_probs[:, 1]
            uncertainties.extend(margin.tolist())
    
    # Select the indices of the top uncertain instances (smallest margin)
    uncertainty_indices = np.argsort(uncertainties)[:n_instances]

    return [all_indices[i] for i in uncertainty_indices]


# Modified uncertainty sampling function to use classification entropy
def entropy_sampling(model, loader, n_instances=10):
    device = 'cuda'
    model.eval()
    all_indices = list(range(len(loader.dataset)))
    uncertainties = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            # Calculate the entropy
            entropy = -torch.sum(outputs * torch.log(outputs + 1e-12), dim=1)  # Adding a small epsilon to avoid log(0)
            uncertainties.extend(entropy.tolist())
    
    # Select the indices of the top uncertain instances (highest entropy)
    uncertainty_indices = np.argsort(uncertainties)[-n_instances:]

    return [all_indices[i] for i in uncertainty_indices]

# Random sampling function
def random_sampling(model, loader, n_instances=10):
    # List of all dataset indices
    all_indices = list(range(len(loader.dataset)))
    
    # Randomly select 'n_instances' indices
    random_indices = random.sample(all_indices, n_instances)
    
    return random_indices


def attack_sampling(model, loader, n_instances=10):
    device = 'cuda'
    
    all_indices = []
    successful_attack_indices = []
    epsilon = 8/255
    n_restarts = 2
    attack_iters = 10
    alpha = epsilon/4
    
    for i, (images, labels) in enumerate(loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        # Apply the attack to the images

        delta = utils.attack_pgd_restart(model, images, labels, epsilon, alpha, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True)
        
        x_adv = images + delta

        model.eval()
        outputs = model( x_adv )
        _, predicted_labels = torch.max(outputs, dim=1)
        # Compare predictions with the true labels
        attack_success = predicted_labels != labels
        # Store indices of successfully attacked observations
        batch_indices = [idx for idx, success in zip(range(i * loader.batch_size, i * loader.batch_size + len(labels)), attack_success) if success]
        successful_attack_indices.extend(batch_indices)

    result = random.sample(successful_attack_indices, n_instances)
    
    return result
