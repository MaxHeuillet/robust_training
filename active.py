import torch
import numpy as np
from torch.utils.data import Dataset
import random


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


# Random sampling function
def random_sampling(model, loader, n_instances=10):
    # List of all dataset indices
    all_indices = list(range(len(loader.dataset)))
    
    # Randomly select 'n_instances' indices
    random_indices = random.sample(all_indices, n_instances)
    
    return random_indices
