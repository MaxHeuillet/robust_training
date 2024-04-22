
import numpy as np
import os
import argparse
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import models 
import utils

###################################
# Experiments
###################################

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

parser = argparse.ArgumentParser()

parser.add_argument("--eval_type", required=True, help="wether to compute clean accuracy, PGD robustness or AA robustness")
# parser.add_argument("--n_folds", required=True, help="number of folds")
# parser.add_argument("--model", required=True, help="model")
# parser.add_argument("--case", required=True, help="case")
# parser.add_argument("--context_type", required=True, help="context type")
# parser.add_argument("--approach", required=True, help="algorithme")
# parser.add_argument("--id", required=True, help="algorithme")

args = parser.parse_args()

ncpus = int ( os.environ.get('SLURM_CPUS_PER_TASK', default=1) )
ngpus = int( torch.cuda.device_count() )
# print('ncpus', ncpus,'ngpus', ngpus)

############################# INITIATE THE EXPERIMENT:

device = 'cuda'

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10)  ])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# model = LeNet().to(device)
model = models.resnet.resnet50(pretrained=True, progress=True, device="cuda").to('cuda')


if args.eval_type == 'clean':
    accuracy = utils.compute_clean_accuracy(model, test_loader)
    print(f'The clean accuracy of the model on the test set is {accuracy}%')
elif args.eval_type == 'PGD':
    accuracy = utils.compute_PGD_accuracy(model, test_loader, device='cuda')
    print(f'The PGD accuracy of the model on the test set is {accuracy}%')
elif args.eval_type == 'AA':
    accuracy = utils.compute_AA_accuracy(model, test_loader, device='cuda')
    print(f'The AA accuracy of the model on the test set is {accuracy}%')

# epsilon = 8/255
# accuracy = utils.compute_robust_accuracy(model, test_loader, epsilon, norm='Linf', device='cuda')
# print(f'The robust accuracy of the model on the test set is {accuracy}%')