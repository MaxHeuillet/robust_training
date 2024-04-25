
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

import sys

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
print('model loaded successfully')
sys.stdout.flush()

##############################################

print('begin the evaluation experiment')
sys.stdout.flush()

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

accuracy = utils.compute_clean_accuracy(model, test_loader)
print(f'The clean accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

accuracy = utils.compute_PGD_accuracy(model, test_loader, device='cuda')
print(f'The PGD accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

##############################################

######## add Lora parametrizations to the model:
import torch.nn.utils.parametrize as parametrize
import lora 

layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
                model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
                model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
                model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]

for conv_layer in layers:
    print('hey')
    lora_param = lora.layer_parametrization(conv_layer, device="cuda", rank=10, lora_alpha=1)
    parametrize.register_parametrization(conv_layer, 'weight', lora_param)

sys.stdout.flush()

######## remove gradient tracking on main weights and put gradient tracking on lora matrices:

for name, param in model.named_parameters():
    if 'mat' not in name:
        print(f'Freezing non-LoRA parameter {name}')
        param.requires_grad = False

for layer in layers:
  layer.parametrizations["weight"][0].requires_grad = True

for name, param in model.named_parameters():
   if param.requires_grad:
        print(f"Parameter: {name}, Shape: {param.size()}")

sys.stdout.flush()

######## configuration of the experiment:

import active
import torch.optim as optim
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2471, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_cifar10, std_cifar10)  ])

pool_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
pool_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize model, optimizer, and loss function
model = models.resnet.resnet50(pretrained=True, progress=True, device="cuda").to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

adapt_dataset = active.EmptyDataset()

pool_indices = list( range(len(pool_loader.dataset ) ) ) 
n_queries = 6
active_batch_size = 100


############# execution of the experiment:
from torch.utils.data import Subset
import trades

for i in range(n_queries):

    print( 'iteration {}'.format(i) )
    
    pool_loader = DataLoader( Subset(pool_dataset, pool_indices), batch_size=1000, shuffle=False)
    query_indices = active.uncertainty_sampling(model, pool_loader, active_batch_size)

    global_query_indices = [ pool_indices[idx] for idx in query_indices]

    for idx in global_query_indices:
        image, label = pool_dataset[idx]
        adapt_dataset.add_data( [image], [label] )

    # Remove queried instances from the pool
    pool_indices = [idx for idx in pool_indices if idx not in global_query_indices]
    print( len(pool_indices) )

    ################# Update the ResNet through low rank matrices:
    print('start training process')
    sys.stdout.flush()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    adapt_loader = DataLoader(adapt_dataset, batch_size=128, shuffle=True)
    for _ in range(10):
        print('epoch {}'.format(_) )
        sys.stdout.flush()
        model.train()
        for batch_idx, (data, target) in enumerate(adapt_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = trades.trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer,)
            print(loss)
            loss.backward()
            optimizer.step()
        print('epoch finished')
        sys.stdout.flush()

print('start saving model')
sys.stdout.flush()
torch.save(model.state_dict(), './models/experiment.pth')
print('finished saving model')
sys.stdout.flush()


##############################################

print('begin the evaluation experiment')
sys.stdout.flush()

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

accuracy = utils.compute_clean_accuracy(model, test_loader)
print(f'The clean accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

accuracy = utils.compute_PGD_accuracy(model, test_loader, device='cuda')
print(f'The PGD accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

#if args.eval_type == 'clean':
#elif args.eval_type == 'PGD':
#elif args.eval_type == 'AA':
#accuracy = utils.compute_AA_accuracy(model, test_loader, device='cuda')
#print(f'The AA accuracy of the model on the test set is {accuracy}%')

# epsilon = 8/255
# accuracy = utils.compute_robust_accuracy(model, test_loader, epsilon, norm='Linf', device='cuda')
# print(f'The robust accuracy of the model on the test set is {accuracy}%')