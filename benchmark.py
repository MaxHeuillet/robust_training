
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

import torch.nn.utils.parametrize as parametrize
import lora 

from torch.utils.data import Subset
import trades
import active
import torch.optim as optim
from torch import nn

###################################
# Experiments
###################################

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

parser = argparse.ArgumentParser()

parser.add_argument("--eval_type", required=True, help="wether to compute clean accuracy, PGD robustness or AA robustness")
parser.add_argument("--n_rounds", required=True, help="nb of active learning rounds")
parser.add_argument("--nb_epochs", required=True, help="nb of Lora epochs")
parser.add_argument("--round_size", required=True, help="wether to compute clean accuracy, PGD robustness or AA robustness")
parser.add_argument("--active_strategy", required=True, help="which observation strategy to choose")

args = parser.parse_args()

n_rounds = int(args.n_rounds)
round_size = int(args.round_size)
nb_epochs = int(args.nb_epochs)

############################# INITIATE THE EXPERIMENT:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet.resnet50(pretrained=True, progress=True, device="cuda").to('cuda')
print('model loaded successfully')
sys.stdout.flush()

##############################################

print('begin the evaluation experiment with {} // {} // {} // {} '.format(args.active_strategy, args.n_rounds,args.round_size,args.nb_epochs) )
sys.stdout.flush()

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2471, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_cifar10, std_cifar10)  ])

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

pool_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=True)

adapt_dataset = active.EmptyDataset()
pool_indices = list( range(len(pool_loader.dataset ) ) ) 


############# execution of the experiment:

epoch_counter = 0

for i in range(n_rounds):

    print( 'iteration {}'.format(i) )
    
    pool_loader = DataLoader( Subset(pool_dataset, pool_indices), batch_size=1000, shuffle=False)
    
    if args.active_strategy == 'uncertainty':
        query_indices = active.uncertainty_sampling(model, pool_loader, round_size)
    elif args.active_strategy == 'random':
        query_indices = active.random_sampling(model, pool_loader, n_instances=10)
    else:
        print('error')

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

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD( model.parameters(),lr=0.001, weight_decay=0.0001, momentum=0.9, nesterov=True, )

    adapt_loader = DataLoader(adapt_dataset, batch_size=128, shuffle=True)
    for _ in range(nb_epochs):
        print('epoch {}'.format(_) )
        sys.stdout.flush()
        model.train()

        # Check if epoch_counter is at a milestone to reduce learning rate
        # if epoch_counter == 30 or epoch_counter == 50:
        #     current_lr = optimizer.param_groups[0]['lr']
        #     new_lr = current_lr * 0.1
        #     utils.update_learning_rate(optimizer, new_lr)
        #     print("Reduced learning rate to {}".format(new_lr))

        for batch_idx, (data, target) in enumerate(adapt_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = trades.trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer,)
            # loss = nn.CrossEntropyLoss()( model(data) ,target)
            print(loss)
            loss.backward()
            optimizer.step()
        print('epoch finished')
        sys.stdout.flush()
        epoch_counter +=1

print('start saving model')
sys.stdout.flush()
torch.save(model.state_dict(), './models/experiment_{}_{}_{}_{}.pth'.format(args.active_strategy, args.n_rounds,args.round_size,args.nb_epochs) )
print('finished saving model')
sys.stdout.flush()


##############################################

print('begin the evaluation experiment with {} // {} // {} // {} '.format(args.active_strategy, args.n_rounds,args.round_size,args.nb_epochs) )
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