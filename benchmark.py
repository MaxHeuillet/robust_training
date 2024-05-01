
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

import pickle as pkl
import gzip

###################################
# Experiments
###################################

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

parser = argparse.ArgumentParser()

parser.add_argument("--n_rounds", required=True, help="nb of active learning rounds")
parser.add_argument("--nb_epochs", required=True, help="nb of Lora epochs")
parser.add_argument("--size", required=True, help="wether to compute clean accuracy, PGD robustness or AA robustness")
parser.add_argument("--active_strategy", required=True, help="which observation strategy to choose")
parser.add_argument("--seed", required=True, help="the random seed")
parser.add_argument("--data", required=True, help="the data used for the experiment")
parser.add_argument("--model", required=True, help="the model used for the experiment")
parser.add_argument("--loss", required=True, help="the loss used to induce robustness")

args = parser.parse_args()


result = {}

n_rounds = int(args.n_rounds)
size = int(args.size)
nb_epochs = int(args.nb_epochs)
seed = int(args.seed)

result['active_strategy'] = n_rounds
result['n_rounds'] = n_rounds
result['size'] = size
result['nb_epochs'] = nb_epochs
result['seed'] = seed
result['data'] = args.data
result['model'] = args.model

############################# INITIATE THE EXPERIMENT:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils.set_seeds(seed)

if args.model == 'resnet50':
    model = models.resnet.resnet50(pretrained=True, progress=True, device="cuda").to('cuda')
else:
    print('model undefined')

print('model loaded successfully')
sys.stdout.flush()

##############################################

print('begin the evaluation experiment with {} // {} // {} // {} '.format(args.active_strategy, args.n_rounds,args.size,args.nb_epochs) )
sys.stdout.flush()

if args.data == 'CIFAR10':

    mean_cifar10 = (0.4914, 0.4822, 0.4465)
    std_cifar10 = (0.2471, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar10, std_cifar10)  ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

else:
    print('undefined data')

accuracy = utils.compute_clean_accuracy(model, test_loader)
print(f'The clean accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

result['init_clean_accuracy'] = accuracy

accuracy = utils.compute_PGD_accuracy(model, test_loader, device='cuda')
print(f'The PGD accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

result['init_PGD_accuracy'] = accuracy

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

lora.set_lora_gradients(model, layers)

sys.stdout.flush()

######## configuration of the experiment:

pool_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=True)

adapt_dataset = active.EmptyDataset()
pool_indices = list( range(len(pool_loader.dataset ) ) ) 


############# execution of the experiment:
import math
epoch_counter = 0
round_size = math.ceil(size / n_rounds)

for i in range(n_rounds):

    print( 'iteration {}'.format(i) )

    pool_loader = DataLoader( Subset(pool_dataset, pool_indices), batch_size=1000, shuffle=False)
        
    if args.active_strategy == 'uncertainty':
        query_indices = active.uncertainty_sampling(model, pool_loader, round_size)
        pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
    elif args.active_strategy == 'entropy':
        query_indices = active.entropy_sampling(model, pool_loader, round_size)
        pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
    elif args.active_strategy == 'margin':
        query_indices = active.margin_sampling(model, pool_loader, round_size)
        pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
    elif args.active_strategy == 'random':
        query_indices = active.random_sampling(model, pool_loader, round_size)
        pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
    elif args.active_strategy == 'attack':
        query_indices = active.attack_sampling(model, pool_loader, round_size)
        pool_indices = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
    elif args.active_strategy == 'full':
        query_indices = utils.get_indices_for_round( len(pool_loader.dataset), n_rounds, i)
        _ = utils.add_data(query_indices, pool_indices, pool_dataset, adapt_dataset)
    else:
        print('error')

    print( len(query_indices) )
    adapt_loader = DataLoader(adapt_dataset, batch_size=128, shuffle=True)
    print( len(adapt_loader.dataset) )

    ################# Update the ResNet through low rank matrices:
    print('start training process')
    sys.stdout.flush()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD( model.parameters(),lr=0.001, weight_decay=0.0001, momentum=0.9, nesterov=True, )

    for _ in range(nb_epochs):
        print('epoch {}'.format(_) )
        sys.stdout.flush()
        model.train()

        #Check if epoch_counter is at a milestone to reduce learning rate
        if epoch_counter == 30 or epoch_counter == 50:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.1
            utils.update_learning_rate(optimizer, new_lr)
            print("Reduced learning rate to {}".format(new_lr))

        if args.loss == 'AT':
            err, loss = utils.epoch_AT_vanilla(adapt_loader, model, optimizer)
        elif args.loss == 'FAT':
            err, loss = utils.epoch_fast_AT(adapt_loader, model, optimizer)
        elif args.loss == 'TRADES':
            err, loss = utils.epoch_TRADES(adapt_loader, model, optimizer)

        print('epoch finished {} - {}'.format(err, loss) )
        sys.stdout.flush()
        epoch_counter +=1

print('total amount of data used: {}'.format( len(adapt_loader)  )  )

print('start saving model')
sys.stdout.flush()
torch.save(model.state_dict(), './models/experiment_{}_{}_{}_{}.pth'.format(args.active_strategy, args.n_rounds,args.size,args.nb_epochs) )
print('finished saving model')
sys.stdout.flush()


##############################################

print('begin the evaluation experiment with {} // {} // {} // {} '.format(args.active_strategy, args.n_rounds,args.size,args.nb_epochs) )
sys.stdout.flush()

accuracy = utils.compute_clean_accuracy(model, test_loader)
print(f'The clean accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

result['final_clean_accuracy'] = accuracy

accuracy = utils.compute_PGD_accuracy(model, test_loader, device='cuda')
print(f'The PGD accuracy of the model on the test set is {accuracy}%')
sys.stdout.flush()

result['final_PGD_accuracy'] = accuracy

print(result)
print('finished')
with gzip.open( './results/{}_{}_{}_{}_{}_{}_{}.pkl.gz'.format(args.data, args.model, args.active_strategy, n_rounds, size, nb_epochs, seed) ,'ab') as f:
        pkl.dump(result,f)
print('saved')

#if args.eval_type == 'clean':
#elif args.eval_type == 'PGD':
#elif args.eval_type == 'AA':
#accuracy = utils.compute_AA_accuracy(model, test_loader, device='cuda')
#print(f'The AA accuracy of the model on the test set is {accuracy}%')

# epsilon = 8/255
# accuracy = utils.compute_robust_accuracy(model, test_loader, epsilon, norm='Linf', device='cuda')
# print(f'The robust accuracy of the model on the test set is {accuracy}%')