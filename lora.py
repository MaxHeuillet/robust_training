"""
References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import models_local 
import utils

import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

import torch
# import lora 
# import utils
from functools import partial
import torch.nn.utils.parametrize as parametrize


class LoRA(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cuda'):
        super().__init__()
        self.mat_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.mat_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.mat_A, mean=0, std=1)

        self.scale = alpha / rank

    def forward(self, W):
        print('test')
        return W + torch.matmul(self.mat_B, self.mat_A).view(W.shape) * self.scale

def layer_parametrization(layer, device, rank = 10, lora_alpha = 1):
  # print( layer.weight.shape, type(layer) )
  if isinstance(layer, nn.Linear):
    features_in, features_out = layer.weight.shape
  elif isinstance(layer, nn.Conv2d):
     features_out, features_in = layer.weight.view(layer.weight.shape[0], -1).shape
  else:
     print('error')
  return LoRA(features_in, features_out, rank = rank, alpha = lora_alpha, device = device)
     

def set_lora_gradients(model, layers):
  for name, param in model.named_parameters():
    if 'mat' not in name:
      # print(f'Freezing non-LoRA parameter {name}')
      param.requires_grad = False
  
  for layer in layers:
    layer.parametrizations["weight"][0].requires_grad = True
  
  for name, param in model.named_parameters():
    if param.requires_grad:
      pass
      # print(f"Parameter: {name}, Shape: {param.size()}")