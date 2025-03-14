"""
References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""

from torch import nn
import torch
import torch.nn.utils.parametrize as parametrize
from peft import LoraConfig, get_peft_model

class LoRA(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cuda'):
        super().__init__()
        self.mat_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.mat_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.mat_A, mean=0, std=1)

        self.scale = alpha / rank

    def forward(self, W):
        return W + torch.matmul(self.mat_B, self.mat_A).view(W.shape) * self.scale

def layer_parametrization(layer, device, rank = 10, lora_alpha = 1):
  
  if isinstance(layer, nn.Linear):
    features_in, features_out = layer.weight.shape
  elif isinstance(layer, nn.Conv2d):
     features_out, features_in = layer.weight.view(layer.weight.shape[0], -1).shape
  else:
     print('error')
     print("you need to add here all the layer times and their input_dim and output_dim")
  return LoRA(features_in, features_out, rank = rank, alpha = lora_alpha, device = device)
     

def set_lora_gradients(args, model, layers):
  # this is to freeze the main parameters of the model and put gradient tracking on the lora matrices
  
  if args.arch == 'ViTs':
    for param in model.parameters():
        param.requires_grad = False
    # Only LoRA parameters will be trainable
    for name, param in model.named_parameters():
       if 'lora' in name:
          param.requires_grad = True
  
  elif args.arch == 'resnet50':
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
  else:
    print('LoRA is not implemented for this architecture')
  
  return model
  
def add_lora(args, model, layers):

  if args.arch == 'resnet50':

    for conv_layer in layers:
      lora_param = layer_parametrization(conv_layer, device="cuda", rank=10, lora_alpha=1)
      parametrize.register_parametrization(conv_layer, 'weight', lora_param)
    
  elif args.arch == 'ViTs':
      # Define the LoRA configuration
      lora_config = LoraConfig(
          r=8,          # Rank of the low-rank approximation
          lora_alpha=16, # Scaling factor for LoRA
          lora_dropout=0.1, # Dropout for LoRA
          target_modules=layers,  # Specify the transformer modules where LoRA should be applied
      )

      # Apply the LoRA adapters to the model
      model = get_peft_model(model, lora_config)

  else:
    print('LoRA is not implemented for this architecture')
    return model






