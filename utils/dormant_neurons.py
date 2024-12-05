import torch
import torch.nn as nn
from functools import partial
from typing import Union, Tuple
import torch.nn.functional as F

class ActivationTracker:
    def __init__(self):
        self.activations = {}

# Register hooks
def register_hooks(model, tracker_nat, ):
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Assign name to module
            module._name = name

            def get_activation(module, model):
                def hook(module, input, output):
                    name = module._name
                    tracker_nat.activations[name] = F.relu(output)

                return hook

            handle = module.register_forward_hook(get_activation(module, model))
            handles.append(handle)

    return handles



@torch.inference_mode()
def _get_masks(activations, tau: float, ineq_type: str) -> torch.Tensor:
    masks = []

    for activation in activations.values():
        if activation.ndim == 4:
            # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer
            score = activation.abs().mean(dim=0)

        normalized_score = score / (score.mean() + 1e-9)
        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)

        if ineq_type == 'leq':
            layer_mask[normalized_score <= tau] = True
        elif ineq_type == 'geq':
            layer_mask[normalized_score >= tau] = True
        elif ineq_type == 'eq':
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = True
        else:
            raise ValueError(f"Invalid inequality type: {ineq_type}")

        masks.append(layer_mask)

    return masks

@torch.inference_mode()
def compute_stats(activations):
    # Masks for tau=0 logging


    # Compute zero masks (neurons with zero normalized activation)
    zero_masks = _get_masks(activations, 0.0, 'eq')
    zero_count = sum([torch.sum(mask).item() for mask in zero_masks])

    # Compute dormant masks (excluding zero neurons)
    dormant_masks = _get_masks(activations, 0.01, 'leq')
    # Exclude zero neurons from dormant neurons
    adjusted_dormant_masks = [dormant_mask & (~zero_mask) for dormant_mask, zero_mask in zip(dormant_masks, zero_masks)]
    dormant_count = sum([torch.sum(mask).item() for mask in adjusted_dormant_masks])

    # Compute overactive masks
    overactive_masks = _get_masks(activations, 3.0, 'geq')
    overactive_count = sum([torch.sum(mask).item() for mask in overactive_masks])

    # Compute total neurons
    total_neurons = sum([mask.numel() for mask in zero_masks])

    return {
        "total_neurons": total_neurons,
        "zero_count": zero_count,
        "dormant_count": dormant_count,
        "overactive_count": overactive_count,
    }











# @torch.inference_mode()
# def _get_masks(activations, tau:float, ineq_type:str) -> torch.Tensor:
#     """
#     Computes the ReDo mask for a given set of activations.
#     The returned mask has True where neurons are dormant and False where they are active.
#     """
#     masks = []

#     for name, activation in list( activations.items() ):
#         # Taking the mean here conforms to the expectation under D in the main paper's formula
#         if activation.ndim == 4:
#             # Conv layer
#             score = activation.abs().mean( dim=(0, 2, 3) )
#         else:
#             # Linear layer
#             score = activation.abs().mean(dim=0)

#         # print('score', score)
#         # Divide by activation mean to make the threshold independent of the layer size
#         # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
#         # https://github.com/google/dopamine/issues/209

#         normalized_score = score / (score.mean() + 1e-9)
#         layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)

#         if tau > 0.0 and ineq_type == 'leq':
#             layer_mask[normalized_score <= tau] = 1
#         elif tau > 0.0 and ineq_type == 'geq':
#             layer_mask[normalized_score >= tau] = 1
#         else:
#             layer_mask[ torch.isclose( normalized_score, torch.zeros_like(normalized_score) ) ] = 1

#         masks.append(layer_mask)

#     return masks

# @torch.inference_mode()
# def _get_activation(name: str, activations):
#     """Fetches and stores the activations of a network layer."""

#     def hook(layer: Union[nn.Linear, nn.Conv2d], input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
#         """
#         Get the activations of a layer with ReLU nonlinearity.
#         ReLU has to be called explicitly here because the hook is attached to the conv/linear layer.
#         """
#         activations[name] = F.relu(output)

#     return hook

# @torch.inference_mode()
# def run_redo( obs: torch.Tensor, model: nn.Module, ):
    
#     """
#     Checks the number of dormant neurons for a given model. Returns the number of dormant neurons.
#     """

#     #print('step1')

#     activations = {}
#     activation_getter = partial(_get_activation, activations=activations)

#     #print('step2')

#     # Register hooks for all Conv2d and Linear layers to calculate activations
#     handles = []
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#             handles.append( module.register_forward_hook(activation_getter(name)) )
#     # print(handles)
#     #print('step3')

#     # Calculate activations
#     _ = model( obs )  # Add batch dimension if necessary,  .unsqueeze(0)

#     #print('step4')

#     # Masks for tau=0 logging
#     zero_masks = _get_masks(activations, 0.0, 'leq')
#     total_neurons = sum([torch.numel(mask) for mask in zero_masks])
#     zero_count = sum([torch.sum(mask) for mask in zero_masks])
#     zero_fraction = zero_count / total_neurons

#     # Calculate the masks actually used for resetting
#     masks = _get_masks(activations, 0.1, 'leq')
#     dormant_count = sum([torch.sum(mask) for mask in masks])
#     total_neurons = sum([torch.numel(mask) for mask in masks])
#     dormant_fraction = dormant_count / total_neurons 

#     # Calculate the masks actually used for resetting
#     masks = _get_masks(activations, 3, 'geq')
#     overactive_count = sum([torch.sum(mask) for mask in masks])
#     total_neurons = sum([torch.numel(mask) for mask in masks])
#     overactive_fraction = overactive_count / total_neurons 
    
#     # print(dormant_count, total_neurons, dormant_fraction)

#     # Remove the hooks again
#     for handle in handles:
#         handle.remove()

#     return {
#         "total_neurons":total_neurons,
#         "zero_fraction": zero_fraction.item(),
#         "zero_count": zero_count.item(),
#         "dormant_fraction": dormant_fraction.item(),
#         "dormant_count": dormant_count.item(),
#         "overactive_fraction": overactive_fraction.item(),
#         "overactive_count": overactive_count.item(),
#     }