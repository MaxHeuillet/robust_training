import torch
import torch.nn as nn
from functools import partial
from typing import Union, Tuple
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationTrackerAggregated:
    def __init__(self):
        # For each layer: store cumulative sums and counts of activations
        self.sums = {}   # Will hold accumulated sums of absolute activations
        self.counts = {} # Will hold total counts of elements processed per neuron
        self.is_conv = {} # Keep track if layer is Conv (for dimension handling)

    def accumulate(self, name, activation):
        # Move to CPU for safety
        activation = activation.detach().cpu()

        # Check if it's a conv layer or linear layer by dimensions
        # Conv: (B, C, H, W)
        # Linear: (B, C)
        if activation.ndim == 4:
            # Conv layer
            # sum over batch, height, and width: result is shape (C,)
            abs_sum = activation.abs().sum(dim=(0, 2, 3))
            # count how many elements per channel
            b, c, h, w = activation.shape
            elem_count = b * h * w
            if name not in self.sums:
                self.sums[name] = torch.zeros(c, dtype=torch.float32)
                self.counts[name] = 0
                self.is_conv[name] = True
            self.sums[name] += abs_sum
            self.counts[name] += elem_count

        else:
            # Linear layer: (B, C)
            abs_sum = activation.abs().sum(dim=0)  # shape (C,)
            b, c = activation.shape
            elem_count = b
            if name not in self.sums:
                self.sums[name] = torch.zeros(c, dtype=torch.float32)
                self.counts[name] = 0
                self.is_conv[name] = False
            self.sums[name] += abs_sum
            self.counts[name] += elem_count

    def get_activations_mean(self):
        # Compute mean absolute activation per neuron
        activations_mean = {}
        for name in self.sums:
            activations_mean[name] = self.sums[name] / (self.counts[name] + 1e-9)
        return activations_mean

    def clear(self):
        self.sums = {}
        self.counts = {}
        self.is_conv = {}

def register_hooks_aggregated(model, tracker_nat, tracker_adv):
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module._name = name

            def get_activation(mod, model):
                def hook(mod, input, output):
                    name = mod._name
                    if model.module.current_tracker == 'nat' and model.module.current_task == 'infer':
                        tracker_nat.accumulate(name, F.relu(output))
                    if model.module.current_tracker == 'adv' and model.module.current_task == 'infer':
                        tracker_adv.accumulate(name, F.relu(output))
                return hook

            handle = module.register_forward_hook(get_activation(module, model))
            handles.append(handle)
    return handles

@torch.inference_mode()
def _compute_masks_from_stats(activations_mean, is_conv, tau, ineq_type):
    masks = []
    for name, mean_vals in activations_mean.items():
        # mean_vals is per-channel (conv) or per-neuron (linear)
        score = mean_vals  # already mean per neuron over dataset
        # Normalize by its own mean
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
def compute_stats_aggregated(tracker):
    # Compute the mean activation values
    activations_mean = tracker.get_activations_mean()

    # Compute zero masks (neurons with zero normalized activation)
    zero_masks = _compute_masks_from_stats(activations_mean, tracker.is_conv, 0.0, 'eq')
    zero_count = sum([mask.sum().item() for mask in zero_masks])

    # Compute dormant masks (excluding zero neurons)
    dormant_masks = _compute_masks_from_stats(activations_mean, tracker.is_conv, 0.01, 'leq')
    adjusted_dormant_masks = [dormant & (~zero) for dormant, zero in zip(dormant_masks, zero_masks)]
    dormant_count = sum([mask.sum().item() for mask in adjusted_dormant_masks])

    # Compute overactive masks
    overactive_masks = _compute_masks_from_stats(activations_mean, tracker.is_conv, 3.0, 'geq')
    overactive_count = sum([mask.sum().item() for mask in overactive_masks])

    # Compute total neurons
    total_neurons = sum([mask.numel() for mask in zero_masks])

    return {
        "total_neurons": total_neurons,
        "zero_count": zero_count,
        "dormant_count": dormant_count,
        "overactive_count": overactive_count,
    }

# import torch
# import torch.nn as nn
# from functools import partial
# from typing import Union, Tuple
# import torch.nn.functional as F

# class ActivationTracker:
#     def __init__(self, ):
#         self.activations = {}
        

# # Register hooks
# def register_hooks(model, tracker_nat, tracker_adv ):
    
#     handles = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#             # Assign name to module
#             module._name = name

#             def get_activation(mod, model):
#                 def hook(mod, input, output):
#                     name = mod._name
#                     # print('hook called', model.module.current_tracker, model.module.current_task)
#                     if model.current_tracker == 'nat' and model.current_task == 'infer':
#                         tracker_nat.activations[name] = F.relu(output)
#                     if model.current_tracker == 'adv' and model.current_task == 'infer':
#                         tracker_adv.activations[name] = F.relu(output)
                    
#                 return hook

#             handle = module.register_forward_hook(get_activation(module, model))
#             handles.append(handle)

#     return handles



# @torch.inference_mode()
# def _get_masks(activations, tau: float, ineq_type: str) -> torch.Tensor:
#     masks = []

#     for activation in activations.values():
#         if activation.ndim == 4:
#             # Conv layer
#             score = activation.abs().mean(dim=(0, 2, 3))
#         else:
#             # Linear layer
#             score = activation.abs().mean(dim=0)

#         print('score', score, score.shape)
#         normalized_score = score / (score.mean() + 1e-9)
#         layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)

#         if ineq_type == 'leq':
#             layer_mask[normalized_score <= tau] = True
#         elif ineq_type == 'geq':
#             layer_mask[normalized_score >= tau] = True
#         elif ineq_type == 'eq':
#             layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = True
#         else:
#             raise ValueError(f"Invalid inequality type: {ineq_type}")

#         masks.append(layer_mask)

#     return masks

# @torch.inference_mode()
# def compute_stats(activations):
#     # Masks for tau=0 logging

#     # Compute zero masks (neurons with zero normalized activation)
#     zero_masks = _get_masks(activations, 0.0, 'eq')
#     zero_count = sum([torch.sum(mask).item() for mask in zero_masks])

#     # Compute dormant masks (excluding zero neurons)
#     dormant_masks = _get_masks(activations, 0.01, 'leq')
#     # Exclude zero neurons from dormant neurons
#     adjusted_dormant_masks = [dormant_mask & (~zero_mask) for dormant_mask, zero_mask in zip(dormant_masks, zero_masks)]
#     dormant_count = sum([torch.sum(mask).item() for mask in adjusted_dormant_masks])

#     # Compute overactive masks
#     overactive_masks = _get_masks(activations, 3.0, 'geq')
#     overactive_count = sum([torch.sum(mask).item() for mask in overactive_masks])

#     # Compute total neurons
#     total_neurons = sum([mask.numel() for mask in zero_masks])

#     return {
#         "total_neurons": total_neurons,
#         "zero_count": zero_count,
#         "dormant_count": dormant_count,
#         "overactive_count": overactive_count,  }











# # @torch.inference_mode()
# # def _get_masks(activations, tau:float, ineq_type:str) -> torch.Tensor:
# #     """
# #     Computes the ReDo mask for a given set of activations.
# #     The returned mask has True where neurons are dormant and False where they are active.
# #     """
# #     masks = []

# #     for name, activation in list( activations.items() ):
# #         # Taking the mean here conforms to the expectation under D in the main paper's formula
# #         if activation.ndim == 4:
# #             # Conv layer
# #             score = activation.abs().mean( dim=(0, 2, 3) )
# #         else:
# #             # Linear layer
# #             score = activation.abs().mean(dim=0)

# #         # print('score', score)
# #         # Divide by activation mean to make the threshold independent of the layer size
# #         # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
# #         # https://github.com/google/dopamine/issues/209

# #         normalized_score = score / (score.mean() + 1e-9)
# #         layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)

# #         if tau > 0.0 and ineq_type == 'leq':
# #             layer_mask[normalized_score <= tau] = 1
# #         elif tau > 0.0 and ineq_type == 'geq':
# #             layer_mask[normalized_score >= tau] = 1
# #         else:
# #             layer_mask[ torch.isclose( normalized_score, torch.zeros_like(normalized_score) ) ] = 1

# #         masks.append(layer_mask)

# #     return masks

# # @torch.inference_mode()
# # def _get_activation(name: str, activations):
# #     """Fetches and stores the activations of a network layer."""

# #     def hook(layer: Union[nn.Linear, nn.Conv2d], input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
# #         """
# #         Get the activations of a layer with ReLU nonlinearity.
# #         ReLU has to be called explicitly here because the hook is attached to the conv/linear layer.
# #         """
# #         activations[name] = F.relu(output)

# #     return hook

# # @torch.inference_mode()
# # def run_redo( obs: torch.Tensor, model: nn.Module, ):
    
# #     """
# #     Checks the number of dormant neurons for a given model. Returns the number of dormant neurons.
# #     """

# #     #print('step1')

# #     activations = {}
# #     activation_getter = partial(_get_activation, activations=activations)

# #     #print('step2')

# #     # Register hooks for all Conv2d and Linear layers to calculate activations
# #     handles = []
# #     for name, module in model.named_modules():
# #         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
# #             handles.append( module.register_forward_hook(activation_getter(name)) )
# #     # print(handles)
# #     #print('step3')

# #     # Calculate activations
# #     _ = model( obs )  # Add batch dimension if necessary,  .unsqueeze(0)

# #     #print('step4')

# #     # Masks for tau=0 logging
# #     zero_masks = _get_masks(activations, 0.0, 'leq')
# #     total_neurons = sum([torch.numel(mask) for mask in zero_masks])
# #     zero_count = sum([torch.sum(mask) for mask in zero_masks])
# #     zero_fraction = zero_count / total_neurons

# #     # Calculate the masks actually used for resetting
# #     masks = _get_masks(activations, 0.1, 'leq')
# #     dormant_count = sum([torch.sum(mask) for mask in masks])
# #     total_neurons = sum([torch.numel(mask) for mask in masks])
# #     dormant_fraction = dormant_count / total_neurons 

# #     # Calculate the masks actually used for resetting
# #     masks = _get_masks(activations, 3, 'geq')
# #     overactive_count = sum([torch.sum(mask) for mask in masks])
# #     total_neurons = sum([torch.numel(mask) for mask in masks])
# #     overactive_fraction = overactive_count / total_neurons 
    
# #     # print(dormant_count, total_neurons, dormant_fraction)

# #     # Remove the hooks again
# #     for handle in handles:
# #         handle.remove()

# #     return {
# #         "total_neurons":total_neurons,
# #         "zero_fraction": zero_fraction.item(),
# #         "zero_count": zero_count.item(),
# #         "dormant_fraction": dormant_fraction.item(),
# #         "dormant_count": dormant_count.item(),
# #         "overactive_fraction": overactive_fraction.item(),
# #         "overactive_count": overactive_count.item(),
# #     }