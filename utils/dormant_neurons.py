import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationTrackerAggregated:
    def __init__(self, train):
        # For each layer: store cumulative sums and counts of activations
        self.train = train
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
            
        elif activation.ndim == 3:
            # This is likely the ViT qkv output: (B, N, C)
            # Treat it similarly to a linear layer by merging B and N
            b, n, c = activation.shape
            abs_sum = activation.abs().sum(dim=(0, 1))  # sum over batch and tokens, leaving (C,)
            elem_count = b * n

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

            if tracker_nat.train and tracker_adv.train:

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
            
            elif not tracker_nat.train and not tracker_adv.train:

                def get_activation(mod, model):
                    def hook(mod, input, output):
                        name = mod._name
                        if model.module.current_tracker == 'nat' and model.module.current_task == 'val_infer':
                            tracker_nat.accumulate(name, F.relu(output))
                        if model.module.current_tracker == 'adv' and model.module.current_task == 'val_infer':
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
