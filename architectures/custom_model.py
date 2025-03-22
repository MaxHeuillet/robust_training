import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from transforms import load_module_transform

# Define Normalization Module
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

# Custom Model Wrapper with Normalization
class CustomModel(nn.Module):
    def __init__(self, config, original_model):
        super(CustomModel, self).__init__()
        self.config = config
        self.base_model = original_model
        self.current_tracker = None
        self.current_task = None

        # Add Normalization Module
        mean_values, std_values = load_module_transform(config)
        self.normalize = Normalize(mean=mean_values, std=std_values)
        self._enable_all_gradients()

    def forward(self, x_1=None, x_2=None):
        
        if x_1 is not None and x_2 is not None:
            self.current_tracker = 'nat'
            x_1 = self.normalize(x_1)  # Apply normalization
            logits_1 = self.base_model(x_1)
            self.current_tracker = None

            self.current_tracker = 'adv'
            x_2 = self.normalize(x_2)  # Apply normalization
            logits_2 = self.base_model(x_2)
            self.current_tracker = None

            return logits_1, logits_2
        
        elif x_1 is None and x_2 is not None:
            self.current_tracker = 'adv'
            x_2 = self.normalize(x_2)  # Apply normalization
            logits_2 = self.base_model(x_2)
            self.current_tracker = None

            return None, logits_2
        
        else:  
            self.current_tracker = 'nat'
            x_1 = self.normalize(x_1)  # Apply normalization
            logits_1 = self.base_model(x_1)
            self.current_tracker = None

            return logits_1

    def set_fine_tuning_strategy(self):
        if self.config.ft_type == 'full_fine_tuning':
            self._enable_all_gradients()

        elif self.config.ft_type == 'linear_probing':
            self._freeze_backbone()
            self._unfreeze_last_layer()

        else:
            raise ValueError(f"Unknown fine-tuning strategy: {self.config.ft_type}")
            
    def _enable_all_gradients(self):
        self.all_gradients_on = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def _freeze_backbone(self):
        self.all_gradients_on = False
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layer(self):
        last_layer = list(self.base_model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True


