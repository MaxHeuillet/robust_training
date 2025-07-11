import torch
import torch.nn as nn

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

        mean_values, std_values = load_module_transform(config)
        self.normalize = Normalize(mean=mean_values, std=std_values)
        
        self._enable_all_gradients()

    def train(self, mode: bool = True):
        """
        We do NOT call super().train(mode) if ft_type == linear_probing, 
        because that forces everything to train before we can revert it, 
        and some submodules won't revert properly in DDP or other frameworks.
        """
        if self.config.ft_type == 'linear_probing':
            # Force everything to eval
            super().train(False)
            # Now only the head goes to train
            head_module = self.get_classification_head()
            head_module.train(mode)
        else:
            # Full fine-tuning: normal behavior
            super().train(mode)

        return self

    def get_classification_head(self):

        # 1) If there's a 'head' attribute, use that
        if hasattr(self.base_model, 'head'):
            return self.base_model.head
        
        # 2) Else if there's an 'fc' (common in ResNet-style models)
        elif hasattr(self.base_model, 'fc'):
            return self.base_model.fc
        
        # 3) Else if there's a 'classifier' (some MobileNet/ConvNeXt variants)
        elif hasattr(self.base_model, 'classifier'):
            return self.base_model.classifier
        
        # 4) Otherwise, no recognized classification head was found
        raise ValueError("Could not find a classification head in this model!")

    def forward(self, x_1=None, x_2=None):
        
        if x_1 is not None and x_2 is not None:
            # self.current_tracker = 'nat'
            x_1 = self.normalize(x_1)  # Apply normalization
            logits_1 = self.base_model(x_1)
            # self.current_tracker = None

            # self.current_tracker = 'adv'
            x_2 = self.normalize(x_2)  # Apply normalization
            logits_2 = self.base_model(x_2)
            # self.current_tracker = None

            return logits_1, logits_2
        
        elif x_1 is None and x_2 is not None:
            # self.current_tracker = 'adv'
            x_2 = self.normalize(x_2)  # Apply normalization
            logits_2 = self.base_model(x_2)
            # self.current_tracker = None

            return None, logits_2
        
        else:  
            # self.current_tracker = 'nat'
            x_1 = self.normalize(x_1)  # Apply normalization
            logits_1 = self.base_model(x_1)
            # self.current_tracker = None

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
        for param in self.base_model.parameters():
            param.requires_grad = True

    def _freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layer(self):
        head_module = self.get_classification_head() 
        for param in head_module.parameters():
            param.requires_grad = True



        
    # def get_head(self,):
    #     model = self.base_model
    #     # Identify the classification head based on architecture type
    #     if 'deit' in self.config.backbone or 'vit' in self.config.backbone or 'eva02' in self.config.backbone:
    #         head_module = model.head
    #     elif hasattr(model, 'classifier'):
    #         head_module = model.classifier
    #     elif hasattr(model, 'fc'):
    #         head_module = model.fc
    #     elif hasattr(model, 'head') and hasattr(model.head, 'fc'):
    #         head_module = model.head.fc
    #     elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
    #         head_module = model.head
    #     else:
    #         raise ValueError(f"Could not identify the classification head for backbone: {self.config.backbone}")

    #     self.head = head_module