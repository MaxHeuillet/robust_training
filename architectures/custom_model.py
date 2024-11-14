import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, original_model):
        super(CustomModel, self).__init__()
        self.base_model = original_model

    
    def forward(self, x_natural, x_adv=None):
        # Forward pass for natural input
        
        logits_nat = self.base_model(x_natural)

        if x_adv is not None:
            logits_adv = self.base_model(x_adv)
            return logits_nat, logits_adv
        else:
            return logits_nat
