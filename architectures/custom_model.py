import torch.nn as nn
from peft import LoraConfig,get_peft_model

class CustomModel(nn.Module):
    def __init__(self, args, original_model):
        
        super(CustomModel, self).__init__()
        self.args = args
        self.base_model = original_model

    def forward(self, x_1, x_2=None):

        logits_1 = self.base_model(x_1)
        if x_2 is not None:
            logits_2 = self.base_model(x_2)
            return logits_1, logits_2
        else:
            return logits_1

    def set_fine_tuning_strategy(self, ):

        if self.args.ft_type == 'full_fine_tuning':
            self._freeze_backbone()
            self._unfreeze_last_layer()

        elif self.args.ft_type == 'linear_probing':
            self._freeze_backbone()
            self._unfreeze_last_layer()

        # elif self.args.strategy == 'lora':
        #     self._freeze_backbone()
        #     self._apply_lora_adapters()

        else:
            raise ValueError(f"Unknown fine-tuning strategy: {self.args.ft_type}")
        
    def update_fine_tuning_strategy(self, iteration):

        if self.args.ft_type == 'full_fine_tuning' and iteration >= self.args.freeze_epochs and self.all_gradients_on==False:
            print('Unfreezing all layers')
            self._enable_all_gradients()
            


    def _enable_all_gradients(self):
        self.all_gradients_on = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def _freeze_backbone(self):
        self.all_gradients_on = False
        # Freeze all parameters in the model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layer(self):
        # Unfreeze parameters of the last layer
        last_layer = list(self.base_model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True

    # def _apply_lora_adapters(self):

    #     if 'convnext' in self.args.backbone:

    #         # Configure LoRA to target intermediate Conv2d and Linear layers only
    #         lora_config = LoraConfig(
    #             r=8,  # Rank of the low-rank adapters
    #             lora_alpha=32,  # Scaling factor for the LoRA adapters
    #             target_modules=["conv_dw", "mlp.fc1", "mlp.fc2"],  # Apply LoRA to these layers across the intermediate blocks
    #             lora_dropout=0.1,  # Dropout rate for LoRA adapters
    #             bias="none"  # No bias term for the adapters
    #         )

    #         self.base_model = get_peft_model(self.base_model, lora_config)

    #     elif 'wideresnet' in self.args.backbone:

    #         # LoRA configuration to apply adapters to intermediate Conv2d layers in WideResNet
    #         lora_config = LoraConfig(
    #             r=8,  # Rank of the low-rank adapters, lower for CNNs than for Transformers
    #             lora_alpha=16,  # Scaling factor for LoRA, less aggressive for convolutional layers
    #             target_modules=["conv_0", "conv_1"],  # Target only intermediate Conv2d layers
    #             lora_dropout=0.1,  # Regularization to prevent overfitting
    #             bias="none"  # No additional bias in LoRA adapters
    #         )

    #         self.base_model = get_peft_model(self.base_model, lora_config)

    #     elif 'vit' in self.args.backbone or 'deit' in self.args.backbone:

    #         lora_config = LoraConfig(
    #             r=8,  # Rank of the low-rank adapters, often suitable for transformer blocks
    #             lora_alpha=32,  # Scaling factor for the LoRA adapters
    #             target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],  # Target only these intermediate layers
    #             lora_dropout=0.1,  # Dropout to regularize the adapters
    #             bias="none"  # No bias for LoRA adapters
    #             )
            
    #         self.base_model = get_peft_model(self.base_model, lora_config)



