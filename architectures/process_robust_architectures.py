import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from typing import Tuple
import os
import argparse
from timm import create_model
import timm

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser(description="Process robust vision model checkpoints.")
parser.add_argument("--path", type=str, required=True, help="Base directory where raw and processed checkpoints are stored.")
args = parser.parse_args()

input_dir = os.path.expanduser(args.path)
output_dir = input_dir #os.path.join(input_dir, "state_dicts_share")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Utility: Image normalization wrapper
# ----------------------------
class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float], persistent: bool = True) -> None:
        super(ImageNormalizer, self).__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1), persistent=persistent)
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1), persistent=persistent)

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

# ----------------------------
# Constants
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

equivalences = {
    'weights_convnext_base': 'convnext_base', 
    'weights_vit_s': 'deit_small_patch16_224',
    'weights_convnext_t': 'convnext_tiny',
    'weights_vit_b_50_ep': 'vit_base_patch16_224'
}

# ----------------------------
# Convert Robust Backbones
# ----------------------------
for backbone in equivalences:
    model = create_model(f"timm/{equivalences[backbone]}", pretrained=False)
    ckpt_path = os.path.join(input_dir, f"{backbone}.pt")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Clean state_dict keys
    ckpt = {k: v for k, v in ckpt.items() if "normalize." not in k}
    ckpt = {k.replace('module.', '').replace('base_model.', '').replace('se_', 'se_module.').replace('model.', ''): v for k, v in ckpt.items()}

    # Try loading weights
    try:
        model.load_state_dict(ckpt)
        print(f"✅ Loaded weights for {backbone}")
    except:
        try:
            ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            print(f"✅ Loaded from clean model for {backbone}")
        except:
            ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            print(f"✅ Fallback loaded for {backbone}")

    # Remove normalization if present
    if isinstance(model, nn.Sequential) and 'normalize' in model._modules:
        model = model._modules['model']

    torch.save(model.state_dict(), os.path.join(output_dir, f"robust_{equivalences[backbone]}.pt"))

# ----------------------------
# Process Madry ResNet-50
# ----------------------------
madry_ckpt_path = os.path.join(input_dir, "imagenet_linf_4.pt") #resnet50_linf_eps4.0.ckpt
state_dict = torch.load(madry_ckpt_path, map_location=torch.device('cpu') )

# Remove unnecessary keys
exact_keys_to_remove = {"module.normalizer.new_std", "module.normalizer.new_mean"}
prefixes_to_remove = ("module.attacker",)

filtered_state_dict = {
    k: v for k, v in state_dict["model"].items()
    if k not in exact_keys_to_remove and not any(k.startswith(prefix) for prefix in prefixes_to_remove)
}

# Clean key names
ckpt = {k.replace('module.model.', ''): v for k, v in filtered_state_dict.items()}

# Load into model
model = timm.create_model("resnet50", pretrained=False)
model.load_state_dict(ckpt)

# Save
torch.save(model.state_dict(), os.path.join(output_dir, "robust_resnet50.pt"))

# Test reload (optional)
state_dict = torch.load(os.path.join(output_dir, "robust_resnet50.pt"))
model = timm.create_model("resnet50", pretrained=False)
model.load_state_dict(state_dict)



# import torch.nn as nn
# import torch
# from torch import nn, Tensor
# from collections import OrderedDict
# from typing import Tuple
# from timm import create_model
# import timm

# class ImageNormalizer(nn.Module):
#     def __init__(self, mean: Tuple[float, float, float],
#         std: Tuple[float, float, float],
#         persistent: bool = True) -> None:
#         super(ImageNormalizer, self).__init__()

#         self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1),
#             persistent=persistent)
#         self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1),
#             persistent=persistent)

#     def forward(self, input: Tensor) -> Tensor:
#         return (input - self.mean) / self.std

# def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
#     std: Tuple[float, float, float]) -> nn.Module:
#     layers = OrderedDict([
#         ('normalize', ImageNormalizer(mean, std)),
#         ('model', model)
#     ])
#     return nn.Sequential(layers)

# IMAGENET_MEAN = [c * 1. for c in (0.485, 0.456, 0.406)] #[np.array([0., 0., 0.]), np.array([0.485, 0.456, 0.406])][-1] * 255
# IMAGENET_STD = [c * 1. for c in (0.229, 0.224, 0.225)] #[np.array([1., 1., 1.]), np.array([0.229, 0.224, 0.225])][-1] * 255

# equivalences = {
#     'weights_convnext_base':'convnext_base', 
#     'weights_vit_s':'deit_small_patch16_224',
#     'weights_convnext_t':'convnext_tiny',
#     'weights_vit_b_50_ep':'vit_base_patch16_224'
# }

# for backbone in [ 'weights_convnext_base', 'weights_vit_s', 'weights_convnext_t', 'weights_vit_b_50_ep' ]: 

#     model = create_model("timm/{}".format(equivalences[backbone]), pretrained=False)

#     ckpt = torch.load('/home/mheuillet/Desktop/{}.pt'.format(backbone), map_location='cpu', weights_only=False)

#     # Remove `normalize.` entries
#     ckpt = {k: v for k, v in ckpt.items() if "normalize." not in k }
#     ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
#     ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
#     ckpt = {k.replace('se_', 'se_module.'): v for k, v in ckpt.items()}
#     ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}

#     try:
#         model.load_state_dict(ckpt)
#         print('standard loading')

#     except:
#         try:
#             ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
#             model.load_state_dict(ckpt)
#             print('loaded from clean model')
#         except:
#             ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
#             # ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
#             model.load_state_dict(ckpt)
#             print('loaded')

#     if isinstance(model, nn.Sequential) and 'normalize' in model._modules: # remove normalization layer
#         # Rebuild the sequential model without the 'normalize' layer
#         model = model._modules['model']

#     torch.save(model.state_dict(), '/home/mheuillet/Desktop/state_dicts_share/robust_{}.pt'.format(equivalences[backbone]) )


# #########################################
# #########################################
# ######## PROCESS THE MADRY LAB RESNET50 ARCHITECTURE
# ##########################################


# state_dict = torch.load('/home/mheuillet/Desktop/resnet50_linf_eps4.0.ckpt', weights_only=False)
# # 2. Create a new dict with comprehension that excludes certain keys

# # Keys to remove exactly
# exact_keys_to_remove = {"module.normalizer.new_std", "module.normalizer.new_mean"}

# # Prefixes we want to remove if a key starts with them
# prefixes_to_remove = ("module.attacker",)  # tuple of prefixes, you can add more if needed

# filtered_state_dict = {
#     k: v
#     for k, v in state_dict["model"].items()
#     # Keep this entry only if:
#     # 1) it’s not in exact_keys_to_remove, AND
#     # 2) it doesn't start with any of the given prefixes
#     if k not in exact_keys_to_remove
#     and not any(k.startswith(prefix) for prefix in prefixes_to_remove)
# }

# ckpt = {k.replace('module.model.', ''): v for k, v in filtered_state_dict.items()}

# model = timm.create_model("resnet50", pretrained=False)
# model.load_state_dict(ckpt)

# torch.save(model.state_dict(), "/home/mheuillet/Desktop/state_dicts_share/robust_resnet50.pt")

# state_dict = torch.load("/home/mheuillet/Desktop/state_dicts_share/robust_resnet50.pt", weights_only=False)
# model = timm.create_model("resnet50", pretrained=False)
# model.load_state_dict(state_dict)
