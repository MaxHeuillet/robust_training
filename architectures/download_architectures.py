# from transformers import AutoImageProcessor, ResNetModel
import torch
import timm
from timm.models import create_model
import os

save_path = os.path.expanduser('~/scratch/state_dicts_share')

backbones=(
    'timm/vit_base_patch16_224.dino',
    'timm/vit_base_patch16_224.mae',
    'timm/vit_base_patch16_224.sam_in1k',
    'timm/vit_base_patch16_clip_224.laion400m_e32',
    'timm/vit_base_patch16_clip_224.laion2b',
    'timm/vit_base_patch16_224.augreg_in21k',
    'timm/vit_base_patch16_224.augreg_in1k',

    'timm/vit_small_patch16_224.dino',
    'timm/vit_small_patch16_224.augreg_in21k',
    'timm/vit_small_patch16_224.augreg_in1k',
    'timm/deit_small_patch16_224.fb_in1k',

    'laion/CLIP-convnext_base_w-laion2B-s13B-b82K',
    'laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
    'timm/convnext_base.fb_in1k',
    'timm/convnext_base.fb_in22k',

    'timm/convnext_tiny.fb_in22k',
    'timm/convnext_tiny.fb_in1k',

    'timm/resnet50.a1_in1k', )  


for backbone in backbones:
    parts = backbone.split("/")

    model = timm.create_model(backbone, pretrained=True)
    torch.save(model.state_dict(), '{}/{}.pt'.format(save_path, parts[1]) )

######################################################################################################################
#### To download the robust checkpoints trained on imagenet ('robust_convnext_tiny', 'robust_deit_small_patch16_224', 'robust_convnext_base', 'robust_vit_base_patch16_224' )
#### Go to this link: "https://nc.mlcloud.uni-tuebingen.de/index.php/s/XLLnoCnJxp74Zqn"
#### This is from official github (https://github.com/nmndeep/revisiting-at?tab=readme-ov-file) of "Revisiting Adversarial Training for Imagenet" (Neurips 2023) paper.
#### The following code aims to reformat the architecture to its native format (remove image normalization from the model, rename modules):
######################################################################################################################

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

# backbone = 'vit_base_patch16_224'

# model = create_model(backbone, pretrained=False)
# # model = timm.create_model(backbone, pretrained=False)
# # model = normalize_model(model, IMAGENET_MEAN, IMAGENET_STD)

# ckpt = torch.load('./state_dicts/weights_vit_b_50_ep.pt', map_location='cpu', weights_only=False)
# ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
# ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
# ckpt = {k.replace('se_', 'se_module.'): v for k, v in ckpt.items()}

# try:
#     model.load_state_dict(ckpt)
#     print('standard loading')

# except:
#     try:
#         ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
#         model.load_state_dict(ckpt)
#         print('loaded from clean model')
#     except:
#         ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
#         # ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
#         model.load_state_dict(ckpt)
#         print('loaded')

# if isinstance(model, nn.Sequential) and 'normalize' in model._modules: # remove normalization layer
#     # Rebuild the sequential model without the 'normalize' layer
#     model = model._modules['model']

# torch.save(model.state_dict(), './state_dicts/robust_{}.pt'.format(backbone) )