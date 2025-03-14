from omegaconf import OmegaConf
from architectures import load_architecture

config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")

backbones=(
 'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
 'CLIP-convnext_base_w-laion2B-s13B-b82K',
 'deit_small_patch16_224.fb_in1k', 
 'robust_resnet50', 
 'vit_base_patch16_224.mae', 
 'vit_small_patch16_224.augreg_in21k', 
 'convnext_base.fb_in1k', 
 'resnet50.a1_in1k', 
 'robust_vit_base_patch16_224', 
 'vit_base_patch16_224.sam_in1k', 
 'vit_small_patch16_224.dino', 
 'convnext_base.fb_in22k', 
 'robust_convnext_base', 
 'vit_base_patch16_224.augreg_in1k',
 'vit_base_patch16_224.augreg_in21k', 
 'vit_base_patch16_224.dino', 
 'vit_base_patch16_clip_224.laion2b',
 'convnext_tiny.fb_in1k',
 'robust_convnext_tiny', 
 'robust_deit_small_patch16_224',
 'vit_base_patch16_224.dino', 
 'vit_small_patch16_224.augreg_in1k', 
 'convnext_tiny.fb_in22k' 
     )  

N = 10 # nombre de classes dans le dataset considere

for backbone in backbones:
    
    config.backbone = backbone
    hp_opt = False
    model = load_architecture(hp_opt, config, N, )
    print('architecture loaded')
    print(model)
    print("##############################")