from omegaconf import OmegaConf
from architectures import load_architecture

config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")

backbones=(
 'deit_small_patch16_224.fb_in1k.pt', 
 'robust_resnet50.pt', 
 'vit_base_patch16_224.mae.pt', 
 'vit_small_patch16_224.augreg_in21k.pt', 
 'convnext_base.fb_in1k.pt', 
 'resnet50.a1_in1k.pt', 
 'robust_vit_base_patch16_224.pt', 
 'vit_base_patch16_224.sam_in1k.pt', 
 'vit_small_patch16_224.dino.pt', 
 'convnext_base.fb_in22k.pt', 
 'robust_convnext_base.pt', 
 'vit_base_patch16_224.augreg_in1k.pt',
 'vit_base_patch16_224.augreg_in21k.pt', 
 'vit_base_patch16_224.dino.pt', 
 'vit_base_patch16_clip_224.laion2b.pt',
 'convnext_tiny.fb_in1k.pt',
 'robust_convnext_tiny.pt', 
 'robust_deit_small_patch16_224.pt',
 'vit_base_patch16_224.dino.pt', 
 'vit_small_patch16_224.augreg_in1k.pt', 
 'convnext_tiny.fb_in22k.pt' 
     )  

N = 10 # nombre de classes dans le dataset considere

for backbone in backbones:
    
    config.backbone = backbone
    hp_opt = False
    model = load_architecture(hp_opt, config, N, )
    print('architecture loaded')
    print(model)
    print("##############################")