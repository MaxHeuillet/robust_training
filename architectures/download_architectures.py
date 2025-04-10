# from transformers import AutoImageProcessor, ResNetModel
import torch
import os
from timm import create_model
from huggingface_hub import hf_hub_download

save_path = os.path.expanduser('/home/mheuillet/Desktop/state_dicts_share')
# save_path = os.path.expanduser('~/scratch/state_dicts_share')
os.makedirs(save_path, exist_ok=True)

############## SET OF SCIENTIFIC BACKBONES

backbones = (

    # 'timm/vit_base_patch16_224.dino',
    # 'timm/vit_base_patch16_224.mae',
    # 'timm/vit_base_patch16_clip_224.laion400m_e32',
    # 'timm/vit_base_patch16_clip_224.laion2b',
    # 'timm/vit_base_patch16_224.augreg_in21k',
    # 'timm/vit_base_patch16_224.augreg_in1k',
    # 'timm/vit_small_patch16_224.augreg_in21k',
    # 'timm/vit_small_patch16_224.augreg_in1k',
    # 'timm/deit_small_patch16_224.fb_in1k',
    # 'laion/CLIP-convnext_base_w-laion2B-s13B-b82K',
    # 'laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
    # 'timm/convnext_base.fb_in1k',
    # 'timm/convnext_base.fb_in22k',
    # 'timm/convnext_tiny.fb_in22k',
    # 'timm/convnext_tiny.fb_in1k',
    # 'timm/resnet50.a1_in1k', 

    # 'timm/vit_base_patch16_clip_224.laion2b_ft_in1k',
    # 'timm/vit_base_patch16_224.augreg_in21k_ft_in1k',
    # 'timm/vit_small_patch16_224.augreg_in21k_ft_in1k',
    # 'timm/eva02_base_patch14_224.mim_in22k',
    # 'timm/eva02_tiny_patch14_224.mim_in22k',
    # 'timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    # 'timm/swin_tiny_patch4_window7_224.ms_in1k',
    # 'timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k',
    # 'timm/convnext_base.fb_in22k_ft_in1k',
    # 'timm/convnext_tiny.fb_in22k_ft_in1k',

    # 'timm/regnetx_004.pycls_in1k',
    # 'google/efficientnet-b0',
    # 'timm/deit_tiny_patch16_224.fb_in1k',
    # 'apple/mobilevit-small',
    # 'timm/mobilenetv3_large_100.ra_in1k',
    # 'timm/edgenext_small.usi_in1k'
    # 'timm/coat_tiny.in1k',

    # 'timm/coatnet_0_rw_224.sw_in1k',
    # 'timm/coatnet_2_rw_224.sw_in12k_ft_in1k',
    # 'timm/coatnet_2_rw_224.sw_in12k'
)



for backbone in backbones:
    parts = backbone.split("/")
    model_source = parts[0]
    model_name = parts[1]
    print('model_source', model_source)

    save_file = os.path.join(save_path, f"{model_name}.pt")

    if model_source == "laion" or backbone == "vit_base_patch16_clip_224.laion400m_e32":
        try:
            file = hf_hub_download(repo_id=backbone, filename="open_clip_pytorch_model.bin")
            state_dict = torch.load(file, map_location="cpu")
            torch.save(state_dict, save_file)
        except Exception as e:
            print(f"❌ Failed to download {backbone}: {e}")

    elif model_source == "timm":
        print(backbone)
        try:
            model = create_model(backbone, pretrained=True)
            torch.save(model.state_dict(), save_file)
        except Exception as e:
            print(f"❌ Failed to download {backbone}: {e}")

    elif model_source in {"google", "apple"} or backbone == "timm/vit_base_patch16_clip_224.laion400m_e32":
        print(backbone)
        try:
            # Try downloading both .safetensors and .bin files (whichever is available)
            try:
                file = hf_hub_download(repo_id=backbone, filename="model.safetensors")
                state_dict = torch.load(file, map_location="cpu")
            except:
                file = hf_hub_download(repo_id=backbone, filename="pytorch_model.bin")
                state_dict = torch.load(file, map_location="cpu")

            
            torch.save(state_dict, save_file)
            print(f"✅ Successfully saved {backbone}")

        except Exception as e:
            print(f"❌ Failed to download {backbone}: {e}")

    else:
        print(f"Unknown source for backbone: {backbone}")