# from transformers import AutoImageProcessor, ResNetModel
import torch
import timm
from timm.models import create_model
import os

save_path = os.path.expanduser('~/scratch/state_dicts')


model = timm.create_model('resnet50', pretrained=True)
torch.save(model.state_dict(), './state_dicts/timm_resnet50_imagenet1k.pt')

model = timm.models.convnext.convnext_tiny(pretrained=True)
torch.save(model.state_dict(), './state_dicts/timm_convnext_imagenet1k.pt')

model = create_model('deit_tiny_patch16_224', pretrained=True)
torch.save(model.state_dict(), './state_dicts/timm_deit_tiny_patch16_224_imagenet1k.pt')

# model = ResNetModel.from_pretrained("microsoft/resnet-50")
# model.save_pretrained("/home/mheuill/scratch/resnet-50")

# from transformers import AutoImageProcessor
# image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# image_processor.save_pretrained("/home/mheuill/scratch/resnet-50")