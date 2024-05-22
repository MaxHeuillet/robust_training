from transformers import AutoImageProcessor, ResNetModel
import torch

model = ResNetModel.from_pretrained("microsoft/resnet-50")

model.save_pretrained("/home/mheuill/scratch/resnet-50")
