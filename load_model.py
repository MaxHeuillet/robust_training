from transformers import AutoImageProcessor, ResNetModel
import torch

model = ResNetModel.from_pretrained("microsoft/resnet-50")

model.save_pretrained("/home/mheuill/scratch/resnet-50")


from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
image_processor.save_pretrained("/home/mheuill/scratch/resnet-50")