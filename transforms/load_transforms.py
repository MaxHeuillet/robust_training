
import torchvision.transforms as transforms

#Change Grayscale Image to RGB for the shape
class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img

def load_data_transforms():

    train_transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to 224x224
                                          GrayscaleToRGB(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(.25,.25,.25),
                                          transforms.RandomRotation(2),
                                          transforms.ToTensor(),])
    
    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to 224x224
                                    GrayscaleToRGB(),
                                    transforms.ToTensor(),])
    
    return train_transform, transform

def load_module_transform(config):
        
    if 'laion2b' in config.backbone:
        # https://github.com/openai/CLIP/issues/20?utm_source=chatgpt.com
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
    else:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

    return mean, std

    

