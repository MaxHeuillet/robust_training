
from torch.utils.data import Dataset
from PIL import Image
import os
import io
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode


class CustomImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        # if self.dataset == 'CIFAR10':
        # elif self.dataset == 'Imagenette':
        # elif self.dataset == 'Imagenet1k':

        self.imagenette_to_imagenet = {
                0: 0,    # n01440764
                1: 217,    # n02102040
                2: 491,    # n02979186
                3: 491,   # n03028079
                4: 497,   # n03394916
                5: 566,  # n03417042
                6: 569,  # n03425413
                7: 571,  # n03445777
                8: 574,  # n03888257
                9: 701   # n04251144 
                }

    def __len__(self):
        return len(self.hf_dataset)
    
    def get_item_imagenet(self,idx):
        image,label = self.hf_dataset[idx]
        return image,label

    def get_item_imagenette(self,idx):
        image = self.hf_dataset[idx]['image']
        label = self.imagenette_to_imagenet[ self.hf_dataset[idx]['label'] ]
        return image, label

    
    def __getitem__(self, idx):
        if self.data == 'Imagenet1k':
            image_data,label = self.get_item_imagenet(idx)
        elif self.data == 'Imagenette':
            image_data,label = self.get_item_imagenette(idx)
        else:
            print('error')
        
        # If the images are being loaded as PIL Images, apply the transform
        # Make sure that the image is opened correctly if it's not already a PIL Image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        if self.transform:
            #print("transorm appplied")
            image = self.transform(image)

        #print(image_data, image_data)
        
        return image, label, idx


def to_rgb(x):
    return x.convert("RGB")


# if __name__ == "__main__":
transform = transforms.Compose([
                    transforms.Lambda(to_rgb),
                    transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),  # Resize the image to 232x232
                    transforms.CenterCrop(224),  # Crop the center of the image to make it 224x224
                    transforms.ToTensor(),  # Convert the image to a tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
                ])

# print('load train')
#     # dataset = load_dataset("imagenet-1k", cache_dir='/home/mheuill/scratch',)
# train_folder = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/train'))
print('load test')
val_folder = datasets.ImageFolder(os.path.join(os.environ['SLURM_TMPDIR'], 'data/imagenet/val'))
# print('load custom train')
# pool_dataset = CustomImageDataset(train_folder, transform= transform )
print('load custom test')        
val_dataset = CustomImageDataset(val_folder, transform= transform) 

print(val_dataset[0])