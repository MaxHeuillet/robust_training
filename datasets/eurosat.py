import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Traverse the directory and collect image paths
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):  # Adjust if images have different extensions
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
