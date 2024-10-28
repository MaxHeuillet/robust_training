# import os
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

# class EuroSATDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []

#         # Traverse the directory and collect image paths
#         for root, _, files in os.walk(root_dir):
#             for file in files:
#                 if file.endswith('.jpg'):  # Adjust if images have different extensions
#                     self.image_paths.append(os.path.join(root, file))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path)

#         if self.transform:
#             image = self.transform(image)

#         return image


import os
from PIL import Image
from torch.utils.data import Dataset

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Create a mapping from class names to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}

        # Traverse the directory and collect image paths and labels
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):  # Adjust if images have different extensions
                    self.image_paths.append(os.path.join(root, file))
                    # Extract label index from directory structure
                    class_name = os.path.basename(root)
                    label = self.class_to_idx[class_name]
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]  # Corresponding label index

        return image, label
