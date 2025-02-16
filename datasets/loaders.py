from torchvision import datasets, transforms

import torch

from torch.utils.data import TensorDataset
import random
import numpy as np
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import random_split
from imagecorruptions import get_corruption_names, corrupt

from sklearn.model_selection import train_test_split
from utils import get_data_dir


#Change Grayscale Image to RGB for the shape
class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img
    
def stratified_subsample(dataset, sample_size=1500):
    """
    Subsample the given dataset to `sample_size` using stratification. 
    If `dataset` has fewer than `sample_size` samples, return dataset unchanged.
    """
    if len(dataset) < sample_size:
        print(f"Warning: test dataset has only {len(dataset)} samples. No subsampling performed.")
        return dataset
    
    # Extract labels for stratification
    labels = [label for _, label in dataset]

    # Perform stratified split to get exactly `sample_size` samples
    indices_subsample, _ = train_test_split(
        range(len(labels)),
        train_size=sample_size,
        stratify=labels,
        random_state=0
    )

    return Subset(dataset, indices_subsample)

# def apply_corruption(img, corruption_type, severity=3):
#     """Applies a specific corruption from imagecorruptions package."""
#     corrupted_img = corrupt(np.array(img), corruption_name=corruption_type, severity=severity)
#     return transforms.functional.to_pil_image(corrupted_img)

SAFE_CORRUPTIONS = [
    # "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    # "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    # "frost",
    # "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression"
]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_corruption(img, corruption_type, severity=3, max_attempts=3):
    """Ensures image is in the correct format before applying corruption."""
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()  
        img = np.transpose(img, (1, 2, 0))  
    
    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Convert from [0,1] to [0,255] range

    if corruption_type not in SAFE_CORRUPTIONS:
        # print(f"Warning: '{corruption_type}' is not in the safe list. Selecting a valid corruption...")
        corruption_type = random.choice(SAFE_CORRUPTIONS)

    # print(f"Applying corruption '{corruption_type}' with severity {severity}...")

    corrupted_img = corrupt(img, corruption_name=corruption_type, severity=severity)

    return DEFAULT_TRANSFORM(Image.fromarray(corrupted_img))

def apply_portfolio_of_corruptions(dataset, severity=3):
    """
    Applies a diversified portfolio of corruptions across the dataset.
    Each image receives exactly one type of corruption, based on index.
    Returns a TensorDataset for efficient sampling.
    """
    corruption_types = get_corruption_names()
    num_corruptions = len(corruption_types)
    
    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError("Dataset has 0 samples.")

    step_size = max(1, num_samples // num_corruptions)  # avoid div-by-zero

    # 1) Figure out the shape of the first sample so we can pre-allocate
    first_img, _ = dataset[0]
    if isinstance(first_img, torch.Tensor):
        # E.g., shape = (C, H, W)
        _, H, W = first_img.shape
        C = first_img.shape[0]
    else:
        # Convert to Tensor just to inspect shape
        first_img = DEFAULT_TRANSFORM(first_img)
        C, H, W = first_img.shape

    # 2) Pre-allocate large tensors for images and labels
    images = torch.empty((num_samples, C, H, W), dtype=torch.float32)
    labels = torch.empty((num_samples,), dtype=torch.long)

    # 3) Fill these tensors with corrupted data
    for i in range(num_samples):
        corruption_type = corruption_types[(i // step_size) % num_corruptions]
        img, label = dataset[i]
        corrupted_img = apply_corruption(img, corruption_type, severity=severity)
        images[i] = corrupted_img
        labels[i] = label

    # 4) Wrap in a TensorDataset so it's a valid PyTorch Dataset
    return TensorDataset(images, labels)

def load_data(hp_opt, config, common_corruption=False):

    dataset =config.dataset
    datadir = get_data_dir(hp_opt,config)

    train_transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to 224x224
                                          GrayscaleToRGB(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(.25,.25,.25),
                                          transforms.RandomRotation(2),
                                          transforms.ToTensor(),
                                          transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),])
    
    test_transform = [transforms.Resize((224, 224)),  # Resize images to 224x224
                       GrayscaleToRGB(),
                       transforms.ToTensor()]
    
    if not common_corruption:
        test_transform.append(transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ))

    transform = transforms.Compose(test_transform)

    if dataset == 'uc-merced-land-use-dataset':
        N = 21
        path = datadir +'/UCMerced_LandUse/Images'

        dataset_train_full = datasets.ImageFolder( root=path, transform=train_transform )
        dataset_val_test_full = datasets.ImageFolder( root=path,  transform=transform )

        labels = [label for _, label in dataset_train_full]

        train_val_indices, test_indices = train_test_split(
            range(len(labels)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_val_labels = [labels[i] for i in train_val_indices]

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.15,  # 0.15 * 0.8 = 0.12 -> 12% for validation
            stratify=train_val_labels,
            random_state=42
        )

        train_dataset = torch.utils.data.Subset(dataset_train_full, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_val_test_full, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_val_test_full, test_indices)

    elif dataset == 'kvasir-dataset':
        N = 8
        path = datadir+'/kvasir-dataset'
        dataset_train_full = datasets.ImageFolder( root=path, transform=train_transform )
        dataset_val_test_full = datasets.ImageFolder( root=path,  transform=transform )

        labels = [label for _, label in dataset_train_full]

        train_val_indices, test_indices = train_test_split(
            range(len(labels)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_val_labels = [labels[i] for i in train_val_indices]

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.15,  # 0.15 * 0.8 = 0.12 -> 12% for validation
            stratify=train_val_labels,
            random_state=42
        )

        train_dataset = torch.utils.data.Subset(dataset_train_full, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_val_test_full, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_val_test_full, test_indices)
                
    elif dataset == 'fgvc-aircraft-2013b': #fgvc-aircraft-2013b

        N = 100
        train_dataset = datasets.FGVCAircraft(root=datadir, split='train', download=False, transform=train_transform)
        val_dataset =   datasets.FGVCAircraft(root=datadir, split='val', download=False, transform=transform)
        test_dataset = datasets.FGVCAircraft(root=datadir, split='test', download=False, transform=transform)

    elif dataset == 'dtd':

        N = 47
        train_dataset = datasets.DTD(root=datadir, split='train', download=False, transform=train_transform)
        val_dataset =   datasets.DTD(root=datadir, split='val', download=False, transform=transform)
        test_dataset = datasets.DTD(root=datadir, split='test', download=False, transform=transform)

    elif dataset == 'eurosat':

        N = 10
        dataset_train_full = datasets.EuroSAT(root=datadir, download=False, transform=train_transform)
        dataset_val_test_full = datasets.EuroSAT(root=datadir, download=False, transform=transform)

        labels = [label for _, label in dataset_train_full]

        train_val_indices, test_indices = train_test_split(
            range(len(labels)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_val_labels = [labels[i] for i in train_val_indices]

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.15,  # 0.15 * 0.8 = 0.12 -> 12% for validation
            stratify=train_val_labels,
            random_state=42
        )

        train_dataset = torch.utils.data.Subset(dataset_train_full, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_val_test_full, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_val_test_full, test_indices)

    elif dataset == 'flowers-102':
            
        # Load the Flowers102 dataset
        N = 102
        train_dataset = datasets.Flowers102(root=datadir, split='train', download=False, transform=train_transform)
        val_dataset = datasets.Flowers102(root=datadir, split='val', download=False, transform=transform)
        test_dataset = datasets.Flowers102(root=datadir, split='test', download=False,  transform=transform)

    elif dataset == 'imagenette2':

        N = 10
        print('datadir', datadir)
        train_dataset = datasets.Imagenette(root=datadir, split='train', download=False,  transform=train_transform)
        dataset = datasets.Imagenette(root=datadir, split='val', download=False,  transform=transform)
        labels = [label for _, label in dataset]

        test_indices, val_indices = train_test_split( range(len(labels)), test_size=0.25, stratify=labels, random_state=42 )  

        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

    elif dataset == 'caltech101':

        N = 101
        dataset_train_full = datasets.Caltech101(root=datadir, download=False, transform=train_transform)
        dataset_val_test_full = datasets.Caltech101(root=datadir, download=False, transform=transform)

        labels = [label for _, label in dataset_train_full]

        train_val_indices, test_indices = train_test_split(
            range(len(labels)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_val_labels = [labels[i] for i in train_val_indices]

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.15,  # 0.15 * 0.8 = 0.12 -> 12% for validation
            stratify=train_val_labels,
            random_state=42
        )

        train_dataset = torch.utils.data.Subset(dataset_train_full, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_val_test_full, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_val_test_full, test_indices)

    elif dataset == 'oxford-iiit-pet':

        N = 37

        # Step 1: Load the 'trainval' split twice with different transforms
        dataset_train_val_full_train = datasets.OxfordIIITPet(
            root=datadir,
            split='trainval',
            download=False,
            transform=train_transform
        )

        dataset_train_val_full_val = datasets.OxfordIIITPet(
            root=datadir,
            split='trainval',
            download=False,
            transform=transform
        )

        # Step 2: Load the 'test' split with transform
        dataset_test_full = datasets.OxfordIIITPet(
            root=datadir,
            split='test',
            download=False,
            transform=transform
        )

        # Step 3: Extract labels from the 'trainval' split
        labels = [label for _, label in dataset_train_val_full_train]

        # Step 4: Split 'trainval' into 'train' and 'val' using stratified sampling
        train_indices, val_indices = train_test_split(
            range(len(labels)),
            test_size=0.15,  # 15% for validation
            stratify=labels,
            random_state=42
        )

        # Step 5: Create training and validation subsets with appropriate transforms
        train_dataset = torch.utils.data.Subset(dataset_train_val_full_train, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_train_val_full_val, val_indices)
        test_dataset = dataset_test_full

    elif dataset == 'stanford_cars':

        N = 196
        print('datadir', datadir)

        # Step 1: Load the 'train' split with train_transform
        dataset_train_full_train = datasets.StanfordCars(
            root=datadir,
            split='train',
            download=False,
            transform=train_transform
        )

        # Step 2: Load the 'train' split again with transform for validation
        dataset_train_full_val = datasets.StanfordCars(
            root=datadir,
            split='train',
            download=False,
            transform=transform
        )

        # Step 3: Load the 'test' split with transform
        dataset_test_full = datasets.StanfordCars(
            root=datadir,
            split='test',
            download=False,
            transform=transform
        )

        # Step 4: Extract labels from the 'train' split
        labels = [label for _, label in dataset_train_full_train]

        # Step 5: Split 'train' into 'train' and 'val' using stratified sampling
        train_indices, val_indices = train_test_split(
            range(len(labels)),
            test_size=0.15,  # 15% for validation
            stratify=labels,
            random_state=42
        )

        # Step 6: Create training and validation subsets with appropriate transforms
        train_dataset = torch.utils.data.Subset(dataset_train_full_train, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_train_full_val, val_indices)
        test_dataset = dataset_test_full
    
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    test_dataset = stratified_subsample(test_dataset, sample_size=1500)

    if common_corruption:
        
        test_dataset = apply_portfolio_of_corruptions(test_dataset, severity=3)
                 
    return train_dataset, val_dataset, test_dataset, N, train_transform, transform