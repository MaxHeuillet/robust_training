import numpy as np
from PIL import Image
from .corruptions import *
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Define a Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.dataset = data
        self.transform = transform  # Apply transformations (if any)

    def __len__(self):
        return len(self.dataset[0])  # Return dataset size

    def __getitem__(self, idx):
        img,label = self.dataset[idx]

        # Convert NumPy/PIL images to Tensor if needed
        if isinstance(img, np.ndarray):  # If NumPy array, convert to PIL
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)  # Apply transforms

        return img, label

def apply_portfolio_of_corruptions(dataset, severity=3):
    """
    Applies a diversified portfolio of corruptions across the dataset.
    Each image receives exactly one type of corruption, based on index.
    Returns a TensorDataset for efficient sampling.
    """
    corruption_types =  [ # "gaussian_noise",
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
                            "jpeg_compression"  ]

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
        first_img = transforms.ToTensor()(first_img)
        C, H, W = first_img.shape

    # 2) Pre-allocate large tensors for images and labels
    data = [-1] * num_samples

    # 3) Fill these tensors with corrupted data
    for i in range(num_samples):
        corruption_type = corruption_types[(i // step_size) % num_corruptions]
        img, label = dataset[i]
        corrupted_img = apply_corruption(img, corruption_type, severity=severity)
        
        data[i] = (corrupted_img,label)

    # 4) Wrap in a TensorDataset so it's a valid PyTorch Dataset
    return  CustomDataset( data )


def apply_corruption(img, corruption_type, severity=3, max_attempts=3):
    """Ensures image is in the correct format before applying corruption."""
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()  
        img = np.transpose(img, (1, 2, 0))  
    
    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Convert from [0,1] to [0,255] range

    corrupted_img = corrupt(img, corruption_name=corruption_type, severity=severity)

    return Image.fromarray(corrupted_img)





corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                    brightness, contrast, elastic_transform, pixelate,
                    jpeg_compression, speckle_noise, gaussian_blur, spatter,
                    saturate)

corruption_dict = {corr_func.__name__: corr_func for corr_func in
                   corruption_tuple}


def corrupt(image, severity=1, corruption_name=None, corruption_number=-1):
    """This function returns a corrupted version of the given image.

    Args:
        image (numpy.ndarray):      image to corrupt; a numpy array in [0, 255], expected datatype is np.uint8
                                    expected shape is either (height x width x channels) or (height x width);
                                    width and height must be at least 32 pixels;
                                    channels must be 1 or 3;
        severity (int):             strength with which to corrupt the image; an integer in [1, 5]
        corruption_name (str):      specifies which corruption function to call, must be one of
                                        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                                        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                                        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                                    the last four are validation corruptions
        corruption_number (int):    the position of the corruption_name in the above list; an integer in [0, 18];
                                        useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    Returns:
        numpy.ndarray:              the image corrupted by a corruption function at the given severity; same shape as input
    """

    if not isinstance(image, np.ndarray):
        raise AttributeError('Expecting type(image) to be numpy.ndarray')
    if not (image.dtype.type is np.uint8):
        raise AttributeError('Expecting image.dtype.type to be numpy.uint8')

    if not (image.ndim in [2,3]):
        raise AttributeError('Expecting image.shape to be either (height x width) or (height x width x channels)')
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)

    height, width, channels = image.shape

    if (height < 32 or width < 32):
        raise AttributeError('Image width and height must be at least 32 pixels')

    if not (channels in [1, 3]):
        raise AttributeError('Expecting image to have either 1 or 3 channels (last dimension)')

    if channels == 1:
        image = np.stack((np.squeeze(image),)*3, axis=-1)

    if severity not in [1, 2, 3, 4, 5]:
        raise AttributeError('Severity must be an integer in [1, 5]')

    if not (corruption_name is None):
        image_corrupted = corruption_dict[corruption_name](Image.fromarray(image),
                                                           severity)
    elif corruption_number != -1:
        image_corrupted = corruption_tuple[corruption_number](Image.fromarray(image),
                                                              severity)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(image_corrupted)


# def get_corruption_names(subset='common'):
#     if subset == 'common':
#         return [f.__name__ for f in corruption_tuple[:15]]
#     elif subset == 'validation':
#         return [f.__name__ for f in corruption_tuple[15:]]
#     elif subset == 'all':
#         return [f.__name__ for f in corruption_tuple]
#     elif subset == 'noise':
#         return [f.__name__ for f in corruption_tuple[0:3]]
#     elif subset == 'blur':
#         return [f.__name__ for f in corruption_tuple[3:7]]
#     elif subset == 'weather':
#         return [f.__name__ for f in corruption_tuple[7:11]]
#     elif subset == 'digital':
#         return [f.__name__ for f in corruption_tuple[11:15]]
#     else:
#         raise ValueError("subset must be one of ['common', 'validation', 'noise', 'blur', 'weather', 'digital', 'all']")