import pickle as pkl
import gzip

from torchvision import datasets

print('download CIFAR10')

# datasets.CIFAR10(root='./data', train=True, download=True)
# datasets.CIFAR10(root='./data', train=False,  download=True)


from datasets import load_dataset ; dataset = load_dataset("frgfm/imagenette", "full_size", cache_dir='/home/mheuill/scratch')