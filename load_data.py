import pickle as pkl
import gzip

from torchvision import datasets


from datasets import load_dataset 

print('load')
dataset = load_dataset("frgfm/imagenette", "full_size", cache_dir='/home/mheuill/scratch')
print('save')
dataset.save_to_disk('/home/mheuill/scratch/imagenette')


# print('load')
# dataset = load_dataset('imagenet-1k',  cache_dir='/home/mheuill/scratch')
# print('save')
# dataset.save_to_disk('/home/mheuill/scratch/imagenet-1k')

# print('download CIFAR10')

# datasets.CIFAR10(root='/home/mheuill/scratch/data', train=True, download=True)
# datasets.CIFAR10(root='/home/mheuill/scratch/data', train=False,  download=True)