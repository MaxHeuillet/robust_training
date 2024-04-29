import pickle as pkl
import gzip

from torchvision import datasets

print('download CIFAR10')

datasets.CIFAR10(root='./data', train=True, download=True)
datasets.CIFAR10(root='./data', train=False,  download=True)
