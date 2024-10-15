from torchvision import datasets


print('download aircraft')

datasets.FGVCAircraft(root='~/scratch/data', split='train', download=True, )
datasets.FGVCAircraft(root='~/scratch/data', split='val', download=True, )
datasets.FGVCAircraft(root='~/scratch/data', split='test', download=True, )

print('download CIFAR100')

datasets.CIFAR100(root='~/scratch/data', train=True, download=True)
datasets.CIFAR100(root='~/scratch/data', train=False,  download=True)

# print('download CIFAR10')

# datasets.CIFAR10(root='~/scratch/data', train=True, download=True)
# datasets.CIFAR10(root='~/scratch/data', train=False,  download=True)

# print('download MNIST')

# datasets.MNIST(root='~/scratch/data', train=True, download=True)
# datasets.MNIST(root='~/scratch/data', train=False,  download=True)



# from datasets import load_dataset 

# print('load')
# dataset = load_dataset("frgfm/imagenette", "full_size", cache_dir='/home/mheuill/scratch')
# print('save')
# dataset.save_to_disk('/home/mheuill/scratch/imagenette')


# print('load')
# dataset = load_dataset('imagenet-1k',  cache_dir='/home/mheuill/scratch')
# print('save')
# dataset.save_to_disk('/home/mheuill/scratch/imagenet-1k')