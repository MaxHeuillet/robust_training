from datasets import load_dataset
print('load')
dataset = load_dataset('imagenet-1k',  cache_dir='/home/mheuill/scratch')
print('save')
dataset.save_to_disk('/home/mheuill/scratch/imagenet-1k')
