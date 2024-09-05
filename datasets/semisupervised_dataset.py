import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset

class SemiSupervisedDataset(Dataset):
    """
    A dataset with auxiliary pseudo-labeled data.
    """
    def __init__(self, args, train=False, **kwargs):

        self.args = args
        self.train = train
        self.load_base_dataset( **kwargs)
        
        if self.train:

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            aux = np.load('./data/1m.npz')
            aux_data = aux['image']
            print(aux_data.shape)
            aux_targets = aux['label']
                
            orig_len = len(self.data)
            print(orig_len)

            self.data = np.concatenate((self.data, aux_data), axis=0)
            print('concatenation')
            
            self.targets.extend(aux_targets)
            self.unsup_indices.extend(range(orig_len, orig_len+len(aux_data)))

        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

    def load_base_dataset(self, **kwargs):
        
        if self.args.dataset == 'CIFAR10s':
            self.dataset = datasets.CIFAR10(root=self.args.data_dir, train=self.train, **kwargs)
            self.dataset_size = len(self.dataset)
        else:
            print('specify dataset')
        
    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        return self.dataset[item]