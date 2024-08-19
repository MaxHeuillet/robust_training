# import torchdatasets as td
# from datasets.datasets import get_dataset

# class IndexedDataset(td.Dataset):
#     def __init__(self, args, train=True, train_transform=False):
#         super().__init__()
#         self.dataset = None #get_dataset(args, train=train, train_transform=train_transform)

#     def __getitem__(self, index):
#         data, target = self.dataset[index]
#         return data, target, index

#     def __len__(self):
#         return len(self.dataset)

#     def clean(self):
#         self._cachers = []


from torch.utils.data import Dataset
from PIL import Image
import io
from datasets.loaders import load_data

class IndexedDataset(Dataset): #Dataset

    def __init__(self, args, train=True): #, transform=None

        super().__init__()

        self.dataset = load_data(args, train=train) #, train_transform=train_transform

        self.dataset_name = args.dataset
        
        # self.transform = transform
        # self.dataset = dataset
        # self.imagenette_to_imagenet = {
        #         0: 0,    # n01440764
        #         1: 217,    # n02102040
        #         2: 491,    # n02979186
        #         3: 491,   # n03028079
        #         4: 497,   # n03394916
        #         5: 566,  # n03417042
        #         6: 569,  # n03425413
        #         7: 571,  # n03445777
        #         8: 574,  # n03888257
        #         9: 701   # n04251144  }
        
        # Initializing the targets attribute
        # if self.dataset_name == 'Imagenette':
        #     self.targets = [self.imagenette_to_imagenet[d['label']] for d in self.dataset]
        # elif self.dataset_name in ['Imagenet1k', 'CIFAR10', 'MNIST']:
        #     self.targets = [d[1] for d in self.dataset]  # Assuming the dataset is a list of tuples (image, label)
        # else:
        #     print('Dataset name error')

    def __len__(self):
        return len(self.dataset)
    
    def get_item_skip(self, idx):
        try:
            data = self.dataset[idx]
            if isinstance(data, tuple) and len(data) == 2:
                return data  # Unpack directly if data is correctly formatted
            else:
                raise ValueError(f"Unexpected data format at index {idx}: {data}")
        except Exception as e:
            next_idx = (idx + 1) % self.__len__()
            print(f"Error with index {idx}: {e}. Trying next index {next_idx}")
            return self.get_item_imagenet(next_idx)

    # def get_item_imagenette(self,idx):
    #     image = self.dataset[idx]['image']
    #     label = self.imagenette_to_imagenet[ self.dataset[idx]['label'] ]
    #     return image, label
    
    def get_item_noskip(self,idx):
        image, label = self.dataset[idx]
        return image, label
    
    def __getitem__(self, idx):

        if self.dataset_name == 'Imagenet1k':
            image_data,label = self.get_item_skip(idx)

        elif self.dataset_name in ['CIFAR10','MNIST', 'random']:
            image_data,label = self.get_item_noskip(idx)
        # elif self.dataset_name == 'Imagenette':
        #     image_data,label = self.get_item_imagenette(idx)
        else:    
            print('error')

        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        # if self.transform:
        #     #print("transorm appplied")
        #     image = self.transform(image)
        #print(image_data, image_data)
        
        return image, label, idx
    
    # def clean(self):
    #     self._cachers = []