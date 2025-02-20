from torchvision import datasets
from corruptions import apply_portfolio_of_corruptions
from datasets.data_processing import process_trainvaltest, process_trainval, stratified_subsample
from utils import get_data_dir
from transforms import load_data_transforms

def load_data(hp_opt, config, common_corruption=False):

    dataset =config.dataset
    datadir = get_data_dir(hp_opt,config)

    train_transform, transform = load_data_transforms()

    if dataset == 'uc-merced-land-use-dataset':

        N = 21
        path = datadir +'/UCMerced_LandUse/Images'
        dataset_train_full = datasets.ImageFolder( root=path )
        train_dataset, val_dataset, test_dataset = process_trainvaltest(dataset_train_full, )
 
    elif dataset == 'kvasir-dataset':

        N = 8
        path = datadir+'/kvasir-dataset'
        dataset_train_full = datasets.ImageFolder( root=path  )
        train_dataset, val_dataset, test_dataset = process_trainvaltest(dataset_train_full, )

    elif dataset == 'caltech101':

        N = 101
        dataset_train_full = datasets.Caltech101(root=datadir, download=False, )
        train_dataset, val_dataset, test_dataset = process_trainvaltest(dataset_train_full, )
                
    elif dataset == 'fgvc-aircraft-2013b': 

        N = 100
        train_dataset = datasets.FGVCAircraft(root=datadir, split='train', download=False, )
        val_dataset =   datasets.FGVCAircraft(root=datadir, split='val', download=False, )
        test_dataset = datasets.FGVCAircraft(root=datadir, split='test', download=False, )

    elif dataset == 'dtd':
        
        N = 47
        train_dataset = datasets.DTD(root=datadir, split='train', download=False, )
        val_dataset =   datasets.DTD(root=datadir, split='val', download=False, )
        test_dataset = datasets.DTD(root=datadir, split='test', download=False, )

    elif dataset == 'flowers-102':
        
        N = 102
        train_dataset = datasets.Flowers102(root=datadir, split='train', download=False)
        val_dataset = datasets.Flowers102(root=datadir, split='val', download=False)
        test_dataset = datasets.Flowers102(root=datadir, split='test', download=False)

    elif dataset == 'oxford-iiit-pet':

        N = 37
        dataset_train_val_full = datasets.OxfordIIITPet(root=datadir, split='trainval', download=False,)
        dataset_test_full = datasets.OxfordIIITPet(root=datadir,split='test',download=False, )
        train_dataset, val_dataset = process_trainval(dataset_train_val_full, )
        test_dataset = dataset_test_full

    elif dataset == 'stanford_cars':

        N = 196
        dataset_train_val_full = datasets.StanfordCars( root=datadir, split='train', download=False,)
        dataset_test_full = datasets.StanfordCars( root=datadir, split='test', download=False, )
        train_dataset, val_dataset = process_trainval(dataset_train_val_full, )
        test_dataset = dataset_test_full
    
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    test_dataset = stratified_subsample(test_dataset, sample_size=1500)

    if common_corruption:
        test_dataset = apply_portfolio_of_corruptions(test_dataset, severity=3)
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = transform
        test_dataset.transform = transform

    else:
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = transform
        test_dataset.dataset.transform = transform



      
                 
    return train_dataset, val_dataset, test_dataset, N

