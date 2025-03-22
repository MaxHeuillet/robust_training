import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

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

    return torch.utils.data.Subset(dataset, indices_subsample)

def process_trainvaltest(dataset_train_full, ):

    labels = [dataset_train_full[i][1] for i in range(len(dataset_train_full))]

    train_val_indices, test_indices = train_test_split(
                                            range(len(labels)),
                                            test_size=0.2,
                                            stratify=labels,
                                            random_state=42
                                        )

    train_val_labels = [labels[i] for i in train_val_indices]

    train_indices, val_indices = train_test_split(
                                            train_val_indices,
                                            test_size=0.15,  
                                            stratify=train_val_labels,
                                            random_state=42
                                        )

    train_dataset = torch.utils.data.Subset(dataset_train_full, train_indices)
    val_dataset = torch.utils.data.Subset(dataset_train_full, val_indices)
    test_dataset = torch.utils.data.Subset(dataset_train_full, test_indices)

    return train_dataset, val_dataset, test_dataset

def process_trainval(dataset_train_val_full, ):

    labels = [dataset_train_val_full[i][1] for i in range(len(dataset_train_val_full))]

    train_indices, val_indices = train_test_split(
            range(len(labels)),
            test_size=0.15,  # 15% for validation
            stratify=labels,
            random_state=42)

    # Step 5: Create training and validation subsets with appropriate transforms
    train_dataset = torch.utils.data.Subset(dataset_train_val_full, train_indices)
    val_dataset = torch.utils.data.Subset(dataset_train_val_full, val_indices)

    return train_dataset, val_dataset