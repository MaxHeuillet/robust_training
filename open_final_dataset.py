import os
import tarfile
import zstandard as zstd
import tempfile
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import csv

from omegaconf import OmegaConf
from databases import load_data
import torch
from pathlib import Path
from torchvision import transforms
import shutil
import csv
import json
import tarfile
import zstandard as zstd
from PIL import Image
import os
import subprocess

class CSVDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        # Read CSV file
        with open(self.root / "labels.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.root / row["filename"]
                label = int(row["label"])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def load_tar_zst_dataset(config):

    # Load splits
    datasets_dict = {}
    for split in ["train", "val", "test", "test_common"]:
        split_path = Path(config.datasets_path) / split
        if split_path.exists():
            datasets_dict[split] = CSVDataset(split_path)

    # Return datasets and the tempdir (which will be deleted if not stored)
    return datasets_dict

if __name__ == "__main__":

    # === Load Config
    config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")

    for dataset_name in [
        'uc-merced-land-use-dataset',
        'flowers-102',
        'caltech101',
        'stanford_cars',
        'fgvc-aircraft-2013b',
        'oxford-iiit-pet'
    ]:

        print(f"📦 Processing: {dataset_name}")
        config.dataset = dataset_name

        # If the script is executable:
        subprocess.run(["./dataset_to_tmpdir_final.sh", dataset_name], check=True)

        dataset = load_tar_zst_dataset(config)


