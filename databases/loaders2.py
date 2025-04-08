import os

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import csv

from pathlib import Path
import csv
import json

from PIL import Image
import os

from transforms import load_data_transforms

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

def load_data2(config):

    train_transform, transform = load_data_transforms()

    base_path = Path(os.environ["SLURM_TMPDIR"]) / "data" / config.dataset

    train_dataset = CSVDataset(base_path / "train", train_transform)
    val_dataset = CSVDataset(base_path / "val", transform)
    test_dataset = CSVDataset(base_path / "test", transform)
    test_common_dataset = CSVDataset(base_path / "test_common", transform)

    # Load metadata.json
    metadata_path = base_path / "metadata.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    N = metadata.get("N", None)
    
    return train_dataset, val_dataset, test_dataset, test_common_dataset, N