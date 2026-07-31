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

    tmpdir = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    dataset_path = tmpdir / "data" / job_id / config.dataset

    train_dataset = CSVDataset(dataset_path / "train", train_transform)
    val_dataset = CSVDataset(dataset_path / "val", transform)
    test_dataset = CSVDataset(dataset_path / "test", transform)
    # test_common/ was intentionally stripped from the RobustGenBench archives
    # (see git history: clean_archives_remove_test_common.py) once common-
    # corruption evaluation moved to a separate eval-time download. Not read
    # during training (discarded by initialize_loaders), so tolerate absence.
    test_common_path = dataset_path / "test_common"
    test_common_dataset = (CSVDataset(test_common_path, transform)
                            if (test_common_path / "labels.csv").exists() else None)

    # Load metadata.json
    metadata_path = dataset_path / "metadata.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    N = metadata.get("N", None)
    
    return train_dataset, val_dataset, test_dataset, test_common_dataset, N