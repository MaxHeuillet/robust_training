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

def save_split_as_folder(dataset, path, transform=None):
    path.mkdir(parents=True, exist_ok=True)
    label_path = path / "labels.csv"

    with open(label_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for i, (img, label) in enumerate(dataset):
            filename = f"{i:05d}.png"
            img_path = path / filename
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img.save(img_path)
            writer.writerow([filename, label])

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

    # Temporary folder to stage extracted contents
    temp_dir = config.datapath / f"tmp_{dataset_name}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    train_dataset, val_dataset, test_dataset, N = load_data(config, common_corruption=False)
    save_split_as_folder(train_dataset, temp_dir / "train")
    save_split_as_folder(val_dataset, temp_dir / "val")
    save_split_as_folder(test_dataset, temp_dir / "test")

    _, _, test_common_dataset, _ = load_data(config, common_corruption=True)
    save_split_as_folder(test_common_dataset, temp_dir / "test_common")

    # Write metadata.json
    metadata = {"N": N, "splits": {}}
    for split in ["train", "val", "test", "test_common"]:
        count = len(list((temp_dir / split).glob("*.png")))
        metadata["splits"][split] = {"count": count}
    with open(temp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Compress into .tar.zst
    archive_path = config.data_path / f"{dataset_name}_processed.tar.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(archive_path, "wb") as f_out:
        with cctx.stream_writer(f_out) as zst_stream:
            with tarfile.open(fileobj=zst_stream, mode="w|") as tar:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        fullpath = Path(root) / file
                        arcname = fullpath.relative_to(temp_dir)
                        tar.add(fullpath, arcname=str(arcname))

    print(f"✅ Saved archive to: {archive_path}")

    # Clean up temp dir
    shutil.rmtree(temp_dir)
