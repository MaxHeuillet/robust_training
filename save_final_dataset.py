import argparse
from omegaconf import OmegaConf
from databases.loaders import load_data
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

# === Parse command-line arguments
parser = argparse.ArgumentParser(description="Preprocess datasets and save to tar.zst")
parser.add_argument("--datasets_path", type=str, default=str(Path.home() / "scratch" / "data"),
                    help="Directory where raw datasets and processed outputs will be stored.")
args = parser.parse_args()

# === Mock Config
config = OmegaConf.create({
    "datasets_path": args.datasets_path,
    "dataset": None
})

# === Dataset list
for dataset_name in [
    # 'uc-merced-land-use-dataset',
    'flowers-102',
    # 'caltech101',
    # 'stanford_cars',
    # 'fgvc-aircraft-2013b',
    # 'oxford-iiit-pet'
]:
    print(f"📦 Processing: {dataset_name}")
    config.dataset = dataset_name

    # Paths
    temp_dir = Path(config.datasets_path) / f"tmp_{dataset_name}"
    archive_name = f"{dataset_name}_processed.tar.zst"
    tmp_archive_path = Path(config.datasets_path) / f"tmp_{archive_name}"
    final_archive_path = Path(config.datasets_path) / archive_name

    # Cleanup old tmp directory if exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    # Load data
    train_dataset, val_dataset, test_dataset, N = load_data(config, common_corruption=False)
    save_split_as_folder(train_dataset, temp_dir / "train")
    save_split_as_folder(val_dataset, temp_dir / "val")
    save_split_as_folder(test_dataset, temp_dir / "test")

    _, _, test_common_dataset, _ = load_data(config, common_corruption=True)
    save_split_as_folder(test_common_dataset, temp_dir / "test_common")

    # Metadata
    metadata = {"N": N, "splits": {}}
    for split in ["train", "val", "test", "test_common"]:
        count = len(list((temp_dir / split).glob("*.png")))
        metadata["splits"][split] = {"count": count}
    with open(temp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Archive with .tar.zst
    cctx = zstd.ZstdCompressor(level=3)
    with open(tmp_archive_path, "wb") as f_out:
        with cctx.stream_writer(f_out) as zst_stream:
            with tarfile.open(fileobj=zst_stream, mode="w|") as tar:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        fullpath = Path(root) / file
                        arcname = fullpath.relative_to(temp_dir)
                        tar.add(fullpath, arcname=str(arcname))

    # Move to final location (same dir in this case)
    shutil.move(str(tmp_archive_path), str(final_archive_path))
    print(f"✅ Archive moved to: {final_archive_path}")

    # Clean up
    shutil.rmtree(temp_dir)


