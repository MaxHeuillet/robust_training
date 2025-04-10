import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path  # In case config.dataset_path is a string



def move_dataset_to_tmpdir(config):

    dataset_name = config.dataset
    dataset_path = Path(config.dataset_path).expanduser().resolve()
    
    # Fixed archive path construction
    archive_path = dataset_path / f"{dataset_name}_processed.tar.zst"
    
    tmpdir = config.work_path
    dest_dir = os.path.join(tmpdir, "data", dataset_name)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"📦 Extracting {archive_path} into {dest_dir} using tar + zstd...")
    
    try:
        subprocess.run(["tar", "-I", "zstd", "-xf", archive_path, "-C", dest_dir], check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"❌ Failed to extract archive: {archive_path}")

    print(f"✅ Extraction of {dataset_name} completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return dest_dir

def move_architecture_to_tmpdir(config):

    backbone = config.backbone
    statedict_dir = Path(config.statedicts_path).expanduser().resolve()
    checkpoint_path = statedict_dir / f"{backbone}.pt"

    tmpdir = Path(config.work_path)
    dest_dir = tmpdir
    dest_path = dest_dir / f"{backbone}.pt"

    if dest_path.exists():
        dest_path.unlink()  # Delete existing file

    shutil.move(str(checkpoint_path), str(dest_dir) )

    print(f"✅ Moved {checkpoint_path} to {dest_dir}")

