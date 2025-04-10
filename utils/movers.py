import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path  # In case config.dataset_path is a string



def move_dataset_to_tmpdir(config):

    dataset_name = config.dataset
    data_path = Path(config.datasets_path).expanduser().resolve()
    archive_path = data_path / f"{dataset_name}_processed.tar.zst"
    
    tmpdir = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
    dest_dir = os.path.join(tmpdir, "data", dataset_name)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"📦 Extracting {archive_path} into {dest_dir} using tar + zstd...")
    
    try:
        subprocess.run(["tar", "-I", "zstd", "-xf", archive_path, "-C", dest_dir], check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"❌ Failed to extract archive: {archive_path}")

    print(f"✅ Extraction of {dataset_name} completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

def move_architecture_to_tmpdir(config):

    backbone = config.backbone
    statedict_dir = Path(config.statedicts_path).expanduser().resolve()

    tmpdir = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
    dest_dir = tmpdir

    if config.backbone in ["efficientnet-b0", "mobilevit-small"]:
        checkpoint_path = statedict_dir / f"{backbone}"
        shutil.copytree(str(checkpoint_path), str(dest_dir), dirs_exist_ok=True)
    else:
        checkpoint_path = statedict_dir / f"{backbone}.pt"
        shutil.copy2(str(checkpoint_path), str(dest_dir) )

    print(f"✅ Moved {checkpoint_path} to {dest_dir}")

