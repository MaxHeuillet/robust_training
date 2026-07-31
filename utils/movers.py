import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path  # In case config.dataset_path is a string



import os
import subprocess
import tarfile
import zstandard as zstd
from pathlib import Path
from datetime import datetime

def move_dataset_to_tmpdir(config):
    dataset_name = config.dataset
    data_path = Path(os.path.expandvars(config.datasets_path)).expanduser().resolve()
    archive_path = data_path / f"{dataset_name}_processed.tar.zst"

    tmpdir = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
    # Keyed by SLURM_JOB_ID (always unique per job, unlike work_path/SLURM_TMPDIR
    # which can resolve to a path shared by concurrently-running jobs on the same
    # node) so concurrent jobs extracting the same dataset don't race on the
    # same directory.
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    dest_dir = tmpdir / "data" / job_id / dataset_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"📦 Extracting {archive_path} into {dest_dir} using Python zstd + tarfile...")

    try:
        with open(archive_path, "rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=dest_dir)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to extract archive: {archive_path}") from e

    print(f"✅ Extraction of {dataset_name} completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def move_architecture_to_tmpdir(config):

    backbone = config.backbone
    statedict_dir = Path(config.statedicts_path).expanduser().resolve()

    tmpdir = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
    # Same job-scoping as move_dataset_to_tmpdir: concurrent jobs training the
    # same backbone (e.g. multiple seeds) must not stage into the same path.
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    tmpdir = tmpdir / job_id
    os.makedirs(tmpdir, exist_ok=True)

    if config.backbone in ["efficientnet-b0", "mobilevit-small"]:
        checkpoint_path = statedict_dir / f"{backbone}"
        dest_dir = tmpdir / f"{backbone}"
        shutil.copytree(str(checkpoint_path), str(dest_dir), dirs_exist_ok=True)

        dest_dir = tmpdir
        checkpoint_path = statedict_dir / f"{backbone}.pt"
        shutil.copy2(str(checkpoint_path), str(dest_dir) )
    else:
        dest_dir = tmpdir
        checkpoint_path = statedict_dir / f"{backbone}.pt"
        shutil.copy2(str(checkpoint_path), str(dest_dir) )

    print(f"✅ Moved {checkpoint_path} to {dest_dir}")

