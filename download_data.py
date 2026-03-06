from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="MaxHeuillet/RobustGenBench",
    repo_type="dataset",
    local_dir=os.path.expanduser("~/data_processed"),
)
