"""
Load the SAME image (00000.png) from both clean and adversarial archives
and print pixel values side by side to understand the Δmax=247 mystery.
"""
import csv, io, os, tarfile
from pathlib import Path
import numpy as np
from PIL import Image
import zstandard as zstd

CLEAN_ROOT = Path("/tmp/robustgenbench/data_processed")
ADV_ROOT   = Path(os.path.expanduser(
    "~/data/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard"))

def load_specific_image(archive_path, filename):
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        raw = tar.extractfile(f"test/{filename}").read()
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

clean_arch = CLEAN_ROOT / "caltech101_processed.tar.zst"
adv_arch   = sorted(ADV_ROOT.glob("caltech101*_processed.tar.zst"))[0]

print("Loading 00000.png from both archives...")
clean_img = load_specific_image(clean_arch, "00000.png")
adv_img   = load_specific_image(adv_arch,   "00000.png")

print(f"\nClean shape : {clean_img.shape}")
print(f"Adv   shape : {adv_img.shape}")
print(f"\nClean pixel [0,0] : {clean_img[0,0,:]}")
print(f"Adv   pixel [0,0] : {adv_img[0,0,:]}")
print(f"Diff  pixel [0,0] : {np.abs(clean_img[0,0,:].astype(int) - adv_img[0,0,:].astype(int))}")

diff = np.abs(clean_img.astype(np.float32) - adv_img.astype(np.float32))
print(f"\nMax diff  : {diff.max():.1f}")
print(f"Mean diff : {diff.mean():.4f}")
print(f"% pixels with diff > 0 : {(diff.sum(-1) > 0).mean()*100:.1f}%")
print(f"% pixels with diff > 2 : {(diff.max(-1) > 2).mean()*100:.1f}%")
print(f"% pixels with diff > 7 : {(diff.max(-1) > 7).mean()*100:.1f}%")

print(f"\nSample of diff values (first 5x5 pixels, max across channels):")
print(diff[:5,:5,:].max(axis=-1).astype(int))

# Also check: are they the same image visually?
print(f"\nAre arrays identical? {np.array_equal(clean_img, adv_img)}")
print(f"Clean mean: {clean_img.mean():.2f}, Adv mean: {adv_img.mean():.2f}")