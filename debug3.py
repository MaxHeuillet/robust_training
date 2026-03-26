"""
Compare the labels.csv content between clean and adversarial archives
to check if row 0 has the same label in both.
Also check if the label sequences match at all.
"""
import csv, io, os, tarfile
from pathlib import Path
import zstandard as zstd

def read_labels_csv(archive_path):
    with open(archive_path, "rb") as f:
        buf = io.BytesIO(zstd.ZstdDecompressor().stream_reader(f).read())
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        for candidate in ["test/labels.csv", "labels.csv"]:
            try:
                f = tar.extractfile(tar.getmember(candidate))
                return list(csv.DictReader(io.TextIOWrapper(f)))
            except KeyError:
                continue
    return []

CLEAN_ROOT = Path("/tmp/robustgenbench/data_processed")
ADV_ROOT   = Path(os.path.expanduser(
    "~/data/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps8_autoattack_standard"))

clean_rows = read_labels_csv(CLEAN_ROOT / "caltech101_processed.tar.zst")
adv_arch   = sorted(ADV_ROOT.glob("caltech101*_processed.tar.zst"))[0]
adv_rows   = read_labels_csv(adv_arch)

print("=== First 5 rows ===")
print(f"{'idx':<6} {'clean_file':<16} {'clean_lbl':<12} {'adv_file':<16} {'adv_lbl':<10} {'match'}")
for i in range(5):
    cf = clean_rows[i]["filename"]
    cl = clean_rows[i]["label"]
    af = adv_rows[i]["filename"]
    al = adv_rows[i]["label"]
    match = "✓" if cf == af and cl == al else "✗ MISMATCH"
    print(f"{i:<6} {cf:<16} {cl:<12} {af:<16} {al:<10} {match}")

# Check how many labels match positionally
matches = sum(
    1 for c, a in zip(clean_rows, adv_rows)
    if c["filename"] == a["filename"] and c["label"] == a["label"]
)
print(f"\nPositional matches: {matches}/{min(len(clean_rows), len(adv_rows))}")

# Check if adv labels exist anywhere in clean
clean_labels = {r["label"] for r in clean_rows}
adv_set = [(r["filename"], r["label"]) for r in adv_rows[:5]]
print(f"\nClean label set size: {len(clean_labels)}")
print(f"Adv first 5 (file, label): {adv_set}")

# Check if adv filenames are a subset of clean filenames
clean_files = {r["filename"] for r in clean_rows}
adv_files   = {r["filename"] for r in adv_rows}
print(f"\nClean filenames: {sorted(clean_files)[:5]}")
print(f"Adv   filenames: {sorted(adv_files)[:5]}")
print(f"Overlap: {len(clean_files & adv_files)}/{len(adv_files)}")