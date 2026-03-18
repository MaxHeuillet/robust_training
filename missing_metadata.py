import csv, json
from pathlib import Path

# Paths
split_dir    = Path("/tmp/llm_classify/flowers-102/test")
adv_dir      = Path("/Users/maximeheuillet/Desktop/robust_training/adversarial_examples/flowers-102__zeroshot_clip_vitb16_laion2b__linf_eps30__apgd-ce")
cls_path     = Path("/Users/maximeheuillet/data_processed/class_names/flowers-102.json")

# Load class names
raw = json.loads(cls_path.read_text())
label_to_name = {int(k): v for k, v in raw.items()}

# Load labels.csv in order
items = []
with open(split_dir / "labels.csv") as f:
    for row in csv.DictReader(f):
        items.append((row["filename"], int(row["label"])))

# Build metadata only for images that exist
records = []
for idx, (filename, label) in enumerate(items):
    png_name = f"{idx:05d}.png"
    if (adv_dir / png_name).exists():
        records.append({
            "image_path": png_name,
            "label_idx":  label,
            "label_name": label_to_name[label],
        })

# Save
with open(adv_dir / "metadata.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"Written {len(records)} records to metadata.jsonl")