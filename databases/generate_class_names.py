"""
Generate class_names.json for each dataset.
============================================
This script loads each torchvision dataset (the same way loaders.py does)
and extracts the integer→class-name mapping, saving it as class_names.json.

You must run this on a machine that has the original raw datasets downloaded
(the same datasets_path used by your preprocessing pipeline).

Usage:
    python generate_class_names.py --datasets_path ~/data
    python generate_class_names.py --datasets_path ~/data --output_dir ./class_names

It will produce one JSON file per dataset:
    class_names/caltech101.json
    class_names/fgvc-aircraft-2013b.json
    class_names/flowers-102.json
    class_names/oxford-iiit-pet.json
    class_names/stanford_cars.json
    class_names/uc-merced-land-use-dataset.json

Each JSON file is a dict: {"0": "class_name_0", "1": "class_name_1", ...}
"""

import argparse
import json
import os
from pathlib import Path

from torchvision import datasets


def get_class_names_caltech101(datadir: str) -> dict:
    """Caltech101: sorted subfolder names minus BACKGROUND_Google."""
    ds = datasets.Caltech101(root=datadir, download=False)
    # ds.categories is the sorted list of class names
    return {str(i): name for i, name in enumerate(ds.categories)}


def get_class_names_fgvc_aircraft(datadir: str) -> dict:
    """FGVCAircraft: uses dataset.classes."""
    ds = datasets.FGVCAircraft(root=datadir, split="train", download=False)
    return {str(i): name for i, name in enumerate(ds.classes)}


def get_class_names_flowers102(datadir: str) -> dict:
    """
    Flowers102: torchvision does not expose human-readable class names
    in dataset.classes. The labels are 0-101 corresponding to the
    102 flower categories. We use the canonical ordering from the
    Oxford dataset's cat_to_name.json / setid mapping.
    
    The canonical names are well-documented. We hardcode them here
    in the standard order used by torchvision (0-indexed, matching
    the original 1-indexed labels shifted by -1).
    """
    # These are the 102 flower names in the order that torchvision
    # assigns labels 0-101 (matching the original dataset's label
    # ordering from 1-102, shifted to 0-101).
    names = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
        "sweet pea", "english marigold", "tiger lily", "moon orchid",
        "bird of paradise", "monkshood", "globe thistle",
        "snapdragon", "colt's foot", "king protea", "spear thistle",
        "yellow iris", "globe-flower", "purple coneflower",
        "peruvian lily", "balloon flower", "giant white arum lily",
        "fire lily", "pincushion flower", "fritillary",
        "red ginger", "grape hyacinth", "corn poppy",
        "prince of wales feathers", "stemless gentian", "artichoke",
        "sweet william", "carnation", "garden phlox",
        "love in the mist", "mexican aster", "alpine sea holly",
        "ruby-lipped cattleya", "cape flower", "great masterwort",
        "siam tulip", "lenten rose", "barbeton daisy",
        "daffodil", "sword lily", "poinsettia",
        "bolero deep blue", "wallflower", "marigold",
        "buttercup", "oxeye daisy", "common dandelion",
        "petunia", "wild pansy", "primula",
        "sunflower", "pelargonium", "bishop of llandaff",
        "gaura", "geranium", "orange dahlia",
        "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
        "black-eyed susan", "silverbush", "californian poppy",
        "osteospermum", "spring crocus", "bearded iris",
        "windflower", "tree poppy", "gazania",
        "azalea", "water lily", "rose",
        "thorn apple", "morning glory", "passion flower",
        "lotus", "toad lily", "anthurium",
        "frangipani", "clematis", "hibiscus",
        "columbine", "desert-rose", "tree mallow",
        "magnolia", "cyclamen", "watercress",
        "canna lily", "hippeastrum", "bee balm",
        "ball moss", "foxglove", "bougainvillea",
        "camellia", "mallow", "mexican petunia",
        "bromelia", "blanket flower", "trumpet creeper",
        "blackberry lily",
    ]
    assert len(names) == 102, f"Expected 102 names, got {len(names)}"
    return {str(i): name for i, name in enumerate(names)}


def get_class_names_oxford_pet(datadir: str) -> dict:
    """OxfordIIITPet: uses dataset.classes."""
    ds = datasets.OxfordIIITPet(root=datadir, split="trainval", download=False)
    return {str(i): name for i, name in enumerate(ds.classes)}


def get_class_names_stanford_cars(datadir: str) -> dict:
    """StanfordCars: uses dataset.classes."""
    ds = datasets.StanfordCars(root=datadir, split="train", download=False)
    return {str(i): name for i, name in enumerate(ds.classes)}


def get_class_names_uc_merced(datadir: str) -> dict:
    """UC Merced: ImageFolder with sorted subfolders."""
    path = os.path.join(datadir, "UCMerced_LandUse", "Images")
    ds = datasets.ImageFolder(root=path)
    # ds.classes is the sorted list of subfolder names
    return {str(i): name for i, name in enumerate(ds.classes)}


# Registry
DATASET_FNS = {
    "caltech101": get_class_names_caltech101,
    "fgvc-aircraft-2013b": get_class_names_fgvc_aircraft,
    "flowers-102": get_class_names_flowers102,
    "oxford-iiit-pet": get_class_names_oxford_pet,
    "stanford_cars": get_class_names_stanford_cars,
    "uc-merced-land-use-dataset": get_class_names_uc_merced,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate class_names.json for each dataset"
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default=str(Path.home() / "data"),
        help="Path to the directory containing raw torchvision datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./class_names",
        help="Directory to save the class_names JSON files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_FNS.keys()),
        choices=list(DATASET_FNS.keys()),
        help="Which datasets to process (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        print(f"Processing: {ds_name}")
        try:
            fn = DATASET_FNS[ds_name]
            mapping = fn(args.datasets_path)
            out_path = output_dir / f"{ds_name}.json"
            with open(out_path, "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"  ✅ Saved {len(mapping)} classes → {out_path}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print(f"     Make sure the raw dataset is available at {args.datasets_path}")


if __name__ == "__main__":
    main()