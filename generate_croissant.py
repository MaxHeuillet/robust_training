"""
Generate a NeurIPS 2026 Datasets & Benchmarks compliant Croissant metadata
file for the RobustGenBench dataset.

Conforms to:
  - http://mlcommons.org/croissant/1.1   (core spec, version required by NeurIPS 2026)
  - http://mlcommons.org/croissant/RAI/1.0   (Responsible AI extension)

Includes the minimal Responsible AI metadata required by the NeurIPS 2026
Evaluations & Datasets track:
  rai:dataLimitations, rai:dataBiases, rai:personalSensitiveInformation,
  rai:dataUseCases, rai:dataSocialImpact, rai:hasSyntheticData,
  prov:wasDerivedFrom, prov:wasGeneratedBy.
"""

import json
from collections import OrderedDict

REPO = "legolasflagstaff/RobustGenBench"
HF_BASE = f"https://huggingface.co/datasets/{REPO}"
RAW = f"{HF_BASE}/resolve/main"

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

# Source datasets we re-processed, for prov:wasDerivedFrom.
SOURCE_DATASET_URIS = [
    "https://data.caltech.edu/records/mzrjq-6wc02",                # Caltech-101
    "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/",        # FGVC-Aircraft
    "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",          # Oxford 102 Flowers
    "https://www.robots.ox.ac.uk/~vgg/data/pets/",                 # Oxford-IIIT Pet
    "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",      # Stanford Cars
    "http://weegee.vision.ucmerced.edu/datasets/landuse.html",     # UC Merced Land Use
]

ADVERSARIAL_MODELS = [
    "zeroshot_clip_vitb16_laion2b",
    "zeroshot_clip_vith14_laion2b",
    "zeroshot_metaclip_vith14_fullcc2_5b",
    "zeroshot_siglip2_base_patch16_224",
    "zeroshot_siglip2_so400m_patch14_384",
    "zeroshot_siglip2_so400m_patch16_naflex",
    "zeroshot_siglip2_so400m_patch16_naflex_patchify",
]

CONTEXT = OrderedDict([
    ("@language", "en"),
    ("@vocab", "https://schema.org/"),
    ("citeAs", "cr:citeAs"),
    ("column", "cr:column"),
    ("conformsTo", "dct:conformsTo"),
    ("containedIn", "cr:containedIn"),
    ("cr", "http://mlcommons.org/croissant/"),
    ("rai", "http://mlcommons.org/croissant/RAI/"),
    ("prov", "http://www.w3.org/ns/prov#"),
    ("data", {"@id": "cr:data", "@type": "@json"}),
    ("dataType", {"@id": "cr:dataType", "@type": "@vocab"}),
    ("dct", "http://purl.org/dc/terms/"),
    ("equivalentProperty", "cr:equivalentProperty"),
    ("examples", {"@id": "cr:examples", "@type": "@json"}),
    ("extract", "cr:extract"),
    ("field", "cr:field"),
    ("fileProperty", "cr:fileProperty"),
    ("fileObject", "cr:fileObject"),
    ("fileSet", "cr:fileSet"),
    ("format", "cr:format"),
    ("includes", "cr:includes"),
    ("isLiveDataset", "cr:isLiveDataset"),
    ("jsonPath", "cr:jsonPath"),
    ("key", "cr:key"),
    ("md5", "cr:md5"),
    ("parentField", "cr:parentField"),
    ("path", "cr:path"),
    ("recordSet", "cr:recordSet"),
    ("references", "cr:references"),
    ("regex", "cr:regex"),
    ("repeated", "cr:repeated"),
    ("replace", "cr:replace"),
    ("samplingRate", "cr:samplingRate"),
    ("sc", "https://schema.org/"),
    ("separator", "cr:separator"),
    ("source", "cr:source"),
    ("subField", "cr:subField"),
    ("transform", "cr:transform"),
    ("wasDerivedFrom", "prov:wasDerivedFrom"),
    ("wasGeneratedBy", "prov:wasGeneratedBy"),
])


def file_object(at_id, name, content_url, encoding_format, description=None, sha256=None):
    obj = OrderedDict([
        ("@type", "cr:FileObject"),
        ("@id", at_id),
        ("name", name),
    ])
    if description:
        obj["description"] = description
    obj["contentUrl"] = content_url
    obj["encodingFormat"] = encoding_format
    obj["sha256"] = sha256 or "unknown-please-fill-in"
    return obj


def file_set(at_id, name, description, includes, encoding_format="application/zstd"):
    return OrderedDict([
        ("@type", "cr:FileSet"),
        ("@id", at_id),
        ("name", name),
        ("description", description),
        ("containedIn", {"@id": "repository-root"}),
        ("encodingFormat", encoding_format),
        ("includes", includes),
    ])


def build():
    distribution = []

    # 0) FileObject for the repository root, used as containedIn for FileSets.
    distribution.append(OrderedDict([
        ("@type", "cr:FileObject"),
        ("@id", "repository-root"),
        ("name", "RobustGenBench Hugging Face dataset repository"),
        ("description",
         "The full dataset repository on the Hugging Face Hub, served over HTTPS. "
         "All FileSets in this Croissant manifest are matched against the file "
         "tree rooted here."),
        ("contentUrl", HF_BASE),
        ("encodingFormat", "git+https"),
        ("sha256", "unknown-please-fill-in"),
    ]))

    for ds in DATASETS:
        fname = f"{ds}_processed.tar.zst"
        distribution.append(file_object(
            at_id=f"clean/{ds}",
            name=fname,
            content_url=f"{RAW}/{fname}",
            encoding_format="application/zstd",
            description=(
                f"Clean (un-perturbed) processed version of the {ds} dataset, "
                f"packaged as a zstd-compressed tar with the layout: "
                f"metadata.json + train|val|test|test_common/{{labels.csv, NNNNN.png}}."
            ),
        ))

    for ds in DATASETS:
        distribution.append(file_object(
            at_id=f"class_names/{ds}",
            name=f"class_names/{ds}.json",
            content_url=f"{RAW}/class_names/{ds}.json",
            encoding_format="application/json",
            description=f"Mapping from integer class id to class label for {ds}.",
        ))

    distribution.append(file_set(
        at_id="adversarial-common",
        name="adversarial_common_corruptions",
        description=(
            "Common corruptions (e.g., noise, blur, weather, digital) applied to each "
            "source dataset at multiple severity levels. Files follow the pattern "
            "adversarial/common/common_severity{N}/{dataset}__common_severity{N}_processed.tar.zst."
        ),
        includes="adversarial/common/**/*_processed.tar.zst",
    ))

    distribution.append(file_set(
        at_id="adversarial-random",
        name="adversarial_random_perturbations",
        description=(
            "Random uniform perturbations applied within an L-infinity ball "
            "(e.g. epsilon=30/255). Files follow the pattern "
            "adversarial/random/{constraint}/{dataset}__random__{constraint}_processed.tar.zst."
        ),
        includes="adversarial/random/**/*_processed.tar.zst",
    ))

    for model in ADVERSARIAL_MODELS:
        distribution.append(file_set(
            at_id=f"adversarial-{model}",
            name=f"adversarial_{model}",
            description=(
                f"Adversarial examples crafted against the {model} model under "
                f"various threat models (Linf/L1/L2 budgets, AutoAttack standard). "
                f"Files follow the pattern "
                f"adversarial/{model}/{{constraint}}/{{dataset}}__{model}__{{constraint}}_processed.tar.zst."
            ),
            includes=f"adversarial/{model}/**/*_processed.tar.zst",
        ))

    class_names_recordset = OrderedDict([
        ("@type", "cr:RecordSet"),
        ("@id", "class-names"),
        ("name", "class_names"),
        ("description",
         "Per-dataset class-id -> class-label mapping. Each class_names/<dataset>.json "
         "file is a flat JSON object whose keys are the integer class ids "
         "(string-encoded as required by JSON) and whose values are the human-readable "
         "class names. The position of each value in this RecordSet matches its key in "
         "the underlying JSON object."),
        ("field", [
            OrderedDict([
                ("@type", "cr:Field"),
                ("@id", "class-names/class_label"),
                ("name", "class_label"),
                ("description", "Human-readable class label."),
                ("dataType", "sc:Text"),
                ("source", OrderedDict([
                    ("fileObject", {"@id": "class_names/uc-merced-land-use-dataset"}),
                    ("extract", {"jsonPath": "$.*"}),
                ])),
            ]),
        ]),
    ])

    # ------------------------------------------------------------------
    # Image RecordSet — built from the CSV manifest + PNG files.
    # ------------------------------------------------------------------
    # Each archive in `distribution` (clean or adversarial) follows an
    # identical internal layout, verified against the UC Merced clean
    # shard:
    #   metadata.json
    #   {split}/labels.csv          # columns: filename, label
    #   {split}/{NNNNN}.png         # 5-digit zero-padded global index
    # where {split} is one of: train, val, test, test_common.
    #
    # We anchor the RecordSet schema to ONE representative archive (the
    # UC Merced clean shard), define explicit FileSets for the per-split
    # CSVs and PNGs INSIDE that archive (using `containedIn`), and
    # document in the description that every other archive in the
    # distribution shares the same schema.

    # FileSets contained inside the canonical archive.
    canonical_archive_id = "clean/uc-merced-land-use-dataset"

    csv_fileset_ids = []
    png_fileset_ids = []
    extra_distribution = []
    for split in ("train", "val", "test", "test_common"):
        csv_id = f"canonical-csv-{split}"
        png_id = f"canonical-png-{split}"
        csv_fileset_ids.append(csv_id)
        png_fileset_ids.append(png_id)
        extra_distribution.append(OrderedDict([
            ("@type", "cr:FileSet"),
            ("@id", csv_id),
            ("name", f"{split}_labels_csv_uc_merced"),
            ("description",
             f"Per-image label manifest for the '{split}' split, inside the "
             f"canonical UC Merced clean archive. Same layout exists in every "
             f"clean and adversarial archive in this distribution."),
            ("containedIn", {"@id": canonical_archive_id}),
            ("encodingFormat", "text/csv"),
            ("includes", f"{split}/labels.csv"),
        ]))
        extra_distribution.append(OrderedDict([
            ("@type", "cr:FileSet"),
            ("@id", png_id),
            ("name", f"{split}_images_uc_merced"),
            ("description",
             f"PNG images for the '{split}' split inside the canonical UC "
             f"Merced clean archive."),
            ("containedIn", {"@id": canonical_archive_id}),
            ("encodingFormat", "image/png"),
            ("includes", f"{split}/*.png"),
        ]))

    # Build the RecordSets following the PASS-dataset pattern from the
    # Croissant 1.1 spec: one RecordSet per split for the CSV manifest
    # (filename, label), and one RecordSet per split for the images, where
    # the image RecordSet has a filename field whose `references` points to
    # the labels RecordSet's filename field — this is the formal join
    # declaration the validator requires.
    label_recordsets = []
    image_recordsets = []
    for split, csv_id, png_id in zip(
        ("train", "val", "test", "test_common"),
        csv_fileset_ids,
        png_fileset_ids,
    ):
        labels_id = f"labels-{split}"
        images_id = f"images-{split}"

        # Labels RecordSet (rows from the CSV).
        label_recordsets.append(OrderedDict([
            ("@type", "cr:RecordSet"),
            ("@id", labels_id),
            ("name", f"labels_{split}"),
            ("description",
             f"One row per image in the '{split}' split: filename and "
             f"integer class id, extracted from {split}/labels.csv inside "
             f"the canonical UC Merced clean archive. The same schema applies "
             f"to every other archive in this dataset's distribution."),
            ("key", {"@id": f"{labels_id}/filename"}),
            ("field", [
                OrderedDict([
                    ("@type", "cr:Field"),
                    ("@id", f"{labels_id}/filename"),
                    ("name", "filename"),
                    ("description",
                     "PNG filename within the split directory, e.g. '00042.png'."),
                    ("dataType", "sc:Text"),
                    ("source", OrderedDict([
                        ("fileSet", {"@id": csv_id}),
                        ("extract", {"column": "filename"}),
                    ])),
                ]),
                OrderedDict([
                    ("@type", "cr:Field"),
                    ("@id", f"{labels_id}/label"),
                    ("name", "label"),
                    ("description",
                     "Integer class id. Matches the keys in the corresponding "
                     "class_names/{dataset}.json mapping."),
                    ("dataType", "sc:Integer"),
                    ("source", OrderedDict([
                        ("fileSet", {"@id": csv_id}),
                        ("extract", {"column": "label"}),
                    ])),
                ]),
            ]),
        ]))

        # Images RecordSet — joins each PNG to a CSV row by filename.
        image_recordsets.append(OrderedDict([
            ("@type", "cr:RecordSet"),
            ("@id", images_id),
            ("name", f"images_{split}"),
            ("description",
             f"One record per PNG in the '{split}' split. Each record's "
             f"filename joins to the corresponding row in the {labels_id} "
             f"RecordSet via cr:references."),
            ("key", {"@id": f"{images_id}/filename"}),
            ("field", [
                OrderedDict([
                    ("@type", "cr:Field"),
                    ("@id", f"{images_id}/filename"),
                    ("name", "filename"),
                    ("description",
                     "PNG filename, used as the join key against "
                     f"{labels_id}/filename."),
                    ("dataType", "sc:Text"),
                    ("source", OrderedDict([
                        ("fileSet", {"@id": png_id}),
                        ("extract", {"fileProperty": "filename"}),
                    ])),
                    ("references", OrderedDict([
                        ("field", {"@id": f"{labels_id}/filename"}),
                    ])),
                ]),
                OrderedDict([
                    ("@type", "cr:Field"),
                    ("@id", f"{images_id}/image"),
                    ("name", "image"),
                    ("description", "PNG image content."),
                    ("dataType", "sc:ImageObject"),
                    ("source", OrderedDict([
                        ("fileSet", {"@id": png_id}),
                        ("extract", {"fileProperty": "content"}),
                    ])),
                ]),
            ]),
        ]))

    dataset = OrderedDict()
    dataset["@context"] = CONTEXT
    dataset["@type"] = "sc:Dataset"
    dataset["name"] = "RobustGenBench"
    dataset["description"] = (
        "RobustGenBench is a benchmark for evaluating the adversarial and "
        "out-of-distribution robustness of zero-shot vision-language classifiers. "
        "It packages six widely-used image-classification datasets (Caltech-101, "
        "FGVC-Aircraft, Oxford 102 Flowers, Oxford-IIIT Pet, Stanford Cars, and "
        "the UC Merced Land-Use Dataset) together with three families of "
        "perturbed copies: (1) common corruptions at multiple severity levels, "
        "(2) random uniform perturbations within fixed L-infinity budgets, and "
        "(3) white-box adversarial examples crafted against several CLIP, "
        "MetaCLIP, and SigLIP-2 zero-shot classifiers using AutoAttack and "
        "related procedures. Each archive is a zstd-compressed tar with a "
        "fixed internal layout: a top-level metadata.json (split sizes and "
        "class count), and one subdirectory per split (train, val, test, "
        "test_common) each containing a labels.csv manifest (columns: "
        "filename, label) and the corresponding zero-padded NNNNN.png images. "
        "Per-dataset class-id to label mappings are distributed alongside "
        "the data in the class_names/ directory.\n\n"
        "LICENSE NOTICE. The CC BY-NC 4.0 license declared on this dataset "
        "applies to the processed compilation (the perturbed shards, the "
        "common-corruption pipeline output, and the random/adversarial "
        "variants generated by the authors of this benchmark). It does NOT "
        "override the terms of the six upstream source datasets. Individual "
        "images are derivative works of imagery whose copyright remains with "
        "the original authors and rights holders. In particular: FGVC-Aircraft "
        "imagery is made available by its photographers for non-commercial "
        "research purposes only, with copyright retained by the individual "
        "photographers; users wishing to use any FGVC-Aircraft-derived images "
        "outside this scope must contact the original photographers directly. "
        "Stanford Cars imagery is redistributed here under terms equivalent to "
        "those of its original release. Oxford-IIIT Pet imagery is licensed "
        "CC BY-SA 4.0 by its creators and that license continues to govern "
        "those underlying images. Caltech-101 (CC BY 4.0), Oxford 102 Flowers, "
        "and UC Merced Land-Use (derived from U.S. public-domain USGS imagery) "
        "carry their respective upstream terms. Users of RobustGenBench must "
        "comply with both this dataset's CC BY-NC 4.0 terms AND the terms of "
        "the relevant upstream source dataset(s). When in doubt, the more "
        "restrictive of the two applies."
    )
    dataset["conformsTo"] = [
        "http://mlcommons.org/croissant/1.1",
        "http://mlcommons.org/croissant/RAI/1.0",
    ]
    dataset["url"] = HF_BASE
    dataset["version"] = "1.0.0"
    dataset["datePublished"] = "2026-04-28"
    dataset["keywords"] = [
        "adversarial robustness",
        "common corruptions",
        "image classification",
        "zero-shot classification",
        "CLIP",
        "SigLIP",
        "MetaCLIP",
        "benchmark",
    ]
    dataset["creator"] = OrderedDict([
        ("@type", "sc:Person"),
        ("name", "Maxime Heuillet"),
        ("url", "https://huggingface.co/legolasflagstaff"),
    ])
    dataset["publisher"] = OrderedDict([
        ("@type", "sc:Organization"),
        ("name", "legolasflagstaff (Hugging Face Hub)"),
        ("url", "https://huggingface.co/legolasflagstaff"),
    ])
    dataset["license"] = "https://spdx.org/licenses/CC-BY-NC-4.0.html"
    dataset["citeAs"] = (
        "@misc{robustgenbench2026,\n"
        "  author = {Maxime Heuillet},\n"
        "  title  = {RobustGenBench: a benchmark for adversarial and "
        "out-of-distribution robustness of zero-shot vision-language classifiers},\n"
        "  year   = {2026},\n"
        "  note   = {TODO: replace with the BibTeX entry for the associated paper.},\n"
        "  url    = {" + HF_BASE + "}\n"
        "}"
    )
    dataset["isLiveDataset"] = False

    # ---- Required RAI metadata ----
    dataset["rai:dataLimitations"] = [
        "RobustGenBench inherits the distributional limits of its six source "
        "datasets, which were each curated for narrow visual domains (everyday "
        "objects, aircraft, flowers, pets, cars, aerial land use). Robustness "
        "scores measured on this benchmark therefore generalise only to the "
        "concepts and image distributions covered by these sources and should "
        "not be interpreted as evidence of robustness on broader open-world "
        "imagery.",
        "All adversarial examples in the 'adversarial/zeroshot_*' subtrees were "
        "generated against a fixed set of vision-language models (specific CLIP, "
        "MetaCLIP and SigLIP-2 checkpoints). Robustness estimates against "
        "attacks transferred from these checkpoints are not a substitute for "
        "evaluation under adaptive attacks, and the benchmark should not be "
        "used to claim robustness against attackers with white-box access to a "
        "different model.",
        "Common-corruption shards are produced by a deterministic pipeline at a "
        "small number of severity levels and do not capture the full range of "
        "real-world distribution shift (e.g. naturalistic adversarial examples, "
        "domain shift, sensor-specific artefacts).",
        "Not recommended uses: training of production classifiers, decisions "
        "about deployment of vision systems in safety-critical settings, or any "
        "application where the upstream dataset licenses prohibit such use.",
    ]
    dataset["rai:dataBiases"] = [
        "The six source datasets are skewed toward classes and visual concepts "
        "well-represented on the Anglophone, Western web at the time of their "
        "collection (Stanford Cars: Western car models from a narrow time "
        "window; Oxford-IIIT Pet: cat and dog breeds common in the UK; "
        "FGVC-Aircraft: aircraft variants from a single photographer's archive; "
        "Caltech-101: web-scraped imagery with class-imbalance and "
        "within-class similarity issues documented in the original paper). "
        "Robustness measurements on RobustGenBench therefore inherit these "
        "selection biases.",
        "Adversarial example difficulty is unevenly distributed across "
        "downstream classes: examples crafted against zero-shot CLIP-family "
        "classifiers tend to succeed more easily on classes for which the "
        "underlying CLIP text-encoder has stronger priors, biasing the "
        "per-class robustness signal.",
    ]
    dataset["rai:personalSensitiveInformation"] = (
        "RobustGenBench contains no annotated personal or sensitive attributes. "
        "Some imagery in the upstream Caltech-101 and Stanford Cars datasets "
        "may incidentally depict people (background pedestrians, drivers, etc.) "
        "or vehicle license plates. No additional anonymisation step was "
        "performed during re-processing beyond what is present in the original "
        "datasets, so users should treat residual depictions of individuals "
        "the same way they would when using the upstream sources directly."
    )
    dataset["rai:dataUseCases"] = [
        "Intended use: evaluation (held-out testing) of the adversarial and "
        "common-corruption robustness of zero-shot vision-language classifiers, "
        "specifically CLIP-family and SigLIP-family models.",
        "Intended use: ablation studies of robustness interventions "
        "(adversarial fine-tuning, prompt engineering, ensembling) on top of "
        "pretrained zero-shot classifiers.",
        "Construct validity has been established for: relative ranking of "
        "zero-shot classifiers by L-infinity AutoAttack robust accuracy on the "
        "included class taxonomies.",
        "Construct validity has NOT been established for: training "
        "discriminative or generative models, fairness auditing across "
        "demographic groups, or robustness claims that generalise outside the "
        "six source taxonomies and the specific attack budgets included here.",
    ]
    dataset["rai:dataSocialImpact"] = (
        "Positive impact: by standardising adversarial and common-corruption "
        "evaluation across multiple zero-shot vision-language classifiers, "
        "RobustGenBench is intended to make robustness comparisons more "
        "reproducible and to surface non-robust deployment configurations "
        "before they reach users. Negative impact: the same adversarial "
        "examples that this benchmark uses to evaluate defenders could in "
        "principle be redistributed to evaluate or develop attacks; the "
        "release is therefore restricted to research use under the upstream "
        "dataset licenses, and we deliberately do not include attack code or "
        "model weights for the targeted classifiers in this repository. Risks "
        "of misuse are further mitigated by the limited domain coverage of the "
        "source datasets, which makes the benchmark unsuitable as a training "
        "corpus for general-purpose recognition systems."
    )
    dataset["rai:hasSyntheticData"] = True
    dataset["rai:hasSyntheticDataDescription"] = (
        "Adversarial and corrupted images are pixel-level perturbations of "
        "natural source images and are therefore considered synthetic data for "
        "the purposes of this field. They were generated procedurally from the "
        "clean source images using AutoAttack-standard, random uniform sampling "
        "within fixed L-p balls, and a deterministic common-corruption "
        "pipeline. No generative-model output is included."
    )
    dataset["wasDerivedFrom"] = [
        OrderedDict([("@id", uri), ("name", uri)]) for uri in SOURCE_DATASET_URIS
    ]
    dataset["wasGeneratedBy"] = (
        "Provenance / re-processing pipeline. (1) Source datasets were "
        "downloaded from their canonical hosting URLs and decoded. (2) Images "
        "were resized and re-encoded into per-split image directories using "
        "pre-processing recipe (resize to model-native input size, JPEG "
        "re-encoding, integer class-id assignment). (3) The clean shards form "
        "the 'clean/' files at the repository root. (4) Common-corruption "
        "variants were generated with a deterministic ImageNet-C-style "
        "corruption pipeline at multiple severity levels. (5) Random "
        "perturbation variants were generated by sampling uniform noise in a "
        "specified L-p ball and clipping back to valid pixel range. (6) "
        "White-box adversarial variants were generated with AutoAttack "
        "(standard) against the listed CLIP, MetaCLIP and SigLIP-2 zero-shot "
        "classifiers, under a range of L-infinity / L-1 / L-2 budgets. All "
        "generated archives are zstd-compressed before upload. No human "
        "annotation step was performed during re-processing; class labels are "
        "inherited unchanged from the source datasets. TODO(authors): replace "
        "this paragraph with a precise description and a link to the public "
        "re-processing code repository."
    )

    dataset["distribution"] = distribution + extra_distribution
    dataset["recordSet"] = (
        [class_names_recordset] + label_recordsets + image_recordsets
    )

    return dataset


if __name__ == "__main__":
    obj = build()
    out_path = "/home/claude/croissant/croissant.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")

    with open(out_path, encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["@type"] == "sc:Dataset"
    assert "http://mlcommons.org/croissant/1.1" in reloaded["conformsTo"]
    assert "http://mlcommons.org/croissant/RAI/1.0" in reloaded["conformsTo"]
    print(f"distribution items: {len(reloaded['distribution'])}")
    print(f"recordSet items   : {len(reloaded['recordSet'])}")
    rai_keys = [k for k in reloaded if k.startswith('rai:') or k in ('wasDerivedFrom', 'wasGeneratedBy')]
    print("RAI / PROV fields :", rai_keys)
