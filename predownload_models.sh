#!/bin/bash
# predownload_models.sh — Run ONCE on a LOGIN NODE (with internet).
# Downloads all model weights to ~/links/scratch so compute nodes
# can load them offline.
#
# Usage:
#   bash predownload_models.sh

set -e

SCRATCH_CACHE=~/links/scratch/robustgenbench
HF_CACHE=$SCRATCH_CACHE/hf_cache
MODEL_CACHE=$SCRATCH_CACHE/model_cache

mkdir -p "$HF_CACHE" "$MODEL_CACHE"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE"
export TORCH_HOME="$MODEL_CACHE"

echo "============================================="
echo "  Pre-downloading models to $SCRATCH_CACHE"
echo "============================================="

python3 -c "
import os, sys

# ---- HuggingFace models ----
print('\n>>> Downloading SigLIP2 SO400M NaFlex...')
from transformers import AutoModel, AutoProcessor, AutoTokenizer
for cls, mid in [
    (AutoModel, 'google/siglip2-so400m-patch16-naflex'),
    (AutoProcessor, 'google/siglip2-so400m-patch16-naflex'),
    (AutoTokenizer, 'google/siglip2-so400m-patch16-naflex'),
]:
    cls.from_pretrained(mid, cache_dir='$HF_CACHE')
print('    done.')

# ---- OpenCLIP models ----
import open_clip

print('\n>>> Downloading MetaCLIP ViT-H/14 (FullCC-2.5B)...')
open_clip.create_model_and_transforms(
    'ViT-H-14-quickgelu', pretrained='metaclip_fullcc',
    cache_dir='$HF_CACHE')
print('    done.')

print('\n>>> Downloading DFN5B CLIP ViT-H/14...')
open_clip.create_model_and_transforms(
    'ViT-H-14-quickgelu', pretrained='dfn5b',
    cache_dir='$HF_CACHE')
print('    done.')

# ---- EVA-CLIP-18B ----
print('\n>>> Downloading EVA-CLIP-18B...')
try:
    from eva_clip import create_model_and_transforms as eva_create
    eva_create('EVA-CLIP-18B', 'eva_clip', force_custom_clip=True)
    print('    done (via eva_clip).')
except ImportError:
    print('    eva_clip not installed, downloading via HuggingFace...')
    from transformers import AutoModel
    AutoModel.from_pretrained(
        'BAAI/EVA-CLIP-18B', trust_remote_code=True,
        cache_dir='$HF_CACHE')
    from transformers import CLIPTokenizer
    CLIPTokenizer.from_pretrained(
        'openai/clip-vit-large-patch14', cache_dir='$HF_CACHE')
    print('    done (via HuggingFace).')

# ---- Also predownload existing models if not cached ----
print('\n>>> Downloading CLIP ViT-B/16 (LAION-2B)...')
open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='laion2b_s34b_b88k', cache_dir='$HF_CACHE')
print('    done.')

print('\n>>> Downloading CLIP ViT-H/14 (LAION-2B)...')
open_clip.create_model_and_transforms(
    'ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir='$HF_CACHE')
print('    done.')

print('\n>>> Downloading SigLIP2 base...')
for cls, mid in [
    (AutoModel, 'google/siglip2-base-patch16-224'),
    (AutoTokenizer, 'google/siglip2-base-patch16-224'),
]:
    cls.from_pretrained(mid, cache_dir='$HF_CACHE')
print('    done.')

print('\n>>> Downloading SigLIP2 SO400M (384)...')
for cls, mid in [
    (AutoModel, 'google/siglip2-so400m-patch14-384'),
    (AutoTokenizer, 'google/siglip2-so400m-patch14-384'),
]:
    cls.from_pretrained(mid, cache_dir='$HF_CACHE')
print('    done.')

print('\n>>> Downloading DINOv2 ViT-L/14 (torch.hub)...')
import torch
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
print('    done.')

# ---- Dataset ----
print('\n>>> Downloading RobustGenBench dataset...')
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='MaxHeuillet/RobustGenBench', repo_type='dataset',
    local_dir='$SCRATCH_CACHE/data_processed',
    cache_dir='$HF_CACHE',
    ignore_patterns='adversarial/*',
)
print('    done.')

print('\n============================================')
print('All downloads complete!')
print('You can now run jobs on compute nodes.')
print('============================================')
"