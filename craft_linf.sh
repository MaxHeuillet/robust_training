#!/bin/bash
# craft_linf_eps8.sh

set -e

BATCH_SIZE=${1:-32}

echo "Crafting Linf eps=8 perturbations (batch_size=$BATCH_SIZE)"
echo ""

# CLIP surrogate
python craft_adversarial.py --surrogate clip --norm Linf --eps 8 --batch_size $BATCH_SIZE --upload_hf

# SigLIP2 surrogate
python craft_adversarial.py --surrogate siglip2 --norm Linf --eps 8 --batch_size $BATCH_SIZE --upload_hf

echo ""
echo "Done!"