import unittest
import torch

from omegaconf import OmegaConf
import numpy as np

import torch
import sys 
import os
import traceback  # ✅ Helps capture detailed errors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from architectures import load_architecture

SCIENTIFIC_BACKBONES=(
  'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
  'CLIP-convnext_base_w-laion2B-s13B-b82K',
  'deit_small_patch16_224.fb_in1k',
  'robust_resnet50',
  'vit_small_patch16_224.augreg_in21k',
  'convnext_base.fb_in1k',
  'resnet50.a1_in1k',
  'robust_vit_base_patch16_224',
  'vit_base_patch16_224.mae',
  'convnext_base.fb_in22k',
  'robust_convnext_base',
  'vit_base_patch16_224.augreg_in1k',
  'vit_base_patch16_224.augreg_in21k',
  'vit_base_patch16_224.dino',
  'vit_base_patch16_clip_224.laion2b',
  'convnext_tiny.fb_in1k',
  'robust_convnext_tiny',
  'robust_deit_small_patch16_224',
  'vit_small_patch16_224.augreg_in1k',
  'convnext_tiny.fb_in22k',
) 

PERFORMANCE_BACKBONES=(
  'vit_base_patch16_clip_224.laion2b_ft_in1k',
  'vit_base_patch16_224.augreg_in21k_ft_in1k',
  'vit_small_patch16_224.augreg_in21k_ft_in1k',
  'eva02_base_patch14_224.mim_in22k',
  'eva02_tiny_patch14_224.mim_in22k',
  'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
  'swin_tiny_patch4_window7_224.ms_in1k',
  'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',
  'convnext_base.fb_in22k_ft_in1k',
  'convnext_tiny.fb_in22k_ft_in1k',
)

COMMON_MODELS = (
    'regnetx_004.pycls_in1k', 
    'efficientnet-b0',
    'deit_tiny_patch16_224.fb_in1k',
    'mobilevit-small',
    'mobilenetv3_large_100.ra_in1k',
    'edgenext_small.usi_in1k',
)

# Combine all sets
ALL_BACKBONES = {
    "CLIP_MODELS": SCIENTIFIC_BACKBONES,
    "FINETUNED_MODELS": PERFORMANCE_BACKBONES,
    "COMMON_MODELS": COMMON_MODELS,
}

class TestModelForwardPass(unittest.TestCase):
    """ Unit test for model forward pass with different backbone sets """

    def setUp(self):
        """ Set up test configuration """
        self.N = 10  # Number of classification classes
        self.batch_size = 2  # Two image tensors
        self.config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")
        # self.config.statedicts_path = '/home/mheuillet/Desktop/state_dicts_share'

    def test_forward_pass(self):
        """ Test forward pass for all backbone sets """
        for category, backbones in ALL_BACKBONES.items():
            print(f"\n🔹 Testing {category} ({len(backbones)} models) 🔹")
            for backbone in backbones:
                with self.subTest(backbone=backbone):
                    print(f"\n🔎 Testing backbone: {backbone}")

                    try:
                        # Load the model
                        self.config.backbone = backbone
                        model = load_architecture(self.config, self.N)
                        self.assertIsNotNone(model, f"❌ Model failed to load! Backbone: {backbone}")

                        # Move model to GPU if available
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)

                        # Create random input tensors simulating images (batch_size=2, 224x224)
                        dummy_input = torch.randn(self.batch_size, 3, 224, 224).to(device)

                        # ✅ Forward Pass
                        logits = model(dummy_input)  # Get logits
                        print(logits)

                        # Validate output shape (should be [batch_size, N])
                        expected_shape = (self.batch_size, self.N)

                        if not hasattr(logits, "shape"):
                            raise TypeError(f"❌ Model output is not a tensor! Backbone: {backbone}, Output Type: {type(logits)}")

                        self.assertEqual(
                            logits.shape, expected_shape,
                            f"❌ Output shape mismatch! Backbone: {backbone} - Got {logits.shape}, expected {expected_shape}"
                        )

                        print(f"✅ Forward pass successful! Backbone: {backbone}, Output shape: {logits.shape}")

                    except Exception as e:
                        print(f"❌ Exception in backbone {backbone}:\n{traceback.format_exc()}")  # ✅ Print full traceback
                        self.fail(f"🚨 Test failed for backbone {backbone} with error: {str(e)}")

                    print()


# Run the unit test
if __name__ == "__main__":
    unittest.main()