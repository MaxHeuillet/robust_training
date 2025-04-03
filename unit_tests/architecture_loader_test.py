import unittest
import torch

from omegaconf import OmegaConf
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torch
import sys 
import os
import traceback  # ✅ Helps capture detailed errors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from architectures import load_architecture, CustomModel
from utils import load_optimizer
from losses import get_loss

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
  'coatnet_0_rw_224.sw_in1k',
  'coatnet_2_rw_224.sw_in12k_ft_in1k',
  'coatnet_2_rw_224.sw_in12k'
)

EDGE_BACKBONES=(
    "regnetx_004.pycls_in1k",
    'efficientnet-b0',
    'deit_tiny_patch16_224.fb_in1k',
    'mobilevit-small',
    'mobilenetv3_large_100.ra_in1k',
    'edgenext_small.usi_in1k',
    'coat_tiny.in1k'
)

# Combine all sets
ALL_BACKBONES = {
    # "SCIENTIFIC_MODELS": SCIENTIFIC_BACKBONES,
    "PERFORMANCE_MODELS": PERFORMANCE_BACKBONES,
    "EDGE_MODELS": EDGE_BACKBONES,
}

class TestModelForwardPass(unittest.TestCase):
    """ Unit test for model forward pass with different backbone sets """

    def setUp(self):
        """ Set up test configuration """
        self.N = 10  # Number of classification classes
        self.batch_size = 2  # Two image tensors
        self.config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")

        self.config.statedicts_path = '/home/mheuillet/Desktop/state_dicts_share'

    def test_forward_pass(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        # dist.init_process_group("gloo", rank=0, world_size=1)
        dist.init_process_group("nccl", rank=0, world_size=1)

        """ Test forward pass for all backbone sets """
        for category, backbones in ALL_BACKBONES.items():
            print(f"\n🔹 Testing {category} ({len(backbones)} models) 🔹")
            for backbone in backbones:

                print(f"\n🔎 Testing backbone: {backbone}")

                for ft_type in ['full_fine_tuning', 'linear_probing']:
                    with self.subTest(backbone=backbone):

                        
                        self.config.lr1 = 1
                        self.config.lr2 = 1
                        self.config.weight_decay1 = 1
                        self.config.weight_decay2 = 1
                        self.config.ft_type = ft_type

                        try:

                            
                            # Load the model
                            self.config.backbone = backbone
                            model = load_architecture(self.config, self.N)
                            model = CustomModel(self.config, model, )
                            model.set_fine_tuning_strategy()
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = model.to(device)
                            model = DDP(model, device_ids=[0])
                            self.assertIsNotNone(model, f"❌ Model failed to load! Backbone: {backbone}")

                            # Create random input tensors simulating images (batch_size=2, 224x224)
                            dummy_input = torch.randn(self.batch_size, 3, 224, 224).to(device)
                            dummy_target = torch.randint(0, self.N, (self.batch_size,)).to(device)
                            logits = model(dummy_input)  # Get logits
                            expected_shape = (self.batch_size, self.N)

                            if not hasattr(logits, "shape"):
                                raise TypeError(f"❌ Model output is not a tensor! Backbone: {backbone}, Output Type: {type(logits)}")

                            self.assertEqual(logits.shape, expected_shape,  f"❌ Output shape mismatch! Backbone: {backbone} - Got {logits.shape}, expected {expected_shape}" )
                            print(f"✅ Forward pass successful! Backbone: {backbone}, Output shape: {logits.shape}")

                            for loss_type in ['CLASSIC_AT', 'TRADES_v2']:
                                self.config.loss_type = loss_type

                                loss_values, logits = get_loss(self.config, model, dummy_input, dummy_target)
                                loss = loss_values.mean()

                                # ✅ Check if the loss is a scalar
                                self.assertTrue(
                                    loss.ndim == 0,
                                    f"❌ Loss {loss_type} is not a scalar! Got shape: {loss.shape}"
)
                        
                            optimizer = load_optimizer(self.config, model)
                            self._validate_optimizer(optimizer, backbone)

                        except Exception as e:
                            print(f"❌ Exception in backbone {backbone}:\n{traceback.format_exc()}")  # ✅ Print full traceback
                            self.fail(f"🚨 Test failed for backbone {backbone} with error: {str(e)}")

                        print()
        dist.destroy_process_group()


    def _validate_optimizer(self, optimizer, backbone):
        """ Ensure optimizer has exactly one parameter in each head group """

        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")  # Get name, fallback to index
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            nb_params = len(group['params'])

            if "head" in group_name:
                self.assertEqual(nb_params, 1, f"❌ Error on the nb of parameters in head for {backbone}")
            if not "head" in group_name:
                if self.config.ft_type == 'full_fine_tuning':
                    self.assertGreater(nb_params, 1, f"❌ Error on the nb of parameters in FE for {backbone}")
                elif self.config.ft_type == 'linear_probing':
                    self.assertEqual(nb_params, 0, f"❌ Error on the nb of parameters in FR for {backbone} in Linear Probing")

        print(f"✅ Optimizer parameters validated for {backbone}")



# Run the unit test
if __name__ == "__main__":
    unittest.main()