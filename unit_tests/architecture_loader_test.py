import unittest
import torch
from torch import nn

from omegaconf import OmegaConf
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torch
import sys 
import os
import traceback  # ✅ Helps capture detailed errors
import torch.nn as nn
import timm
from autoattack import AutoAttack

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from architectures import load_architecture, CustomModel
from utils import load_optimizer, move_architecture_to_tmpdir, Setup
from losses import get_loss

#  'vit_base_patch16_clip_224.laion400m_e32',


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
    "SCIENTIFIC_MODELS": SCIENTIFIC_BACKBONES,
    "PERFORMANCE_MODELS": PERFORMANCE_BACKBONES,
    "EDGE_MODELS": EDGE_BACKBONES,
}

    
EXPECTED_HEAD={
  'CLIP-convnext':{'expected_cls':timm.layers.classifier.NormMlpClassifierHead,
          'len_head_decay':1, 'len_head_nodecay':3},

  'deit':{'expected_cls':nn.Linear,
          'len_head_decay':1, 'len_head_nodecay':1},
        
  'resnet50':{'expected_cls':nn.Linear,
          'len_head_decay':1, 'len_head_nodecay':1},
              
  'vit':{'expected_cls':nn.Linear,
         'len_head_decay':1, 'len_head_nodecay':1},

  'convnext':{'expected_cls':timm.layers.classifier.NormMlpClassifierHead,
          'len_head_decay':1, 'len_head_nodecay':3},

  'eva02':{'expected_cls':nn.Linear,
           'len_head_decay':1, 'len_head_nodecay':1},

  'swin':{'expected_cls':timm.layers.classifier.ClassifierHead,
           'len_head_decay':1, 'len_head_nodecay':1},

  'coatnet':{'expected_cls':timm.layers.classifier.ClassifierHead,
            'len_head_decay':1, 'len_head_nodecay':1},

  "regnetx":{'expected_cls':timm.layers.classifier.ClassifierHead,
             'len_head_decay':1, 'len_head_nodecay':1},
             
  'efficientnet':{'expected_cls':nn.Linear,
              'len_head_decay':1, 'len_head_nodecay':1},

  'mobilevit':{'expected_cls':nn.Linear,
               'len_head_decay':1, 'len_head_nodecay':1},

  'mobilenetv3':{'expected_cls':nn.Linear,
                 'len_head_decay':1, 'len_head_nodecay':1},

  'edgenext':{'expected_cls':timm.layers.classifier.NormMlpClassifierHead,
              'len_head_decay':1, 'len_head_nodecay':3},

  'coat':{'expected_cls':nn.Linear,
          'len_head_decay':1, 'len_head_nodecay':1},
}

def get_expected_head_type(arch_name):
    """
    Match the architecture name to the most specific key in expected_head_dict.
    Longer keys are prioritized (e.g., "CLIP-convnext" before "convnext").
    """
    for key in sorted(EXPECTED_HEAD.keys(), key=len, reverse=True):
        if key in arch_name:
            return EXPECTED_HEAD[key]['expected_cls']
    raise ValueError(f"No expected head type found for architecture '{arch_name}'")

def get_expected_optmizer_params(arch_name):
    """
    Match the architecture name to the most specific key in expected_head_dict.
    Longer keys are prioritized (e.g., "CLIP-convnext" before "convnext").
    """
    for key in sorted(EXPECTED_HEAD.keys(), key=len, reverse=True):
        if key in arch_name:
            return EXPECTED_HEAD[key]['len_head_decay'], EXPECTED_HEAD[key]['len_head_nodecay'],
    raise ValueError(f"No expected head type found for architecture '{arch_name}'")

class TestModelForwardPass(unittest.TestCase):
    """ Unit test for model forward pass with different backbone sets """

    def setUp(self):
        """ Set up test configuration """
        self.N = 10  # Number of classification classes
        self.batch_size = 1  # Two image tensors
        self.config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")
        
        # self.config.statedicts_path = '/home/mheuillet/Desktop/state_dicts_share'

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

                for ft_type in ['linear_probing', 'full_fine_tuning']:

                    with self.subTest(backbone=backbone):
                            
                        self.config.lr1 = 1
                        self.config.lr2 = 1
                        self.config.weight_decay1 = 1
                        self.config.weight_decay2 = 1
                        self.config.ft_type = ft_type

                        try:
    
                            # Load the model
                            self.config.backbone = backbone
                            move_architecture_to_tmpdir(self.config)
                            model = load_architecture(self.config, self.N)
                            self.assertIsNotNone(model, f"❌ Model failed to load! Backbone: {backbone}")
                            model = CustomModel(self.config, model, )
                                
                            ### test the retrieval of the head module:

                            head_submodule = model.get_classification_head()
                            expected_cls = get_expected_head_type(backbone)

                            assert isinstance(head_submodule, expected_cls), (
                                        f"For {backbone}, expected {expected_cls} but got {type(head_submodule)}"
                                    )

                            ### test the gradient tracking
                                
                            device = torch.device("cuda")
                            model = model.to(device)
                            model = DDP(model, device_ids=[0])

                            if self.config.ft_type == 'linear_probing':
                                self._test_linear_probing(model)
                                self._test_linear_probing_train_eval_modes(model)
                            elif self.config.ft_type == 'full_fine_tuning':
                                self._test_full_fine_tuning(model)
                                self._test_full_finetuning_train_eval_modes(model)
                            else:
                                print('raise an error here')


                            setup = Setup(1)
                            bs = setup.train_batch_size(self.config)
                            dummy_input = torch.randn(bs, 3, 224, 224).to(device)
                            dummy_target = torch.randint(0, self.N, (bs,)).to(device)


                            for loss_type in ['CLASSIC_AT', 'TRADES_v2']:

                                ### test the loss
                                self.config.loss_type = loss_type
                                loss_values, logits = get_loss(self.config, model, dummy_input, dummy_target)
                                loss = loss_values.mean()

                                self.assertTrue(loss.ndim == 0,
                                     f"❌ Loss {loss_type} is not a scalar! Got shape: {loss.shape}" )

                                ### test the optimizer 
                                optimizer = load_optimizer(self.config, model)
                                self._validate_optimizer(self.config, optimizer, model, backbone)

                                ### test the forward pass output shape
                                logits = model(dummy_input)  # Get logits
                                expected_shape = (bs, self.N)

                                if not hasattr(logits, "shape"):
                                    raise TypeError(f"❌ Model output is not a tensor! Backbone: {backbone}, Output Type: {type(logits)}")

                                self.assertEqual(logits.shape, expected_shape,  f"❌ Output shape mismatch! Backbone: {backbone} - Got {logits.shape}, expected {expected_shape}" )

                                self.test_autoattack_batch_pass(model)
                        
                        except Exception as e:
                            print(f"❌ Exception in backbone {backbone}:\n{traceback.format_exc()}")  # ✅ Print full traceback
                            self.fail(f"🚨 Test failed for backbone {backbone} with error: {str(e)}")

                        
        dist.destroy_process_group()


    def _test_linear_probing_train_eval_modes(self, model):
        """
        Ensures that when we call model.train() in linear_probing mode:
        - The backbone is forced to eval (training=False)
        - The classification head is in train mode (training=True)
        Then ensures model.eval() sets everything to eval (training=False).
        """

        # 1. Freeze backbone & unfreeze head
        model.module.set_fine_tuning_strategy()

        # 2. Enter train mode on the DDP wrapper
        model.train()

        # 3. Identify classification head module
        head_module = model.module.get_classification_head()

        # We'll gather all submodules in the head so we can distinguish them
        head_submodules = set(head_module.modules())  # includes head_module itself + children

        # 4. Check states:
        for name, submodule in model.module.base_model.named_modules():
            # Skip the top-level base_model itself to avoid confusion
            if submodule is model.module.base_model:
                continue

            if submodule in head_submodules:
                # Classification head submodule → expect train=True
                assert submodule.training, (
                    f"Submodule '{name}' (in classification head) should be train=True, "
                    "but got train=False."
                )
            else:
                # Backbone submodule → expect train=False
                assert not submodule.training, (
                    f"Submodule '{name}' (in backbone) should be train=False, "
                    "but got train=True."
                )

        # 5. Now call model.eval() and confirm everything becomes eval
        model.eval()

        for name, submodule in model.module.base_model.named_modules():
            # After eval(), *all* submodules should be training=False
            assert not submodule.training, (
                f"After model.eval(), submodule '{name}' should be train=False, "
                "but got train=True."
            )

    def _test_full_finetuning_train_eval_modes(self, model):
        """
        Ensures that in full_fine_tuning mode:
        - All submodules in the backbone (and head) go to train mode when we call model.train().
        - All submodules go to eval mode when we call model.eval().
        """
        # 1. Apply the fine-tuning strategy
        model.module.set_fine_tuning_strategy()

        # 2. Put the DDP-wrapped model in train mode
        model.train()

        # 3. Check that every submodule in base_model is in training mode
        for name, submodule in model.module.base_model.named_modules():
            # If you want to skip checking the top-level base_model container, do:
            if submodule is model.module.base_model:
                continue
            assert submodule.training, (
                f"[FULL FINETUNE] After model.train(), submodule '{name}' should be training=True "
                f"but got {submodule.training}"
            )

        # 4. Now call eval()
        model.eval()

        # 5. Confirm every submodule in base_model is now eval
        for name, submodule in model.module.base_model.named_modules():
            if submodule is model.module.base_model:
                continue
            assert not submodule.training, (
                f"[FULL FINETUNE] After model.eval(), submodule '{name}' should be training=False "
                f"but got {submodule.training}"
            )

    def test_autoattack_batch_pass(self, model):
        """
        Test if AutoAttack can process a single synthetic batch without OOM or shape issues.
        """

        device = torch.device("cuda")

        def forward_pass(x):
            return model(x)

        # Config
        corruption_type = 'Linf'
        epsilon = self.config.epsilon if hasattr(self.config, "epsilon") else 8 / 255  # fallback default
        batch_size = self.batch_size
        input_shape = (3, 224, 224)  # Standard image size

        setup = Setup(1)
        device = torch.device("cuda")
        bs = setup.test_batch_size(self.config)
        x_nat = torch.randn(bs, 3, 224, 224).to(device)
        target = torch.randint(0, self.N, (bs,)).to(device)


        # Initialize AutoAttack
        adversary = AutoAttack(
            forward_pass,
            norm=corruption_type,
            eps=epsilon,
            version='standard',
            device=device,
            verbose=False
        )

        try:
            # Run attack
            x_adv = adversary.run_standard_evaluation(x_nat, target, bs=batch_size)

            # Forward both clean and adversarial
            logits_nat, logits_adv = model(x_nat, x_adv)

            # Compute predictions
            preds_nat = torch.argmax(logits_nat, dim=1)
            preds_adv = torch.argmax(logits_adv, dim=1)

            nat_correct = (preds_nat == target).sum().item()
            adv_correct = (preds_adv == target).sum().item()

            print(f"✅ Clean Acc: {nat_correct}/{batch_size}, Adversarial Acc: {adv_correct}/{batch_size}")

        except RuntimeError as e:
            self.fail(f"❌ OOM or other RuntimeError during AutoAttack: {e}")



    def _test_linear_probing(self, model):

        # Apply fine-tuning strategy
        model.module.set_fine_tuning_strategy()

        head_module = model.module.get_classification_head()
        head_params = set(p for p in head_module.parameters())

        for name, param in model.module.base_model.named_parameters():
            if param in head_params:
                assert param.requires_grad, f"{name} should be trainable (head param)"
            else:
                assert not param.requires_grad, f"{name} should be frozen (feature extractor)"

    def _test_full_fine_tuning(self, model):

        # Apply fine-tuning strategy
        model.module.set_fine_tuning_strategy()

        for name, param in model.module.base_model.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"


    def _validate_optimizer(self, config, optimizer, model, backbone):
        """ Ensure optimizer has exactly one parameter in each head group """

        assert len(optimizer.param_groups) == 4, "Expected 4 param groups."

        # -----------------------------
        # 5) Check param groups meta
        # -----------------------------
        # We'll store them for convenience:
        pg_backbone_decay = optimizer.param_groups[0]
        pg_backbone_no_decay = optimizer.param_groups[1]
        pg_head_decay = optimizer.param_groups[2]
        pg_head_no_decay = optimizer.param_groups[3]

        # A) Check the "name" field if you rely on that:
        assert pg_backbone_decay.get('name') == 'backbone_decay'
        assert pg_backbone_no_decay.get('name') == 'backbone_no_decay'
        assert pg_head_decay.get('name') == 'head_decay'
        assert pg_head_no_decay.get('name') == 'head_no_decay'

        # B) Check LR and weight_decay for each group
        assert pg_backbone_decay['lr'] == config.lr1
        assert pg_backbone_decay['weight_decay'] == config.weight_decay1

        assert pg_backbone_no_decay['lr'] == config.lr1
        assert pg_backbone_no_decay['weight_decay'] == 0.0

        assert pg_head_decay['lr'] == config.lr2
        assert pg_head_decay['weight_decay'] == config.weight_decay2

        assert pg_head_no_decay['lr'] == config.lr2
        assert pg_head_no_decay['weight_decay'] == 0.0

        # -----------------------------
        # 6) Inspect actual param assignment
        # -----------------------------
        # We'll gather references to the actual param Tensors from each group
        backbone_decay_params   = set(pg_backbone_decay['params'])
        backbone_no_decay_params = set(pg_backbone_no_decay['params'])
        head_decay_params       = set(pg_head_decay['params'])
        head_no_decay_params    = set(pg_head_no_decay['params'])

        # Check they are disjoint
        assert (
            backbone_decay_params & backbone_no_decay_params & head_decay_params & head_no_decay_params
        ) == set(), "No param should appear in multiple groups."

        # Now systematically check each param in base_model
        # We'll differentiate "backbone" vs "head" by seeing if param is in the classification head module
        # (like you do in load_optimizer).
        classification_head = model.module.get_classification_head() 
        head_param_set = set(classification_head.parameters())

        # We'll define a helper to see if a name suggests no_decay
        def is_no_decay(name: str) -> bool:
            return ("bias" in name or "norm" in name or "bn" in name)

        # Now check each parameter in the model
        for full_name, param in model.module.base_model.named_parameters(): 
            if not param.requires_grad:
                continue  # skip frozen or untrainable
            
            in_head = param in head_param_set
            should_be_no_decay = is_no_decay(full_name)

            # Decide which group we expect
            if in_head:
                # This param belongs to head
                if should_be_no_decay:
                    assert param in head_no_decay_params, (
                        f"Param '{full_name}' is in head + no_decay but not found in 'head_no_decay' group."
                    )
                else:
                    assert param in head_decay_params, (
                        f"Param '{full_name}' is in head + decay but not found in 'head_decay' group."
                    )
            else:
                # This param belongs to backbone
                if should_be_no_decay:
                    assert param in backbone_no_decay_params, (
                        f"Param '{full_name}' is in backbone + no_decay but not in 'backbone_no_decay' group."
                    )
                else:
                    assert param in backbone_decay_params, (
                        f"Param '{full_name}' is in backbone + decay but not in 'backbone_decay' group."
                    )

        len_head_decay, len_head_nodecay = get_expected_optmizer_params(backbone)
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")  # Get name, fallback to index
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            nb_params = len(group['params'])

            if not "head" in group_name:
                if config.ft_type == 'full_fine_tuning':
                    self.assertGreater(nb_params, 0, f"❌ Error on the nb of parameters ")
                elif config.ft_type == 'linear_probing':
                    self.assertEqual(nb_params, 0, f"❌ Error on the nb of parameters ")
            
            if "head_decay" in group_name:
                self.assertEqual(nb_params, len_head_decay, f"❌ Error on the nb of parameters ")

            elif "head_no_decay" in group_name:
                self.assertEqual(nb_params, len_head_nodecay, f"❌ Error on the nb of parameters ")

# Run the unit test
if __name__ == "__main__":
    unittest.main()