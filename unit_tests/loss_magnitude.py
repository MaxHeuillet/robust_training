import torch
import matplotlib.pyplot as plt
import os
import sys
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import nn

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from architectures import load_architecture, CustomModel
from corruptions import apgd_attack
from losses import get_eval_loss


SCIENTIFIC_BACKBONES=(
#   'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K',
#   'CLIP-convnext_base_w-laion2B-s13B-b82K',
#   'deit_small_patch16_224.fb_in1k',
#   'robust_resnet50',
#   'vit_small_patch16_224.augreg_in21k',
#   'convnext_base.fb_in1k',
  'resnet50.a1_in1k',
#   'robust_vit_base_patch16_224',
#   'vit_base_patch16_224.mae',
#   'convnext_base.fb_in22k',
#   'robust_convnext_base',
#   'vit_base_patch16_224.augreg_in1k',
#   'vit_base_patch16_224.augreg_in21k',
#   'vit_base_patch16_224.dino',
#   'vit_base_patch16_clip_224.laion2b',
#   'convnext_tiny.fb_in1k',
#   'robust_convnext_tiny',
#   'robust_deit_small_patch16_224',
#   'vit_small_patch16_224.augreg_in1k',
#   'convnext_tiny.fb_in22k',
) 

PERFORMANCE_BACKBONES=(
#   'vit_base_patch16_clip_224.laion2b_ft_in1k',
#   'vit_base_patch16_224.augreg_in21k_ft_in1k',
#   'vit_small_patch16_224.augreg_in21k_ft_in1k',
#   'eva02_base_patch14_224.mim_in22k',
#   'eva02_tiny_patch14_224.mim_in22k',
#   'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
#   'swin_tiny_patch4_window7_224.ms_in1k',
#   'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',
#   'convnext_base.fb_in22k_ft_in1k',
#   'convnext_tiny.fb_in22k_ft_in1k',
#   'coatnet_0_rw_224.sw_in1k',
#   'coatnet_2_rw_224.sw_in12k_ft_in1k',
#   'coatnet_2_rw_224.sw_in12k'
)

EDGE_BACKBONES=(
    "regnetx_004.pycls_in1k",
    # 'efficientnet-b0',
    # 'deit_tiny_patch16_224.fb_in1k',
    # 'mobilevit-small',
    # 'mobilenetv3_large_100.ra_in1k',
    # 'edgenext_small.usi_in1k',
    # 'coat_tiny.in1k'
)

# Combine all sets
ALL_BACKBONES = {
    "SCIENTIFIC_MODELS": SCIENTIFIC_BACKBONES,
    "PERFORMANCE_MODELS": PERFORMANCE_BACKBONES,
    "EDGE_MODELS": EDGE_BACKBONES,
}


def run_apgd_experiment_all_backbones(config, ALL_BACKBONES):
    """
    For each category, for each backbone in ALL_BACKBONES[category]:
      1) Load the model (DDP)
      2) Generate a random single image + label
      3) For each of the 4 combos:
         (ft_type in ["full_fine_tuning","linear_probing"],
          loss_type in ["CLASSIC_AT","TRADES_v2"])
         -> run APGD attack for i=1..10, compute loss, store
      4) Plot all 4 combos horizontally in ONE figure & save
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the process group for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=0, world_size=1)

    # Create directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Loop over each category in ALL_BACKBONES
    for category, backbone_list in ALL_BACKBONES.items():
        print(f"\n===== Category: {category} ({len(backbone_list)} models) =====")

        # Loop over each backbone in this category
        for backbone in backbone_list:
            print(f"\n===== Running APGD experiment on backbone: {backbone} =====")

            # 1) Load the model for this backbone
            config.backbone = backbone
            model = load_architecture(config, N=10)  # adapt your real function
            model = CustomModel(config, model)
            model = model.to(device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[0])
            model.eval()  # put in eval mode for the attacks, if needed

            # 2) Create a single random observation (B=1)
            x = torch.randn(1, 3, 224, 224, device=device)
            y = torch.randint(0, 10, (1,), device=device)

            # Prepare a single figure with 4 horizontal subplots
            fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
            subplot_index = 0

            # 3) For each fine-tuning type and loss type, gather losses
            for ft_type in ["full_fine_tuning", "linear_probing"]:
                config.fine_tuning_type = ft_type  # or config.ft_type = ft_type
                for loss_type in ["CLASSIC_AT", "TRADES_v2"]:
                    config.loss_function = loss_type  # or config.loss_type

                    # Gather the losses for 10 iterations
                    losses_over_iterations = []
                    for i in range(1, 11):
                        config.perturb_steps = i

                        # Run APGD
                        x_adv = apgd_attack(config, model, x, y)

                        # Compute the loss
                        loss_values, _, _ = get_eval_loss(config, model, x_adv, y)
                        loss_scalar = loss_values.mean().item()
                        losses_over_iterations.append(loss_scalar)

                    # Plot in the correct subplot
                    ax = axs[subplot_index]
                    ax.plot(range(1, 11), losses_over_iterations, marker='o')
                    ax.set_title(f"{ft_type}, {loss_type}")
                    ax.set_xlabel("APGD Iterations")
                    ax.set_ylabel("Loss")
                    ax.grid(True)

                    subplot_index += 1

            # Once all 4 subplots are filled, save this figure for the backbone
            safe_backbone = backbone.replace("/", "-").replace(" ", "_")
            save_filename = f"plots/{category}_{safe_backbone}_4subplots.png"
            fig.suptitle(f"{category} - Backbone: {backbone}")
            plt.tight_layout()
            plt.savefig(save_filename)
            plt.close(fig)

            print(f"Saved combined figure to {save_filename}")

    dist.destroy_process_group()
    print("All APGD experiments completed.")



def combine_pngs_to_pdf(input_dir="plots", output_pdf="combined_plots.pdf"):
    """
    Combines all PNG images in `input_dir` into a single multi-page PDF file at `output_pdf`.
    Order is alphabetical by filename.
    """

    # Find all .png files in the directory
    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    # Create/overwrite the multi-page PDF
    with PdfPages(output_pdf) as pdf:
        for png_path in png_files:
            print(f"Adding {png_path} to PDF...")
            # Read the image
            img = mpimg.imread(png_path)
            
            # Make a new figure per image
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.set_title(os.path.basename(png_path))
            ax.axis('off')  # hide axes if you want a clean image

            # Save this figure (page) to the PDF
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nAll images combined into '{output_pdf}'")



if __name__ == "__main__":
    # Example config, adapt as needed
    # config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")        
    # config.statedicts_path = '/home/mheuillet/Desktop/state_dicts_share'
    # run_apgd_experiment_all_backbones(config, ALL_BACKBONES)
    # combine_pngs_to_pdf(input_dir="plots", output_pdf="all_backbones_combined.pdf")
