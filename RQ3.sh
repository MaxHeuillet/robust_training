#!/bin/bash

# Define variables
seeds=1
datas=( 'Flowers' ) # 'CIFAR100' 'EuroSAT' 'Aircraft' 'CIFAR10'  
losses=( 'CLASSIC_AT' ) #'TRADES_v2' 
exp='RQ1'

backbones=( 
            #'deit_small_patch16_224.fb_in1k' 'robust_deit_small_patch16_224' 'random_deit_small_patch16_224' 
            #'vit_base_patch16_224.augreg_in1k' 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224' 'random_vit_base_patch16_224'
            #'convnext_base' 'convnext_base.fb_in22k' 'robust_convnext_base' 'random_convnext_base'
            'convnext_tiny' # 'robust_convnext_tiny' 'convnext_tiny.fb_in22k' 'random_convnext_tiny' 
           ) # 'wideresnet_28_10' 'robust_wideresnet_28_10' 

ft_type=( 'full_fine_tuning' ) #'lora' ,  'linear_probing'

# Loop over architectures, pruning ratios, strategies, and learning rates
for data in "${datas[@]}"; do
  for loss in "${losses[@]}"; do
    for id in $(seq 1 $seeds); do
      for bckbn in "${backbones[@]}"; do
        for fttype in "${ft_type[@]}"; do
                    
            if [ "$CC_CLUSTER" = "beluga" ]; then
          
                    sbatch --export=ALL,\
BCKBN=$bckbn,\
FTTYPE=$fttype,\
DATA=$data,\
SEED=$id,\
LOSS=$loss,\
EXP=$exp \
./distributed_experiment_beluga.sh

            else

                    sbatch --export=ALL,\
BCKBN=$bckbn,\
FTTYPE=$fttype,\
DATA=$data,\
SEED=$id,\
LOSS=$loss,\
EXP=$exp \
./distributed_experiment_other.sh

          fi

        done
      done
    done
  done
done 