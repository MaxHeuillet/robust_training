#!/bin/bash

# Define variables
seeds=1
datas=( 'Flowers'  'CIFAR100' 'EuroSAT' 'Aircraft' 'CIFAR10'  ) #  
losses=( 'APGD' 'TRADES_v2' ) 
sched='nosched'
iterations=50
aug='aug'
exp='RQ1'

init_lrs=( 0.001 0.0005 0.0001  ) # 
pruning_ratios=( 0 )
pruning_strategies=( 'random' )
batch_strategies=('random')
backbones=( #'wideresnet_28_10' 'robust_wideresnet_28_10' 
            #'deit_small_patch16_224.fb_in1k' 'robust_deit_small_patch16_224' 
            # 'vit_base_patch16_224.augreg_in1k' 'vit_base_patch16_224.augreg_in21k', 'robust_vit_base_patch16_224'
            # 'convnext_base' 'convnext_base.fb_in22k' 'robust_convnext_base'
            'convnext_tiny' #'robust_convnext_tiny' 'convnext_tiny.fb_in22k'
             ) 

ft_type=( 'full_fine_tuning' ) #'lora' ,  'linear_probing'

# Loop over architectures, pruning ratios, strategies, and learning rates
for data in "${datas[@]}"; do
  for loss in "${losses[@]}"; do
    for pruning_ratio in "${pruning_ratios[@]}"; do
      for pruning_strategy in "${pruning_strategies[@]}"; do
        for batch_strategy in "${batch_strategies[@]}"; do
          for init_lr in "${init_lrs[@]}"; do
            for id in $(seq 1 $seeds); do
              for bckbn in "${backbones[@]}"; do
                for fttype in "${ft_type[@]}"; do
                    
                  if [ "$CC_CLUSTER" = "beluga" ]; then
          
                    sbatch --export=ALL,\
NITER=$iterations,\
BCKBN=$bckbn,\
FTTYPE=$fttype,\
RATIO=$pruning_ratio,\
PSTRAT=$pruning_strategy,\
BSTRAT=$batch_strategy,\
DATA=$data,\
SEED=$id,\
LOSS=$loss,\
SCHED=$sched,\
LR=$init_lr,\
AUG=$aug,\
EXP=$exp \
./distributed_experiment_beluga.sh

                  else

                    sbatch --export=ALL,\
NITER=$iterations,\
BCKBN=$bckbn,\
FTTYPE=$fttype,\
RATIO=$pruning_ratio,\
PSTRAT=$pruning_strategy,\
BSTRAT=$batch_strategy,\
DATA=$data,\
SEED=$id,\
LOSS=$loss,\
SCHED=$sched,\
LR=$init_lr,\
AUG=$aug,\
EXP=$exp \
./distributed_experiment_other.sh

                    fi
                done
              done
            done
          done
        done
      done
    done
  done
done 