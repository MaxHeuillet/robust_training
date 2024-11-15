import argparse
import torch
import os


def get_exp_name(args):

    grd = ""
    grd += args.loss_function
    grd += f"_{args.init_lr}"
    grd += f"_{args.sched}"
    grd += f"_{args.dataset}" 
    grd += f"_{args.backbone}" 

    grd += f"_{args.pruning_strategy}" 
    grd += f"_{args.pruning_ratio}"  

    grd += f"_{args.batch_strategy}"          
    grd += f"_{args.iterations}"
    
    grd += f"_{args.seed}"

    return grd

def get_args():

    parser = argparse.ArgumentParser()

    if "calculquebec" in os.uname().nodename:  # Check for a substring that is unique to the cluster
        default_data_dir = '~/scratch/data'
    elif "calcul.quebec" in os.uname().nodename:
        default_data_dir = '~/scratch/data'
    else:
        default_data_dir = './data'

    # General options
    # parser.add_argument("--arch", default="resnet50",type=str, help="model architecture")
    # parser.add_argument("--pre_trained", type=str, help="load pretrained non robust model")
    parser.add_argument("--backbone", type=str, help="load backbone")
    parser.add_argument("--finetuning_type", type=str, help="fine tuning type")

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset: ' + ' (default: cifar10)')
    parser.add_argument("--pruning_strategy", default="random", type=str, help="the pruning strategy")
    parser.add_argument("--pruning_ratio", default=0, type=float, help="the pruning ratio")
    parser.add_argument("--batch_strategy", default="random",type=str, choices=['random',], help="the batching strategy")
    parser.add_argument("--aug", type=str, choices=['aug', 'noaug',], help="use data augmentation")

    parser.add_argument("--iterations", default=10, type=int, metavar="N", help="number of total iterations to run")
    parser.add_argument("--delta", default=1, type=float,help="the proportion of pruning iterations")
    parser.add_argument("--batch_size", default=2, type=int, help="mini-batch size (default: 128)") #64
    parser.add_argument("--sample_size", default=64, type=int, help="mini-batch sampling size (default: 256)")

    parser.add_argument("--loss_function", default="TRADES_v2", type=str, help="the loss function")
    parser.add_argument("--init_lr", type=float,help="initial learning rate")
    parser.add_argument("--sched", default='sched',type=str, choices=['sched', 'nosched'], help="the scheduler")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", "--wd", default=1e-4, type=float, help="weight decay (default: 5e-4)")
    
    parser.add_argument('--seed', default=0, type=int, help="random seed")

    parser.add_argument("--log_dir", default="./logs", type=str, help="The directory used to save logs")
    parser.add_argument("--statedict_dir", default="./state_dicts", type=str, help="The directory used to save state dics")
    parser.add_argument('--data_dir', default=default_data_dir,type=str,)
    print(default_data_dir)

    parser.add_argument('--task', default='train',type=str, choices=['train', 'eval'], help="wether to train or to evaluate the model")

    parser.add_argument("--num_workers",default=1, type=int, help="number of data loading workers (default: 4)")

    ### arguments for TRADES loss function:
    parser.add_argument("--epsilon", default=4/255, type=float, help="epsilon of trades") #8/255
    parser.add_argument("--step_size", default=2/255, type=float, help="step size of trades")
    parser.add_argument("--perturb_steps", default=10, type=int, help="number of steps of trades")
    parser.add_argument("--beta", default=1.0, type=float, help="beta of trades")
    parser.add_argument("--distance", default='l_inf', type=str, help="distance of trades")

    ### arguments for diffusion augmented learning:
    parser.add_argument("--unsup_fraction", default=0.3, type=float, help="fraction of data generated from diffusion model")
    parser.add_argument("--exp", default='', type=str, help="the experiment type")
    


    args, unknown = parser.parse_known_args()

    #args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args
