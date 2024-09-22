import argparse
import torch
import os


def get_exp_name(args):

    grd = ""
    grd += args.loss_function
    grd += f"_{args.init_lr}"
    grd += f"_{args.sched}"
    grd += f"_{args.dataset}" 
    grd += f"_{args.arch}" 

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
    else:
        default_data_dir = './data'

    # General options
    parser.add_argument("--arch", default="resnet50",type=str, choices=['resnet50', 'LeNet5'], help="model architecture")
    parser.add_argument('--dataset', default='CIFAR10', type=str,choices=['CIFAR10', 'CIFAR10s', 'CIFAR100', 'tinyimagenet', 'MNIST'], help='dataset: ' + ' (default: cifar10)')
    
    parser.add_argument("--pruning_strategy", default="random", type=str, help="the pruning strategy")
    parser.add_argument("--pruning_ratio", default=0, type=float,choices=[0, 0.3, 0.5, 0.7], help="the pruning ratio")
    parser.add_argument("--batch_strategy", default="random",type=str, choices=['random',], help="the batching strategy")
    parser.add_argument("--aug", type=str, choices=['aug', 'noaug',], help="use data augmentation")

    parser.add_argument("--iterations", default=10, type=int, metavar="N", help="number of total iterations to run")
    parser.add_argument("--delta", default=0.875, type=float,help="the amount of pruning iterations")
    parser.add_argument("--batch_size", default=128, type=int, help="mini-batch size (default: 128)")
    parser.add_argument("--sample_size", default=256, type=int, help="mini-batch sampling size (default: 256)")

    parser.add_argument("--loss_function", default="TRADES_v2", type=str,choices=['TRADES', 'TRADES_v2', 'TRADES_v3'], help="the loss function")
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
    parser.add_argument("--epsilon", default=8/255, type=float, help="epsilon of trades")
    parser.add_argument("--step_size", default=2/255, type=float, help="step size of trades")
    parser.add_argument("--perturb_steps", default=10, type=int, help="number of steps of trades")
    parser.add_argument("--beta", default=1.0, type=float, help="beta of trades")
    parser.add_argument("--distance", default='l_inf', type=str, help="distance of trades")

    parser.add_argument("--c_fixed", default=False, type=bool, help="assume c=0 in exponential decay")


    ### arguments for diffusion augmented learning:
    parser.add_argument("--unsup_fraction", default=0.3, type=float, help="fraction of data generated from diffusion model")
    


    args, unknown = parser.parse_known_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return args



# def parse_bool(v):
#     if v.lower()=='true':
#         return True
#     elif v.lower()=='false':
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

    # parser.add_argument('--data_dir', default='~/scratch/data')

    # parser.add_argument("--resume_from_iterations", default=0, type=int, help="resume from a specific iterations")


    # parser.add_argument("--save_freq", type=int, default=200, help="Saves checkpoints at every specified number of iterations")
    # parser.add_argument("--gpu", type=int, nargs='+', default=[0])

    # parser.add_argument("--selection_method", default="crest", choices=['none', 'random', 'crest'], help="subset selection method")
    # parser.add_argument("--smtk", type=int, help="smtk", default=0)
    # parser.add_argument("--train_frac", "-s", type=float, default=0.1, help="training fraction")
    # parser.add_argument("--lr_milestones", type=int, nargs='+', default=[100,150])
    # parser.add_argument("--gamma", type=float, default=0.1, help="learning rate decay parameter")
    # 
    # parser.add_argument("--runs", type=int, help="num runs", default=1)
    # parser.add_argument("--warm_start_iterations", default=20, type=int, help="iterations to warm start learning rate")
    # parser.add_argument("--subset_start_iterations", default=0, type=int, help="iterations to start subset selection")

    # data augmentation options
    # parser.add_argument("--cache_dataset", default=True, type=parse_bool, const=True, nargs='?', help="cache the dataset in memory")
    # parser.add_argument("--clean_cache_selection", default=False, type=parse_bool, const=True, nargs='?', help="clean the cache when selecting a new subset")
    # parser.add_argument("--clean_cache_iteration", default=True, type=parse_bool, const=True, nargs='?', help="clean the cache after iterating over the dataset")

    # Crest options
    # parser.add_argument("--approx_moment", default=True, type=parse_bool, const=True, nargs='?', help="use momentum in approximation")
    # parser.add_argument("--approx_with_coreset", default=True, type=parse_bool, const=True, nargs='?', help="use all (selected) coreset data for loss function approximation")
    # parser.add_argument("--check_interval", default=20, type=int, help="frequency to check the loss difference")
    # parser.add_argument("--num_minibatch_coreset", default=5, type=int, help="number of minibatches to select together")
    # parser.add_argument("--batch_num_mul", default=5, type=float, help="multiply the number of minibatches to select together")
    # parser.add_argument("--interval_mul", default=1., type=float, help="multiply the interval to check the loss difference")
    # parser.add_argument("--check_thresh_factor", default=0.1,type=float, help="use loss times this factor as the loss threshold",)
    # parser.add_argument("--shuffle", default=True, type=parse_bool, const=True, nargs='?',help="use shuffled minibatch coreset")

    # random subset options
    # parser.add_argument("--random_subset_size", default=0.01, type=float, help="partition the training data to select subsets")
    # parser.add_argument("--partition_start", default=0, type=int, help="which iterations to start selecting by minibatches")

    # dropping examples below a loss threshold
    # parser.add_argument('--drop_learned', default=True, type=parse_bool, const=True, nargs='?', help='drop learned examples')
    # parser.add_argument('--watch_interval', default=5, type=int, help='decide whether an example is learned based on how many iterations')
    # parser.add_argument('--drop_interval', default=20, type=int, help='decide whether an example is learned based on how many iterations')
    # parser.add_argument('--drop_thresh', default=0.1, type=float, help='loss threshold')
    # parser.add_argument('--min_train_size', default=40000, type=int)

    # others
    # parser.add_argument('--use_wandb', default=False, type=parse_bool, const=True, nargs='?')

# args = parser.parse_args()
    # if args.dataset == 'CIFAR10':
    #     args.num_classes = 10
    # elif args.dataset == 'CIFAR100':
    #     args.num_classes = 100
    # elif args.dataset == 'tinyimagenet':
    #     args.num_classes = 200
    # else:
    #     raise NotImplementedError
    

    # args.save_dir = get_exp_name(args)