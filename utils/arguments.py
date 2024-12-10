import argparse
import torch
import os


def get_exp_name(args):
    # Access all attributes of args as a dictionary
    args_dict = vars(args) if hasattr(args, '__dict__') else args.__dict__

    # Create the experiment name dynamically by concatenating all key-value pairs
    exp_name = "_".join(f"{key}={value}" for key, value in args_dict.items())

    return exp_name





def get_args2():

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--backbone", default = 'deit_small_patch16_224.fb_in1k', type=str, help="load backbone")
    parser.add_argument('--dataset', default='Flowers', type=str, help='dataset: ' + ' (default: cifar10)')
    parser.add_argument("--loss_function", default="CLASSIC_AT", type=str, help="the loss function")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument("--ft_type", default = 'full_fine_tuning', type=str, help="fine tuning type")
    parser.add_argument("--task", default = 'HPO', type=str, help="task")
    parser.add_argument("--project_name", default = '0000', type=str, help="task")
    

    # General options
    # parser.add_argument("--arch", default="resnet50",type=str, help="model architecture")
    # parser.add_argument("--pre_trained", type=str, help="load pretrained non robust model")
    # 

    # parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total iterations to run")
    # parser.add_argument("--log_dir", default="./logs", type=str, help="The directory used to save logs")
    # parser.add_argument("--statedict_dir", default=statedict_dir, type=str, help="The directory used to save state dics")
    # parser.add_argument('--data_dir', default=data_dir,type=str,)

    ### arguments for TRADES loss function:
    # parser.add_argument("--epsilon", default=4/255, type=float, help="epsilon of trades") #8/255
    # parser.add_argument("--step_size", default=2/255, type=float, help="step size of trades")
    # parser.add_argument("--perturb_steps", default=10, type=int, help="number of steps of trades")
    # parser.add_argument("--distance", default='Linf', type=str, help="distance of trades")
    # parser.add_argument("--beta", default=1.0, type=float, help="beta of trades")

    
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    #args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args_dict
