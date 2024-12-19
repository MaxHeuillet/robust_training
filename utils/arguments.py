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
    parser.add_argument("--loss_function", default="TRADES_v2", type=str, help="the loss function")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument("--ft_type", default = 'full_fine_tuning', type=str, help="fine tuning type")
    parser.add_argument("--task", default = 'HPO', type=str, help="task")
    parser.add_argument("--project_name", default = '0001', type=str, help="task")
        
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    #args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args_dict
