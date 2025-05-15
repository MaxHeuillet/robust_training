import argparse


def get_args2():

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--backbone", default = 'convnext_tiny.fb_in22k', type=str, help="load backbone")
    parser.add_argument('--dataset', default='flowers-102', type=str, help='dataset: ' + ' (default: cifar10)')
    parser.add_argument("--loss_function", default="TRADES_v2", type=str, help="the loss function")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument("--project_name", default = 'full_fine_tuning_5epochs_reproduce', type=str, help="task")
    parser.add_argument("--mode", default = 'train', type=str, )
        
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    #args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args_dict
