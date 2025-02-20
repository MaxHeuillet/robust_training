import sys
sys.path.append('/home/mheuillet/Desktop/robust_training')

import architectures.resnet_cifar10
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import utils
from datasets import IndexedDataset
import logging
import time

args = utils.get_args()

# Set up logging

logger = logging.getLogger( args.log_dir.split('/')[-1] + time.strftime("-%Y-%m-%d-%H-%M-%S") )

logging.basicConfig(
    filename=f"{args.log_dir}/output.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,)

# define a Handler which writes INFO messages or higher to the sys.stderr
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
args.logger = logger

# Print arguments
args.logger.info("Arguments: {}".format(args))
args.logger.info("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))


def main(args):

    train_dataset = IndexedDataset(args, train=True)
    args.train_size = len(train_dataset)

    val_loader = torch.utils.data.DataLoader(
            IndexedDataset(args, train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,   )

    model = architectures.resnet_cifar10.resnet50(pretrained=False, progress=True).to(args.device)

    from trainers import CRESTTrainer
    trainer = CRESTTrainer(args, model, train_dataset, val_loader )

    trainer.train()

if __name__ == "__main__":
    main(args)



