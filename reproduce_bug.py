import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.models import resnet50
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label


def update(rank, args): 

        dataset, world_size = args

        setup(world_size, rank)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=world_size) 

        model = resnet50(pretrained=True)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        model.train()

        optimizer = torch.optim.SGD( model.parameters(),lr=0.001, weight_decay=0.0001, momentum=0.9, nesterov=True, )
        
        print('start epochs')
        epochs = 5
        for epoch in range(epochs):
            total_loss, total_err = 0.,0.
            sampler.set_epoch(epoch)
            for data, target in loader:
                data, target = data.to(rank), target.to(rank)
                optimizer.zero_grad()
                logits, loss = trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer,)

                loss.backward()
                optimizer.step()
        cleanup()


def trades_loss(model, x_natural, y,optimizer,step_size=0.003,epsilon=0.031,perturb_steps=10, beta=1.0, distance='l_inf'):

    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()

    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()
        for _ in range(perturb_steps):
            x_adv = x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax( model(x_adv), dim=1), F.softmax( model(x_natural), dim=1))

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon).detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return logits, loss


def setup(world_size, rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print('init process group ok')

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':

    world_size = 4
    data = [torch.rand(3, 250, 250) for _ in range(100)]
    labels = torch.randint(0, 10, (100,))
    dataset = CustomDataset(data, labels)
    arg = (dataset, world_size)
    torch.multiprocessing.spawn(update, args=(arg,), nprocs=world_size, join=True)
