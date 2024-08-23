import torch
from architectures.resnet_cifar10 import ResNet_cifar10, Bottleneck_cifar10
from architectures.resnet_imagenet import ResNet_imagenet, Bottleneck_imagenet
from architectures.LeNet import LeNet5

def load_architecture(args,):

    if args.arch == 'resnet50' and args.dataset == 'CIFAR10':
        model = ResNet_cifar10(Bottleneck_cifar10, [3, 4, 6, 3] )
        state_dict = torch.load('./state_dicts/resnet50_cifar10.pt')
        model.load_state_dict(state_dict)
        model.to('cuda')

        target_layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
                model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
                model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
                model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]
        
    elif args.arch == 'LeNet5':
        model = LeNet5()
        model.to('cuda')
        target_layers = [model.conv1.weight, model.conv2.weight, model.fc1.weight, model.fc2.weight,model.fc3.weight] 

    elif args.arch == 'resnet50' and args.dataset in ['Imagenet1k' , 'Imagenette']:
        model = ResNet_imagenet(Bottleneck_imagenet, [3, 4, 6, 3], )
        state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
        model.load_state_dict(state_dict)
        model.to('cuda')

        target_layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
                model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
                model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
                model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]

    return model, target_layers




def load_statedict(args,):

    if args.arch == 'resnet50' and args.dataset == 'CIFAR10':
        
        state_dict = torch.load('./state_dicts/resnet50_cifar10.pt')
        
        
    elif args.arch == 'LeNet5':
        print('no state dict at the moment')

    elif args.arch == 'resnet50' and args.dataset in ['Imagenet1k' , 'Imagenette']:
        
        state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')


    return state_dict