import torch
from architectures.resnet_cifar10 import ResNet_cifar10, Bottleneck_cifar10
from architectures.resnet_imagenet import ResNet_imagenet, Bottleneck_imagenet
from architectures.LeNet import LeNet5

import timm
from timm.models import create_model
import torch.nn as nn


def load_architecture(args,):

    if args.arch == 'resnet50':

        model = timm.create_model('resnet50', pretrained=False)
        if args.dataset == 'CIFAR10':
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes
        # model = ResNet_cifar10(Bottleneck_cifar10, [3, 4, 6, 3] )
        # model.to('cuda')

        # target_layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
        #         model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
        #         model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
        #         model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]
        
    # elif args.arch == 'LeNet5':
    #     model = LeNet5()
    #     model.to('cuda')
    #     target_layers = [model.conv1.weight, model.conv2.weight, model.fc1.weight, model.fc2.weight,model.fc3.weight] 

    # elif args.arch == 'resnet50' and args.dataset in ['Imagenet1k' , 'Imagenette']:
    #     model = ResNet_imagenet(Bottleneck_imagenet, [3, 4, 6, 3], )
    #     state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
    #     model.load_state_dict(state_dict)
    #     model.to('cuda')

    #     target_layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
    #             model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
    #             model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
    #             model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]
        
    # elif args.arch == 'ViTs':

    #     model = create_model('vit_small_patch16_224', pretrained=True)
        # target_layers = ["qkv", "proj"]

    elif args.arch == 'convnext':

        model = timm.models.convnext.convnext_tiny(pretrained=False)



        if args.dataset == 'CIFAR10':
            num_features = model.head.fc.in_features
            model.head.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes

    return model


def load_statedict(args,):

    if args.arch == 'resnet50' and args.pre_trained_data in ['CIFAR10', 'CIFAR10s']:
        
        state_dict = torch.load('./state_dicts/resnet50_cifar10.pt')

    if args.arch == 'resnet50' and args.pre_trained_data == 'Imagenet1k':

        state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
        
    
    elif args.arch == 'resnet50' and args.dataset in ['Imagenet1k' , 'Imagenette']:
        
        state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')

    elif args.arch == 'LeNet5':
        print('no state dict at the moment')


    return state_dict