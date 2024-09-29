import torch
from architectures.resnet_cifar10 import ResNet_cifar10, Bottleneck_cifar10
from architectures.resnet_imagenet import ResNet_imagenet, Bottleneck_imagenet
from architectures.LeNet import LeNet5

import timm
from timm.models import create_model
import torch.nn as nn

import types


def custom_forward(self, x_natural, x_adv=None):
    # Implement your custom forward logic
    if x_adv is not None:
        logits_nat = self.forward_features(x_natural)  # Assuming forward_features is the main forward logic
        logits_adv = self.forward_features(x_adv)
        logits_nat = self.head(logits_nat)  # Final classification head
        logits_adv = self.head(logits_adv)
        return logits_nat, logits_adv
    else:
        logits_nat = self.forward_features(x_natural)
        logits_nat = self.head(logits_nat)
        return logits_nat





def load_architecture(args,):

    if args.arch == 'resnet50':

        # model = timm.create_model('resnet50', pretrained=False)
        model = ResNet_imagenet(Bottleneck_imagenet, [3, 4, 6, 3], )
        
        
        if args.pre_trained:
            # state_dict = torch.load('./state_dicts/timm_resnet50_imagenet1k.pt')
            state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
            model.load_state_dict(state_dict)

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
        # Replace the model's forward method with your custom one
        model.forward = types.MethodType(custom_forward, model)

        if args.pre_trained:
            state_dict = torch.load('./state_dicts/timm_convnext_imagenet1k.pt')
            model.load_state_dict(state_dict)

        if args.dataset == 'CIFAR10':
            num_features = model.head.fc.in_features
            model.head.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes

    return model


# def load_statedict(args,):

#     # if args.arch == 'resnet50' and args.pre_trained_data in ['CIFAR10']:
        
#     #     state_dict = torch.load('./state_dicts/resnet50_cifar10.pt')

#     if args.arch == 'resnet50':

#         state_dict = torch.load('./state_dicts/timm_resnet50_imagenet1k.pt')
        
#     elif args.arch == 'convnext':

#         state_dict = torch.load('./state_dicts/timm_convnext_imagenet1k.pt')

#     elif args.arch == 'LeNet5':
#         print('no state dict at the moment')


#     return state_dict