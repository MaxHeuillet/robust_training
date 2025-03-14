import torch

import timm
import torch.nn as nn
import os
import open_clip

# from utils import get_state_dict_dir
from architectures.clip_wrapper import CLIPConvNeXtClassifier


def load_architecture(config, N):
    backbone = config.backbone
    statedict_dir = os.path.abspath(os.path.expanduser(config.statedicts_path)) #get_state_dict_dir(config)

    # Determine if it's an OpenCLIP model
    openclip_models = {
        'CLIP-convnext_base_w-laion2B-s13B-b82K': 'convnext_base_w',
        'CLIP-convnext_base_w-laion_aesthetic-s13B-b82K': 'convnext_base_w',
    }

    if backbone in openclip_models:
        

        model_name = openclip_models[backbone]
        print('Loading OpenCLIP model:', model_name)

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None  # you're loading your own weights
        )

        checkpoint_path = os.path.join(statedict_dir, f'{backbone}.pt')
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']  # Handle wrapped checkpoints

        model.load_state_dict(state_dict, strict=False)

    else:
        # Use timm model
        equivalencies = {
            'robust_convnext_base': 'convnext_base',
            'robust_convnext_tiny': 'convnext_tiny',
            'robust_resnet50': 'resnet50',
            'robust_deit_small_patch16_224': 'deit_small_patch16_224',
            'robust_vit_base_patch16_224': 'vit_base_patch16_224',
        }

        model_name = equivalencies.get(backbone, backbone)
        print('Loading timm model:', model_name)

        model = timm.create_model(model_name, pretrained=False)

        checkpoint_path = os.path.join(statedict_dir, f'{backbone}.pt')
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        model.load_state_dict(state_dict)

    # Replace classification head if needed
    model = change_head(backbone, model, N)

    return model


# def load_architecture(hp_opt,config, N, ):

#     backbone = config.backbone
#     statedict_dir = get_state_dict_dir(hp_opt,config)

#     equivalencies = { 'robust_convnext_base':'convnext_base',
#                       'robust_convnext_tiny':'convnext_tiny',
#                       'robust_resnet50':'resnet50',
#                       'robust_deit_small_patch16_224': 'deit_small_patch16_224',
#                       'robust_vit_base_patch16_224': 'vit_base_patch16_224',   }
    
#     if equivalencies.get(backbone):
#         model_name = equivalencies[backbone]
#     else:
#         model_name = backbone
#     print('model_name', model_name)
#     model = timm.create_model(model_name, pretrained=False)     
#     state_dict = torch.load( statedict_dir + '/{}.pt'.format(backbone) , weights_only=True, map_location='cpu')
#     model.load_state_dict(state_dict)

#     model = change_head(backbone, model, N)
    
#     return model


def change_head(backbone, model, N):

    if 'CLIP-convnext' in backbone:
        model = CLIPConvNeXtClassifier(model, num_classes=N)

    elif "convnext" in backbone or 'swinv2' in backbone: 
        num_features = model.head.fc.in_features
        model.head.fc = nn.Linear(num_features, N)

    elif "resnet" in backbone:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, N)

    elif "deit" in backbone:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, N)

    elif "vit" in backbone: 

        if isinstance(model.head, nn.Identity):
            num_features = 768
            model.head = nn.Linear(num_features, N)

        num_features = model.head.in_features
        model.head = nn.Linear(num_features, N)


    return model





# def custom_forward(self, x_natural, x_adv=None):
#     # Implement your custom forward logic
#     if x_adv is not None:
#         logits_nat = self.forward_features(x_natural)  # Assuming forward_features is the main forward logic
#         logits_adv = self.forward_features(x_adv)
#         logits_nat = self.head(logits_nat)  # Final classification head
#         logits_adv = self.head(logits_adv)
#         return logits_nat, logits_adv
#     else:
#         logits_nat = self.forward_features(x_natural)
#         logits_nat = self.head(logits_nat)
#         return logits_nat

# def custom_forward(model, x_natural, x_adv=None):

#     # Directly access the model's forward method without using model(x)
#     def get_logits(x):
#         return model.forward(x)
    
#     logits_nat = get_logits(x_natural)  # Get natural input logits
    
#     logits_adv = None
#     if x_adv is not None:
#         logits_adv = get_logits(x_adv)  # Get adversarial input logits if provided
    
#     return logits_nat, logits_adv
  
# class ImageNormalizer(nn.Module):
#     def __init__(self, mean: Tuple[float, float, float],
#         std: Tuple[float, float, float],
#         persistent: bool = True) -> None:
#         super(ImageNormalizer, self).__init__()

#         self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1),
#             persistent=persistent)
#         self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1),
#             persistent=persistent)

#     def forward(self, input: Tensor) -> Tensor:
#         return (input - self.mean) / self.std

# def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
#     std: Tuple[float, float, float]) -> nn.Module:
#     layers = OrderedDict([
#         ('normalize', ImageNormalizer(mean, std)),
#         ('model', model)
#     ])
#     return nn.Sequential(layers)

# IMAGENET_MEAN = [c * 1. for c in (0.485, 0.456, 0.406)] #[np.array([0., 0., 0.]), np.array([0.485, 0.456, 0.406])][-1] * 255
# IMAGENET_STD = [c * 1. for c in (0.229, 0.224, 0.225)] #[np.array([1., 1., 1.]), np.array([0.229, 0.224, 0.225])][-1] * 255



# def load_architecture(args,):

#     if args.arch == 'resnet50':

#         # model = timm.create_model('resnet50', pretrained=False)
#         model = ResNet_imagenet(Bottleneck_imagenet, [3, 4, 6, 3], )
        
        
#         if args.pre_trained == 'non_robust':
#             # state_dict = torch.load('./state_dicts/timm_resnet50_imagenet1k.pt')
#             state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
#             model.load_state_dict(state_dict)

#         if args.dataset == 'CIFAR10':
#             num_features = model.fc.in_features
#             model.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes
#         # model = ResNet_cifar10(Bottleneck_cifar10, [3, 4, 6, 3] )
#         # model.to('cuda')

#         # target_layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
#         #         model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
#         #         model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
#         #         model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]
        
#     elif args.arch == 'LeNet5':
#         model = LeNet5()
#         model.to('cuda')
#         # target_layers = [model.conv1.weight, model.conv2.weight, model.fc1.weight, model.fc2.weight,model.fc3.weight] 

#     # elif args.arch == 'resnet50' and args.dataset in ['Imagenet1k' , 'Imagenette']:
#     #     model = ResNet_imagenet(Bottleneck_imagenet, [3, 4, 6, 3], )
#     #     state_dict = torch.load('./state_dicts/resnet50_imagenet1k.pt')
#     #     model.load_state_dict(state_dict)
#     #     model.to('cuda')

#     #     target_layers = [ model.conv1, model.layer1[0].conv1, model.layer1[0].conv2, model.layer1[0].conv3,
#     #             model.layer2[0].conv1, model.layer2[0].conv2, model.layer2[0].conv3,
#     #             model.layer3[0].conv1, model.layer3[0].conv2, model.layer3[0].conv3,
#     #             model.layer4[0].conv1, model.layer4[0].conv2, model.layer4[0].conv3, model.fc ]
        
# #     elif args.arch == 'vitsmall':

# #         if args.dataset == 'CIFAR10':
# #             model = timm.create_model('vit_small_patch16_224', pretrained=False, img_size=32, patch_size=4)
# # )
# #         if args.pre_trained:
# #             state_dict = torch.load('./state_dicts/timm_vit_small_patch16_224_imagenet1k.pt')
# #             model.load_state_dict(state_dict)

#         # target_layers = ["qkv", "proj"]

#     elif args.arch == 'convnext':

#         model = timm.models.convnext.convnext_tiny(pretrained=False)
#         # Replace the model's forward method with your custom one
        
#         if args.pre_trained == 'imagenet1k_non_robust':
#             state_dict = torch.load('./state_dicts/timm_convnext_imagenet1k.pt')
#             model.load_state_dict(state_dict)
        
#         elif args.pre_trained == 'imagenet21k_non_robust':
#             num_features = model.head.fc.in_features
#             model.head.fc = nn.Linear(num_features, 21841) 
#             state_dict = torch.load('./state_dicts/convnext_imagenet21k.pt') 
#             model.load_state_dict(state_dict)
        
#         elif args.pre_trained == 'imagenet1k_robust': # code from: https://github.com/nmndeep/revisiting-at/blob/main/utils_architecture.py
#             model = normalize_model(model, IMAGENET_MEAN, IMAGENET_STD)
#             ckpt = torch.load('./state_dicts/weights_convnext_t.pt', map_location='cpu', weights_only=False)

#             ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
#             try:
#                 model.load_state_dict(ckpt)
#                 print('standard loading')

#             except:
#                 try:
#                     ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
#                     model.load_state_dict(ckpt)
#                     print('loaded from clean model')
#                 except:
#                     ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
#                     # ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
#                     model.load_state_dict(ckpt)
#                     print('loaded')

#             if isinstance(model, nn.Sequential) and 'normalize' in model._modules: # remove normalization layer
#                 # Rebuild the sequential model without the 'normalize' layer
#                 model = model._modules['model']

#         else:
#             print('no pre-trained model specified')

#         model.forward = types.MethodType(custom_forward, model)

#         if args.dataset in [ 'CIFAR10', 'EuroSAT' ]:
#             num_features = model.head.fc.in_features
#             model.head.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes
#         elif args.dataset in ['CIFAR100', 'Aircraft']:
#             num_features = model.head.fc.in_features
#             model.head.fc = nn.Linear(num_features, 100)  # CIFAR-10 has 10 classes


#     elif args.arch == 'wideresnet-28-10': 

#         depth = 28
#         widen = 10
#         act_fn = 'swish'  # Assuming 'swish' is the desired activation function
#         num_classes = 200
#         model = wideresnet(depth, widen, act_fn, num_classes)

#         if args.pre_trained == 'tinyimagenet_semisup_robust':

#             ckpt = torch.load('./state_dicts/tiny_linf_wrn28-10.pt')
#             ckpt = {k.replace('module.0.', ''): v for k, v in ckpt['model_state_dict'].items()}
#             model.load_state_dict(ckpt)

#         if args.dataset in [ 'CIFAR10', 'EuroSAT' ]:
#             num_features = model.logits.in_features
#             model.logits = nn.Linear(num_features, 10)
#         elif args.dataset in ['CIFAR100', 'Aircraft']:
#             num_features = model.logits.in_features
#             model.logits = nn.Linear(num_features, 100)


#     return model


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