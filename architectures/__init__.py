from .LeNet import LeNet5
from .resnet_cifar10 import ResNet_cifar10,Bottleneck_cifar10
from .resnet_imagenet import ResNet_imagenet,Bottleneck_imagenet
from .loaders import load_architecture, change_head #, load_statedict
from .lora import add_lora, set_lora_gradients
from .wideresnetswish import wideresnet