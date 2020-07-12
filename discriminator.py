from torchvision.models.densenet import DenseNet
from torch import nn


def convert_d(net):
    for name in net._modules.keys():
        if "norm" in name:
            net._modules[name] = nn.InstanceNorm2d(net._modules[name].num_features, affine=True)
        elif "relu" in name:
            net._modules[name] = nn.LeakyReLU(0.2)
        else:
            convert_d(net._modules[name])


def create_d(in_channels):
    d = DenseNet(growth_rate=16, block_config=(3,3,3,3), num_classes=1, drop_rate=0.0)
    d.features[0] = nn.Conv2d(in_channels * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    convert_d(d)
    return d
