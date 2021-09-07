import torch
from torch.nn import ReLU


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def powerlaw(x, eps=1e-6):
    x = x + eps
    return x.abs().sqrt().mul(x.sign())


def alexnet(net):
    return list(net.features.children())[:-1]


def vggnet(net):
    return list(net.features.children())[:-1]


def resnet(net):
    return list(net.children())[:-2]


def densenet(net):
    features = list(net.features.children())
    features.append(ReLU(inplace=True))
    return features


def squeezenet(net):
    return list(net.features.children())


def generate_coordinates(h, w):
    """
    Generate coordinates

    Args:
        h: height
        w: width

    Returns: [h*w, 2] FloatTensor
    """

    x = torch.floor((torch.arange(0, w * h) / w).float())
    y = torch.arange(0, w).repeat(h).float()

    coord = torch.stack([x, y], dim=1)
    return coord

