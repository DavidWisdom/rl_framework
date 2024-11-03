import torch
import torch.nn as nn
import torch.nn.functional as F
class Algorithm(nn.Module):
    def __init__(self):
        pass

def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """
    :param in_features:
    :param out_features:
    :param use_bias:
    :return:
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    # initialize weight and bias
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer

class MLP(nn.Module):
    def __init__(self):
        pass
    def forward(self, data):
        pass

def make_conv_layer():
    pass

if __name__ == '__main__':
    pass