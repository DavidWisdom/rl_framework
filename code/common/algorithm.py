import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil, floor
from typing import List, Tuple

from code.common.config import Config


class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()
        self.var_beta = Config.BETA_START
        self.learning_rate = Config.INIT_LEARNING_RATE_START

        pass

    def compute_loss(self, data_list, rst_list):
        self.value_cost = 0
        self.policy_cost = torch.tensor(0.0)

        self.entropy_cost = torch.tensor(0.0)

        self.loss = (
            self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        )
        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

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
    def __init__(self, fc_feat_dim_list: List[int], name: str, non_linearity: nn.Module = nn.ReLU, non_linearity_last: bool = False):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) or non_linearity_last:
                self.fc_layers.add_module(
                    "{0}_non_linear{1}".format(name, i + 1), non_linearity()
                )

    def forward(self, data):
        return self.fc_layers(data)

def _compute_conv_out_shape(kernel_size: Tuple[int, int], padding: Tuple[int, int], input_shape: Tuple[int, int], stride: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    out_x = (
        floor((input_shape[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0])
        + 1
    )
    out_y = (
        floor((input_shape[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1])
        + 1
    )
    return (out_x, out_y)

def make_conv_layer(kernel_size: Tuple[int, int], in_channels: int, out_channels: int, padding: str, stride: Tuple[int, int] = (1, 1), input_shape = None):
    if isinstance(padding, str):
        assert padding in [
            "same",
            "valid",
        ], "Padding scheme must be either 'same' or 'valid'"
        if padding == "valid":
            padding = (0, 0)
        else:
            assert stride == 1 or (
                stride[0] == 1 and stride[1] == 1
            ), "Stride must be 1 when using 'same' as padding scheme"
            assert (
                kernel_size[0] % 2 and kernel_size[1] % 2
            ), "Currently, requiring kernel height and width to be odd for simplicity"
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    nn.init.orthogonal_(conv_layer.weight)
    nn.init.zeros_(conv_layer.bias)
    output_shape = None
    if input_shape:
        output_shape = _compute_conv_out_shape(
            kernel_size, padding, input_shape, stride
        )
    return conv_layer, output_shape

if __name__ == '__main__':
    pass