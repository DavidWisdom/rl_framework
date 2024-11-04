import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()

        pass

    def compute_loss(self, data_list, rst_list):
        self.value_cost = 0
        self.policy_cost = 0
        self.entropy_cost = 0
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

def make_conv_layer():
    pass

if __name__ == '__main__':
    pass