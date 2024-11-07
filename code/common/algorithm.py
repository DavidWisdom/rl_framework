from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil, floor
from typing import List, Tuple

from torch.nn import ModuleDict

from code.common.config import Config


class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        """public concat"""

        concat_dim = (
            self.global_feature_dim
        )
        fc_concat_dim_list = [concat_dim, 16]
        self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp", non_linearity_last=True)

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        """output label"""
        self.label_mlp = ModuleDict(
            {
                "label{0}_mlp".format(label_index): MLP(
                    [self.lstm_unit_size, self.label_size_list[label_index]],
                    "label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )

        """output value"""
        self.value_mlp = MLP([self.lstm_unit_size, 1], "value_mlp")

    def forward(self, data_list):
        _, data_list = data_list
        feature_vec, legal_action, lstm_initial_state = data_list
        result_list = []


        # public concat
        fc_public_result = self.concat_mlp(concat_result)
        reshape_fc_public_result = fc_public_result.reshape(
            -1, self.lstm_time_steps, self.lstm_unit_size
        )

        # public lstm
        lstm_initial_state_in = [
            lstm_initial_state[0].unsqueeze(0),
            lstm_initial_state[1].unsqueeze(0),
        ]
        lstm_outputs, state = self.lstm(reshape_fc_public_result, lstm_initial_state_in)

        lstm_outputs = torch.cat(
            [lstm_outputs[:, idx, :] for idx in range(lstm_outputs.size(1))], dim=1
        )
        self.lstm_cell_output = state[1]
        self.lstm_hidden_output = state[0]
        reshape_lstm_outputs_result = lstm_outputs.reshape(-1, self.lstm_unit_size)

        # output label
        for label_index, label_dim in enumerate(self.label_size_list):
            label_mlp_out = self.label_mlp["label{0}_mlp".format(label_index)](
                reshape_lstm_outputs_result
            )
            result_list.append(label_mlp_out)

        lstm_tar_embed_result = self.lstm_tar_embed_mlp(reshape_lstm_outputs_result)

        tar_embedding = torch.stack(tar_embed_list, dim=1)

        ulti_tar_embedding = self.target_embed_mlp(tar_embedding)
        reshape_label_result = lstm_tar_embed_result.reshape(
            -1, self.target_embed_dim, 1
        )

        label_result = torch.matmul(ulti_tar_embedding, reshape_label_result)
        target_output_dim = int(np.prod(label_result.shape[1:]))

        reshape_label_result = label_result.reshape(-1, target_output_dim)
        result_list.append(reshape_label_result)

        # output value
        value_result = self.value_mlp(reshape_lstm_outputs_result)
        result_list.append(value_result)
        return result_list

    def compute_loss(self, data_list, rst_list):
        data_list, _ = data_list
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3: 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index]
                .reshape(-1, self.data_split_shape[3 + shape_index])
                .long()
            )

        old_label_probability_list = data_list[
                                     3 + len(self.label_size_list): 3 + 2 * len(self.label_size_list)
                                     ]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[
                shape_index
            ].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[
                          3 + 2 * len(self.label_size_list): 3 + 3 * len(self.label_size_list)
                          ]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(
            feature_legal_action_shape
        )

        legal_action_flag_list = torch.split(
            feature_legal_action, self.label_size_list, dim=1
        )

        # loss of value net
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(
            torch.square(reward - fc2_value_result_squeezed), dim=0
        )
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        # for entropy loss calculate
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []

        epsilon = 1e-5  # 0.00001

        # policy loss: ppo clip loss
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(
                    label_list[task_index].long(), self.label_size_list[task_index]
                )

                legal_action_flag_list_max_mask = (
                                                          1 - legal_action_flag_list[task_index]
                                                  ) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_logits_subtract_max_list.append(label_logits_subtract_max)

                label_exp_logits = (
                        legal_action_flag_list[task_index]
                        * torch.exp(label_logits_subtract_max)
                        + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_sum_exp_logits_list.append(label_sum_exp_logits)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (
                        one_hot_actions * old_label_probability_list[task_index] + epsilon
                ).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = (
                        ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                        * advantage
                )
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * 1
                ) / torch.maximum(
                    torch.sum((weight_list[task_index].float()) * 1), torch.tensor(1.0)
                )

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(
                        label_probability_list[current_entropy_loss_index] + epsilon
                    ),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * 1)
                ) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * 1), torch.tensor(1.0)
                )  # add - because need to minize

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

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
    model = Algorithm()

    pass