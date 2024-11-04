import torch
from code.common.algorithm import Algorithm
torch.set_num_threads(1)
torch.set_num_interop_threads(1)



class Model(Algorithm):
    def __init__(self):
        super().__init__()

    def format_data(self, data_list):
        feature, legal_action, init_lstm_cell, init_lstm_hidden = data_list
        lstm_initial_state = (init_lstm_hidden, init_lstm_cell)
        return data_list, [feature, legal_action, lstm_initial_state]

    def forward(self, data_list):
        format_list = self.format_data(data_list)
        result_list = super().forward(format_list)

        logits = torch.flatten(torch.cat(result_list[:-1], dim=1), start_dim=1)
        value = result_list[-1]
        return [logits, value, ] # TODO: