import numpy as np
from predictor import TorchPredictor as LocalPredictor

class Agent:
    def __init__(
        self,

    ):
        super().__init__()

        self.agent_type = "network"

    def _get_random_model(self):
        pass

    def _get_latest_model(self):
        pass

    def feature_post_process(self, state_dict):
        return state_dict

    def process(self, state_dict, battle=False):
        state_dict = self.feature_post_process(state_dict)
        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )
        pred_ret = self._predict_process(feature_vec, legal_action)
        _, _, action, d_action = pred_ret
        if battle:
            return d_action
        return action, d_action

    def _predict_process(self, feature, legal_action):
        pass

    def _legal_soft_max(self):
        pass

    def _legal_sample(self, probs, legal_action=None, use_max=False):

        pass

    def set_lstm_info(self, lstm_info):
        self.lstm_hidden, self.lstm_cell = lstm_info

    def get_lstm_info(self):
        return (self.lstm_hidden, self.lstm_cell)