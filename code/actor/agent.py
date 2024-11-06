import numpy as np
import time

from code.actor.model_pool_apis import ModelPoolAPIs
from predictor import TorchPredictor as LocalPredictor


_G_RAND_MAX = 10000
_G_MODEL_UPDATE_RATIO = 0.8


class Agent:
    def __init__(
        self,
        model,
        model_pool_addr,
        config,
        keep_latest=False,
        single_test=False,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.single_test = single_test

        self._predictor = LocalPredictor(self.model)

        if single_test or not model_pool_addr:
            self._model_pool_api = None
        else:
            self._model_pool_api = ModelPoolAPIs(model_pool_addr)

        self.model_version = ""
        self.is_latest_model: bool = False
        self.keep_latest = keep_latest
        self.model_list = []

        self.lstm_unit_size = self.config.LSTM_UNIT_SIZE

        self.lstm_hidden = None
        self.lstm_cell = None

        self.player_id = 0
        self.hero_camp = 0
        self.last_model_path = None
        self.label_size_list = self.config.LABEL_SIZE_LIST
        self.legal_action_size = self.config.LEGAL_ACTION_SIZE_LIST

        self.agent_type = "network"

    def _update_model_list(self):
        model_key_list = []
        while len(model_key_list) == 0:
            model_key_list = self._model_pool_api.pull_keys()
            if not model_key_list:
                time.sleep(1)
        self.model_list = model_key_list

    def _load_model(self, model_version):
        if model_version == self.model_version:
            return True
        model_path = self._model_pool_api.pull_model_path(model_version)
        ret = self._predictor.load_model(model_path)
        if ret:
            # if failed, do not update model_version
            self.model_version = model_version
        return ret

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