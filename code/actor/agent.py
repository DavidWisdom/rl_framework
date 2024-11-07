import numpy as np
import time
import random
from code.common.model_pool_apis import ModelPoolAPIs
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

    def reset(self, agent_type=None, model_path=None):
        # reset lstm input
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

        if agent_type is not None:
            if self.keep_latest:
                self.agent_type = "network"
            else:
                self.agent_type = agent_type

        # for test without model pool
        if self.single_test:
            self.is_latest_model = True
        else:
            if model_path is None:
                while True:
                    try:
                        if self.keep_latest:
                            self._get_latest_model()
                        else:
                            self._get_random_model()
                        self.last_model_path = None
                        return
                    except Exception as e:  # pylint: disable=broad-except
                        time.sleep(1)
                        raise
            else:
                if model_path != self.last_model_path:
                    self._predictor.load_model(model_path)
                    self.last_model_path = model_path


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
        if self.agent_type in ["common_ai", "random"]:
            self.is_latest_model = False
            self.model_version = ""
            return True

        self._update_model_list()
        rand_float = float(random.uniform(0, _G_RAND_MAX)) / float(_G_RAND_MAX)
        if rand_float <= _G_MODEL_UPDATE_RATIO:
            midx = len(self.model_list) - 1
            self.is_latest_model = True
        else:
            midx = int(random.random() * len(self.model_list))
            if midx == len(self.model_list):
                midx = len(self.model_list) - 1
            self.is_latest_model = False
        return self._load_model(self.model_list[midx])

    def _get_latest_model(self):
        self._update_model_list()
        self.is_latest_model = True
        return self._load_model(self.model_list[-1])

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

    def _update_legal_action(self, original_la, actions):
        # TODO:
        return original_la

    def _sample_process(self, state_dict, pred_ret):
        is_train = False
        req_pb = state_dict["req_pb"]
        frame_no = req_pb["frame_no"]
        feature_vec, reward, sub_action_mask = (
            state_dict["observation"],
            state_dict["reward"],
            state_dict["sub_action_mask"],
        )
        done = False
        prob, value, action, _ = pred_ret
        legal_action = self._update_legal_action(state_dict["legal_action"], action)
        keys = (
            "frame_no",
            "vec_feature",
            "legal_action",
            "action",
            "reward",
            "value",
            "prob",
            "sub_action",
            "lstm_cell",
            "lstm_hidden",
            "done",
            "is_train",
        )
        values = (
            frame_no,
            feature_vec,
            legal_action,
            action,
            reward[-1],
            value,
            prob,
            sub_action_mask,
            self.lstm_cell,
            self.lstm_hidden,
            done,
            is_train,
        )
        sample = dict(zip(keys, values))
        self.last_sample = sample

    def _predict_process(self, feature, legal_action):
        input_list = []
        input_list.append(np.array(feature))
        input_list.append(np.array(legal_action))
        input_list.append(self.lstm_cell)
        input_list.append(self.lstm_hidden)

        np_output = self._predictor.inference(input_list)

        logits, value, self.lstm_cell, self.lstm_hidden = np_output[:4]

        prob, action, d_action = self._sample_masked_action(logits, legal_action)

        return prob, value, action, d_action

    def _sample_masked_action(self, logits, legal_action):
        prob_list = []
        action_list = []
        d_action_list = []

        label_split_size = [
            sum(self.label_size_list[: index + 1])
            for index in range(len(self.label_size_list))
        ]
        legal_actions = np.split(legal_action, label_split_size)
        logits_split = np.split(logits[0], label_split_size)
        for index in range(0, len(self.label_size_list)):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

    def set_lstm_info(self, lstm_info):
        self.lstm_hidden, self.lstm_cell = lstm_info

    def get_lstm_info(self):
        return (self.lstm_hidden, self.lstm_cell)