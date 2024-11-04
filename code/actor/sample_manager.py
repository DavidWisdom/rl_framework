import collections

import numpy as np
from code.actor.rl_data_info import RLDataInfo


class SampleManager:
    def __init__(self, num_agents, game_id=None, single_test=False, gamma=0.995, lamda=0.95):
        self.agents = None
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self._replay_buffer = [[] for _ in range(num_agents)]
        self.gamma = gamma
        self.lamda = lamda

    def reset(self, agents, game_id):
        self._game_id = game_id
        self.agents = agents
        self.num_agents = len(agents)
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self._replay_buffer = [[] for _ in range(self.num_agents)]

    def save_sample(
        self,
        frame_no,
        vec_feature,
        legal_action,
        action,
        reward,
        value,
        prob,
        sub_action,
        lstm_cell,
        lstm_hidden,
        agent_id,
        is_train=True,
        game_id=None,
    ):
        reward = self._clip_reward(reward)
        pass

    def save_last_sample(self):
        pass

    def send_samples(self):
        self._calc_reward()
        self._format_data()
        self._send_game_data()

    def _calc_reward(self):
        pass

    def _format_data(self):
        pass

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    def _send_game_data(self):
        pass