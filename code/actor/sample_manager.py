import collections

import numpy as np

from code.actor import rl_data_info
from code.actor.mem_pool_api import MemPoolAPIs
from code.actor.rl_data_info import RLDataInfo


class SampleManager:
    def __init__(
        self,
        mem_pool_addr,
        mem_pool_type,
        num_agents,
        game_id=None,
        single_test=False,
        data_shapes=None,
        lstm_time_steps=16,
        gamma=0.995,
        lamda=0.95,
    ):
        self.single_test = single_test
        # connect to mem pool
        mem_pool_addr = mem_pool_addr
        ip, port = mem_pool_addr.split(":")
        self.m_mem_pool_ip = ip
        self.m_mem_pool_port = port
        self._data_shapes = data_shapes
        self._LSTM_FRAME = lstm_time_steps

        if not self.single_test:
            self._mem_pool_api = MemPoolAPIs(
                self.m_mem_pool_ip, self.m_mem_pool_port, socket_type=mem_pool_type
            )

        self.m_game_id = game_id
        self.m_task_id, self.m_task_uuid = 0, "default_task_uuid"
        self.num_agents = num_agents
        self.agents = None
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

        # load config from config file
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
        rl_data_info = RLDataInfo()
        rl_data_info.frame_no = frame_no
        rl_data_info.legal_action = legal_action.reshape([-1])
        rl_data_info.reward = reward
        rl_data_info.value = value
        rl_data_info.prob = prob
        rl_data_info.action = action
        rl_data_info.sub_action = sub_action[action[0]]
        rl_data_info.is_train = False if action[0] < 0 else is_train

    def _add_extra_info(self, frame_no, sample):
        return sample.astype(np.float32).tobytes()

    def _send_game_data(self):
        pass

    def save_last_sample(self):
        pass

    def send_samples(self):
        self._calc_reward()
        self._format_data()
        self._send_game_data()

    def _calc_reward(self):
        """
        Calculate cumulated reward and advantage with GAE.
        reward_sum: used for value loss
        advantage: used for policy loss
        V(s) here is a approximation of target network
        """
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae, last_gae = 0.0, 0.0
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]
                # TD error
                delta = (
                    -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                )
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    def _format_data(self):
        pass

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    def _add_extra_info(self, frame_no, sample):
        return sample.astype(np.float32).tobytes()

    def _send_game_data(self):
        for i in range(self.num_agents):
            samples = []
            for sample in self.m_replay_buffer[i]:
                samples.append(self._add_extra_info(*sample))
            if (not self.single_test) and len(samples) > 0:
                self._mem_pool_api.push_samples(samples)