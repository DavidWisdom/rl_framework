class RLDataInfo:
    def __init__(self):
        self.frame_no = -1
        self.feature = b""
        self.advantage = 0
        self.is_game_over = 0
        # rl_data_info.frame_no = frame_no

        # rl_data_info.legal_action = legal_action.reshape([-1])
        self.reward = 0
        # rl_data_info.reward = reward
        self.value = 0
        # rl_data_info.value = value
        self.prob = None
        # rl_data_info.prob = prob
        self.action = 0
        # rl_data_info.action = action
        self.sub_action = None
        # rl_data_info.sub_action = sub_action[action[0]]
        self.is_train = False
        # rl_data_info.is_train = False if action[0] < 0 else is_train