from env.env import Env


class Card(Env):
    PLAYER_NUM = 2
    LABEL_SIZE_LIST = [
        9,
        3,
        16, 16,
        16,
        16,
        16, 16, 16, 16,
        16, 16, 16, 16
    ]
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1, player_num=2):

        pass

    def get_random_action(self, info):
        raise NotImplementedError("build model: not implemented")

    def step(self, actions):
        raise NotImplementedError("build model: not implemented")

    def reset(self, eval_mode=False):
        raise NotImplementedError("build model: not implemented")

    def close_game(self):
        raise NotImplementedError("build model: not implemented")


