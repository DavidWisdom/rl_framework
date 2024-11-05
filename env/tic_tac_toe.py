from env.env import Env


class TicTacToe(Env):
    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1):
        raise NotImplementedError("build model: not implemented")

    def get_random_action(self, info):
        raise NotImplementedError("build model: not implemented")

    def step(self, actions):
        raise NotImplementedError("build model: not implemented")

    def reset(self, eval_mode=False):
        raise NotImplementedError("build model: not implemented")

    def close_game(self):
        raise NotImplementedError("build model: not implemented")