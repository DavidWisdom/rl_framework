from env.env import Env


class CliffWalking(Env):
    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1):
        self.is_turn = False
        self.eval_mode = eval_mode
        self.predict_frequency = predict_frequency

    def get_random_action(self, info):
        raise NotImplementedError("build model: not implemented")

    def obs_space(self):
        pass

    def action_space(self):
        pass

    def _state2ret(self, state):
        pass

    def step(self, actions):
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def reset(self, eval_mode=False):
        raise NotImplementedError("build model: not implemented")

    def close_game(self):
        raise NotImplementedError("build model: not implemented")