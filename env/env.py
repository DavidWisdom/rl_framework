from abc import abstractmethod

class Env:
    def __init__(self, eval_mode=False, predict_frequency=1):
        raise NotImplementedError("build model: not implemented")

    @abstractmethod
    def get_random_action(self, info):
        raise NotImplementedError("build model: not implemented")

    @abstractmethod
    def step(self, actions):
        raise NotImplementedError("build model: not implemented")

    @abstractmethod
    def reset(self, eval_mode=False):
        raise NotImplementedError("build model: not implemented")

    @abstractmethod
    def close_game(self):
        raise NotImplementedError("build model: not implemented")

