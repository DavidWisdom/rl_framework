from abc import abstractmethod

class Env(object):
    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1):
        self.is_turn = is_turn
        self.eval_mode = eval_mode
        self.predict_frequency = predict_frequency
        self.turn_no = -1

    def get_turn_no(self):
        return self.turn_no

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

