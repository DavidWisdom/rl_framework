from env.env import Env
class Deck:
    pass

class Card(Env):
    PLAYER_NUM = 2
    LABEL_SIZE_LIST = [
        15,
        # 0: no action
        # 1: 单子
        # 2: 对子
        # 3: 三不带
        # 4: 三带一
        # 5: 三带二
        # 6: 炸不带
        # 7: 炸带一
        # 8: 炸带二
        # 9: 单顺子
        # 10: 双顺子
        # 11: 三顺子
        # 12: 飞机不带
        # 13: 飞机带一
        # 14: 飞机带二
        16, 16,
        16,
        16,
        16, 16, 16, 16,
        16, 16, 16, 16
    ]
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    # obs dim: 170
    # 全局信息 全局总手牌数 dim: 1 全局剩余手牌数 dim: 1 全局模式 dim: 3 全局展示手牌: 15 全局已出手牌 dim: 15 全局剩余手牌 dim: 15
    # 我方信息 我方初始手牌数 dim: 1 我方剩余手牌数 dim: 1 我方阵营 dim: 3 我方已出手牌 dim: 15 我方剩余手牌 dim: 15 我方上一帧动作手牌 dim: 15
    # 下家信息 下家初始手牌数 dim: 1 下家剩余手牌数 dim: 1 下家阵营 dim: 3 下家已出手牌 dim: 15 下方上一帧动作手牌 dim: 15
    # 上家信息 上家初始手牌数 dim: 1 上家剩余手牌数 dim: 1 上家阵营 dim: 3 上家已出手牌 dim: 15 上方上一帧动作手牌 dim: 15

    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1, player_num=2, player_card_num=17):
        super().__init__()
        self.is_turn = True
        self.eval_mode = eval_mode
        assert predict_frequency == 1
        self.predict_frequency = 1
        self._init_game(player_num, player_card_num)

    def get_random_action(self, info):
        raise NotImplementedError("build model: not implemented")

    def step(self, actions, render=False):

        raise NotImplementedError("build model: not implemented")

    def reset(self, eval_mode=False, player_num=2, player_card_num=17):
        self._init_game(player_num, player_card_num)

    def close_game(self):
        raise NotImplementedError("build model: not implemented")

    def _init_game(self, player_num=2, player_card_num=17):
        self.turn_no = 0
        self.PLAYER_NUM = player_num
        self.player_card_num = player_card_num
        self.player_init_cards = [0, 0, 0]
        self.player_curr_cards = [0, 0, 0]
        self.player_obs = [[0] * 16, [0] * 16, [0] * 16]
        self.last_action = [[0] * 16, [0] * 16, [0] * 16]

    def _render(self):
        pass
