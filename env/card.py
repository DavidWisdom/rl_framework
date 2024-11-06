import random

from env import Env


class Card(Env):
    PLAYER_NUM = 3
    LABEL_SIZE_LIST = [
        16,
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
        # 15: 抢地主
        16, 16,
        16, # 4张（包含火箭）
        16, # 3张
        16, 16, 16, 16, 16, # 2张（不包含火箭）
        16, 16, 16, 16, 16, # 1张
    ]
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    # obs dim: 172
    # 全局信息 全局初始手牌数 dim: 1 全局剩余手牌数 dim: 1 全局模式 dim: 3 全局阶段 dim: 2 全局展示手牌: 15 全局已出手牌 dim: 15 全局剩余手牌 dim: 15
    # 我方信息 我方初始手牌数 dim: 1 我方剩余手牌数 dim: 1 我方阵营 dim: 3 我方已出手牌 dim: 15 我方剩余手牌 dim: 15 我方上一帧动作手牌 dim: 15
    # 下家信息 下家初始手牌数 dim: 1 下家剩余手牌数 dim: 1 下家阵营 dim: 3 下家已出手牌 dim: 15 下方上一帧动作手牌 dim: 15
    # 上家信息 上家初始手牌数 dim: 1 上家剩余手牌数 dim: 1 上家阵营 dim: 3 上家已出手牌 dim: 15 上方上一帧动作手牌 dim: 15

    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1, player_mode=None, player_card_num=17):
        super().__init__()
        if player_mode is None:
            player_mode = [1, 0, 0]
        self.is_turn = True
        self.eval_mode = eval_mode
        assert predict_frequency == 1
        self.predict_frequency = 1
        self._init_game(player_mode, player_card_num)

    def get_random_action(self, info):
        raise NotImplementedError("build model: not implemented")

    def step(self, actions, render=False):

        raise NotImplementedError("build model: not implemented")

    def reset(self, eval_mode=False, player_mode=None, player_card_num=17):
        if player_mode is None:
            player_mode = [1, 0, 0]
        self._init_game(player_mode, player_card_num)

    def close_game(self):
        raise NotImplementedError("build model: not implemented")

    def _init_game(self, player_mode=None, player_card_num=17):
        self.turn_no = 0
        self.player_mode = player_mode
        if self.player_mode[0] == 1:
            self.player_camp = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        else:
            self.player_camp = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        if self.player_mode[0] == 1:
            self.player_init_cards = [player_card_num, player_card_num, 0]
        else:
            self.player_init_cards = [player_card_num, player_card_num, player_card_num]
        self.player_curr_cards = self.player_init_cards
        self.cards = [4] * 15
        self.cards[13] = 1
        self.cards[14] = 1
        self.player_cards = [[0] * 15, [0] * 15, [0] * 15]
        self.player_used_cards = [[0] * 15, [0] * 15, [0] * 15]
        self.last_action = [[0] * 15, [0] * 15, [0] * 15]
        self.player_legal_action = [[0] * 16, [0] * 16, [0] * 16]
        self.winner = -1
        self._init_player_cards()
        self._render()

    def _init_player_cards(self):
        for i in range(self.PLAYER_NUM):  # 假设 PLAYER_NUM 是玩家的数量
            cards_num = self.player_init_cards[i]  # 每个玩家初始的牌数
            for j in range(cards_num):
                available_cards = [index for index, value in enumerate(self.cards) if value > 0]  # 非0数值的索引列表
                if available_cards:  # 检查是否有可用的牌
                    index = random.choice(available_cards)  # 随机选择一个非0数值的索引
                    self.player_cards[i][index] += 1  # 将牌的索引分配给玩家
                    self.cards[index] -= 1  # 将已分配的牌标记为0，表示已被拿走
                else:
                    break

    def _render(self):
        # 牌面值映射
        card_values = {
            1: 'A', 11: 'J', 12: 'Q', 13: 'K', 14: '小王', 15: '大王'
        }
        for i in range(3):
            if i == 2 and self.player_mode[0] == 1:
                break
            print(f"玩家 {i + 1} 的手牌:")
            for j in range(len(self.player_cards[i])):
                if self.player_cards[i][j] > 0:
                    for k in range(self.player_cards[i][j]):
                        # 检查牌面值是否在映射表中，如果不在则使用默认的数字表示
                        card_value = card_values.get(j + 1, str(j + 1))
                        print(f"{card_value}", end=" ")
            print()  # 打印空行以分隔不同玩家的手牌

    def _is_over(self, player_id):
        if self.winner == -1:
            return self.player_curr_cards[player_id] == 0
        return True


if __name__ == "__main__":
    env = Card()
