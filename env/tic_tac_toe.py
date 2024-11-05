import random
from env import Env
import numpy as np

class TicTacToe(Env):
    LABEL_SIZE_LIST = [11] # 0-10 0: no action
    PLAYER_NUM = 2
    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1):
        self.is_turn = True
        self.eval_mode = eval_mode
        assert predict_frequency == 1
        self.predict_frequency = 1
        self._init_game()

    def action_space(self):
        return self.LABEL_SIZE_LIST

    def get_random_action(self, info):
        legal_positions = [index - 1 for index in range(1, len(info[self.turn_no]["legal_action"])) if info[self.turn_no]["legal_action"][index] == 1]

        # 如果没有合法的位置，返回 None 或者抛出异常
        if not legal_positions:
            assert False
        # 从合法的位置中随机选择一个
        return [random.choice(legal_positions)]

    def step(self, actions, render=False):
        # action = [pos]
        obs = [self.state_dict[i]["observation"] for i in range(2)]
        reward = 0
        for i, act in enumerate(actions):
            if act:
                assert self.turn_no == i
                a = act[0]
                assert len(act) == 1 and 0 <= a <= 9
                self.turn_no = (self.turn_no + 1) % 2
                if 1 <= a <= 9:
                    assert self.legal_action[a + 1] == 1
                    self.state_dict[i]["observation"][(a - 1)] = 1
                    self.state_dict[self.turn_no]["observation"][(a - 1) + 9] = 1
                    self.legal_action[a + 1] = 0
                self.done[i] = self._is_over()
                if self.done[i]:
                    self.legal_action[1] = 1
                    for j in range(1, 10):
                        self.legal_action[j + 1] = 0
                self.state_dict[i]["legal_action"] = self.legal_action
                self.state_dict[self.turn_no]["legal_action"] = self.legal_action
        if render:
            self._render()

        return obs, reward, self.done, self.state_dict

    def reset(self, eval_mode=False):
        self.eval_mode = eval_mode
        self._init_game()
        return None, 0, False, self.state_dict

    def close_game(self):
        pass

    def _render(self):
        # 打印分隔线
        print("-----------------")

        # 打印棋盘状态
        # 棋盘的字符表示
        player_mark = "X"
        enemy_mark = "O"
        empty_mark = " "

        # 打印棋盘顶部
        print("    0   1   2")
        print("  +---+---+---+")

        # 打印棋盘的每一行
        for row in range(3):
            print(f"{row} | ", end="")
            for col in range(3):
                # 根据 one-hot 编码确定标记
                mark = empty_mark
                if self.state_dict[0]["observation"][row * 3 + col] == 1:
                    mark = player_mark
                if self.state_dict[0]["observation"][row * 3 + col + 9] == 1:
                    mark = enemy_mark

                # 打印当前位置的标记
                print(f"{mark} | ", end="")
            print()
            print("  +---+---+---+")

        # 打印棋盘底部
        print("    0   1   2")
        print("-----------------")

    def _init_game(self):
        self.state_dict = [{} for _ in range(2)]
        self.turn_no = 0
        self.done = [False for _ in range(2)]
        self.legal_action = [1] * 11
        self.legal_action[1] = 0
        for i in range(2):
            self.state_dict[i]["observation"] = [0] * 18
            self.state_dict[i]["legal_action"] = self.legal_action
            self.state_dict[i]["reward"] = 0

    def _is_over(self):
        for i in range(3):
            # 水平连线
            if all(self.state_dict[0]["observation"][i * 3 + j] == 1 for j in range(3)) or \
                    all(self.state_dict[0]["observation"][i * 3 + j + 9] == 1 for j in range(3)):
                return True
            # 垂直连线
            if all(self.state_dict[0]["observation"][j * 3 + i] == 1 for j in range(3)) or \
                    all(self.state_dict[0]["observation"][j * 3 + i + 9] == 1 for j in range(3)):
                return True

            # 检查对角线连线
        if all(self.state_dict[0]["observation"][i * 3 + i] == 1 for i in range(3)) or \
                all(self.state_dict[0]["observation"][i * 3 + i + 9] == 1 for i in range(3)):
            return True
        if all(self.state_dict[0]["observation"][i * 3 + (2 - i)] == 1 for i in range(3)) or \
                all(self.state_dict[0]["observation"][i * 3 + (2 - i) + 9] == 1 for i in range(3)):
            return True

            # 检查是否平局：棋盘已满且没有连线
        if all(self.legal_action[i + 2] != 0 for i in range(9)):
            return True

            # 如果没有结束，返回 False
        return False

if __name__ == "__main__":
    env = TicTacToe()
    done = False
    _, r, d, state_dict = env.reset(eval_mode=True)
    step = 0
    while not done:
        actions = [[] for _ in range(2)]
        for i in range(2):
            action = env.get_random_action(state_dict)
            actions[i] = action
            _, r, d, state_dict = env.step(actions, render=(i == 0))
            done = all(d)
            actions[i] = []
        step += 1
    env.close_game()