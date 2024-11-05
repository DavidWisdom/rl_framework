import random
from env import Env
import numpy as np

class TicTacToe(Env):
    LABEL_SIZE_LIST = [10]
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
        actions = [[], []]

        return actions

    def step(self, actions, render=False):
        # action = [pos]
        obs = [self.state_dict[i]["observation"] for i in range(2)]
        done = [False for i in range(2)]
        reward = 0
        for i, act in enumerate(actions):
            if act:
                assert self.turn_no == i
                a = act[0]
                assert len(act) == 1 and 0 <= a <= 9
                self.turn_no = (self.turn_no + 1) % 2
                if a == 0:
                    done = [True, True]
                elif 1 <= a <= 9:
                    assert self.legal_action[a + 2] == 1
                    self.state_dict[i]["observation"][a] = 1
                    self.state_dict[self.turn_no]["observation"][a + 9] = 1
                    self.legal_action[a + 2] = 0
                done[i] = self._is_over()
                if done[i]:
                    self.legal_action[1] = 1
        if render:
            self._render()

        return obs, reward, done, self.state_dict

    def reset(self, eval_mode=False):
        self.eval_mode = eval_mode
        self._init_game()

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
        self.done = False
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
        if all(self.state_dict[0]["observation"][i] != 0 for i in range(9)):
            return True

            # 如果没有结束，返回 False
        return False

if __name__ == "__main__":
    game = TicTacToe()
    pass
    # while not done:
    #     actions = [[] for _ in range(len(self.agents))]
    #     for i in range(2):
    #         action = game.get_random_action()
    #         action, d_action, sample = agent.process(state_dict[i])
    #         if eval_mode:
    #             action = d_action
    #         actions[i] = action
    #         rewards[i].append(sample["reward"])
    #         if agent.is_latest_model and not eval_mode:
    #             sample_manager.save_sample(
    #                 **sample, agent_id=i, game_id=game_id,
    #             )
    #         if self.env.is_turn:
    #             _, r, d, state_dict = self.env.step(actions)
    #             done = done and d[i]
    #             actions[i] = []
    #     if not self.env.is_turn:
    #         _, r, d, state_dict = self.env.step(actions)
    #         done = any(d)
    #     step += 1
    #     self._save_last_sample(done, eval_mode, sample_manager, state_dict)
    # self.env.close_game()