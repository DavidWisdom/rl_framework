import random
import time

from env import Env


class TicTacToe(Env):
    LABEL_SIZE_LIST = [10]
    # 0-9
    # 0: no action
    PLAYER_NUM = 2
    def __init__(self, is_turn=False, eval_mode=False, predict_frequency=1):
        super().__init__()
        self.is_turn = True
        self.eval_mode = eval_mode
        self.predict_frequency = 1
        self._init_game()

    def action_space(self):
        return self.LABEL_SIZE_LIST

    def get_random_action(self, info):
        legal_positions = [index - 1 for index in range(1, len(info[self.turn_no]["legal_action"])) if info[self.turn_no]["legal_action"][index] == 1]
        if not legal_positions:
            assert False
        return [random.choice(legal_positions)]

    def step(self, actions, render=False, slow_time=0):
        act = actions[self.turn_no]
        assert len(act) == 1
        a = act[0]
        assert 0 <= a <= 9
        if 1 <= a <= 9:
            assert self.legal_action[1 + a] == 1
            self.player_obs[self.turn_no][(-1) + a] = 1
            self.legal_action[1 + a] = 0
        if self.winner != -1:
            self.done[self.turn_no] = True
        else:
            self.done[self.turn_no] = self._is_over()
            if self.done[self.turn_no]:
                self.winner = self.turn_no
                self._limit_action()
        if self.winner != self.turn_no:
            reward = -0.1
        else:
            reward = +1
        self.state_dict[self.turn_no]["reward"] = reward
        self._next_turn()
        self.state_dict[self.turn_no]["legal_action"] = self.legal_action
        if self.turn_no == 0:
            self.state_dict[self.turn_no]["observation"] = self.player_obs[0] + self.player_obs[1]
        else:
            self.state_dict[self.turn_no]["observation"] = self.player_obs[1] + self.player_obs[0]
        self.state_dict[self.turn_no]["turn_no"] = self.turn_no
        self.state_dict[self.turn_no]["winner"] = self.winner
        obs = [self.state_dict[i]["observation"] for i in range(self.PLAYER_NUM)]
        if render:
            self._render(slow_time)
        return obs, reward, self.done, self.state_dict

    def reset(self, eval_mode=False):
        self.eval_mode = eval_mode
        self._init_game()
        return None, 0, False, self.state_dict

    def close_game(self):
        pass

    def _limit_action(self):
        self.legal_action[1] = 1
        for j in range(1, 10):
            self.legal_action[1 + j] = 0

    def _render(self, slow_time):
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
                if self.player_obs[0][row * 3 + col] == 1:
                    mark = player_mark
                if self.player_obs[1][row * 3 + col] == 1:
                    mark = enemy_mark
                # 打印当前位置的标记
                print(f"{mark} | ", end="")
            print()
            print("  +---+---+---+")
        # 打印棋盘底部
        print("    0   1   2")
        print("-----------------")
        time.sleep(slow_time)

    def _init_game(self):
        self.turn_no = 0
        self.legal_action = [1] * 11
        self.legal_action[1] = 0
        self.done = [False for _ in range(self.PLAYER_NUM)]
        self.player_obs = [[0] * 9 for _ in range(self.PLAYER_NUM)]
        self.state_dict = [{} for _ in range(self.PLAYER_NUM)]
        self.winner = -1
        for i in range(self.PLAYER_NUM):
            self.state_dict[i]["observation"] = (self.player_obs[0] + self.player_obs[1]) if self.turn_no == 0 else (self.player_obs[1] + self.player_obs[0])
            self.state_dict[i]["legal_action"] = self.legal_action
            self.state_dict[i]["reward"] = 0

    def _next_turn(self):
        self.turn_no = (self.turn_no + 1) % self.PLAYER_NUM

    def _is_over(self):
        for i in range(3):
            # 水平连线
            if all(self.player_obs[0][i * 3 + j] == 1 for j in range(3)) or \
                    all(self.player_obs[1][i * 3 + j] == 1 for j in range(3)):
                return True
            # 垂直连线
            if all(self.player_obs[0][j * 3 + i] == 1 for j in range(3)) or \
                    all(self.player_obs[1][j * 3 + i] == 1 for j in range(3)):
                return True
        # 检查对角线连线
        if all(self.player_obs[0][j * 3 + j] == 1 for j in range(3)) or \
                all(self.player_obs[1][j * 3 + j] == 1 for j in range(3)):
            return True
        if all(self.player_obs[0][j * 3 + (2 - j)] == 1 for j in range(3)) or \
                all(self.player_obs[1][j * 3 + (2 - j)] == 1 for j in range(3)):
            return True
        # 检查是否平局：棋盘已满且没有连线
        if all(self.legal_action[j + 2] == 0 for j in range(9)):
            return True
        # 如果没有结束，返回 False
        return False

if __name__ == "__main__":
    env = TicTacToe()
    done = False
    _, r, d, state_dict = env.reset(eval_mode=True)
    step = 0
    PLAYER_NUM = 2
    actions = [[] for _ in range(PLAYER_NUM)]
    while not done:
        turn_no = env.turn_no
        action = env.get_random_action(state_dict)
        actions[turn_no] = action
        _, r, d, state_dict = env.step(actions, render=True)
        done = all(d)
        step += 1
    env.close_game()