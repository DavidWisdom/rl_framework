import time
import traceback
import numpy as np


class Actor:
    def __init__(self, actor_id, agents, max_episode: int = 0, env=None, is_train=True):
        self.actor_id = actor_id
        self.agents = agents
        self._max_episode = max_episode
        self._episode_num = 0
        self.env = env
        self.is_train = is_train
        self._sample_manager = None

    def set_env(self, environment):
        self.env = environment

    def set_sample_manager(self, sample_manager):
        self._sample_manager = sample_manager

    def set_agents(self, agents):
        self.agents = agents

    def _reload_agents(self, eval_mode=False):
        for i, agent in enumerate(self.agents):
            if eval_mode:
                agent.reset("common_ai")
            else:
                agent.reset("network")

    def _save_last_sample(self, done, eval_mode, sample_manager, state_dict, turn_no=-1):
        if done:
            for i, agent in enumerate(self.agents):
                if turn_no != -1:
                    i = turn_no
                    agent = self.agents[i]
                if agent.is_latest_model and not eval_mode:
                    if state_dict[i]["reward"] is not None:
                        if type(state_dict[i]["reward"]) == tuple:
                            sample_manager.save_last_sample(
                                agent_id=i, reward=state_dict[i]["reward"][-1]
                            )
                        else:
                            sample_manager.save_last_sample(
                                agent_id=i, reward=state_dict[i]["reward"]
                            )
                    else:
                        sample_manager.save_last_sample(agent_id=i, reward=0)
                if turn_no != -1:
                    break

    def _run_episode(self, eval_mode=False):
        sample_manager = self._sample_manager
        done = False
        self._reload_agents(eval_mode=eval_mode)

        _, r, d, state_dict = self.env.reset(eval_mode=eval_mode)
        if state_dict[0] is None:
            game_id = state_dict[1]["game_id"]
        else:
            game_id = state_dict[0]["game_id"]
        sample_manager.reset(agents=self.agents, game_id=game_id)
        actions = [[] for _ in range(len(self.agents))]
        rewards = [[] for _ in range(len(self.agents))]
        step = 0
        game_info = {}
        if not self.env.is_turn:
            while not done:
                for i, agent in enumerate(self.agents):
                    action, d_action, sample = agent.process(state_dict[i])
                    if eval_mode:
                        action = d_action
                    actions[i] = action
                    rewards[i].append(sample["reward"])
                    if agent.is_latest_model and not eval_mode:
                        sample_manager.save_sample(
                            **sample, agent_id=i, game_id=game_id,
                        )
                _, r, d, state_dict = self.env.step(actions)
                done = any(d)
                step += 1
                self._save_last_sample(done, eval_mode, sample_manager, state_dict)
        else:
            while not done:
                i = self.env.get_turn_no()
                agent = self.agents[i]
                action, d_action, sample = agent.process(state_dict[i])
                if eval_mode:
                    action = d_action
                actions[i] = action
                rewards[i].append(sample["reward"])
                if agent.is_latest_model and not eval_mode:
                    sample_manager.save_sample(
                        **sample, agent_id=i, game_id=game_id,
                    )
                _, r, d, state_dict = self.env.step(actions)
                done = all(d)
                step += 1
                self._save_last_sample(d[i], eval_mode, sample_manager, state_dict, turn_no=i)
        self.env.close_game()

        if self.is_train and not eval_mode:
            sample_manager.send_samples()
        self._print_info(
            # TODO
        )

    def _print_info(self):
        # TODO
        pass

    def run(self, eval_freq):
        self._episode_num = 0
        while True:
            try:
                self._episode_num += 1
                eval_mode = (
                    self._episode_num % eval_freq == 0
                )
                self._run_episode(eval_mode=eval_mode)
            except Exception:
                traceback.print_exc()
                time.sleep(1)
                raise
            if 0 < self._max_episode <= self._episode_num:
                break