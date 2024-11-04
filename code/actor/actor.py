import time
import traceback


class Actor:
    def __init__(self, actor_id, agents, max_episode: int = 0, env=None, is_train=True):
        self.actor_id = actor_id
        self.agents = agents
        self._max_episode = max_episode
        self._episode_num = 0
        self.env = env
        self.is_train = is_train
        self.sample_manager = None

    def set_env(self, environment):
        self.env = environment

    def set_sample_manager(self, sample_manager):
        self.sample_manager = sample_manager

    def set_agents(self, agents):
        self.agents = agents

    def _reload_agents(self, eval_mode=False):
        for i, agent in enumerate(self.agents):
            if eval_mode:
                agent.reset("common_ai")
            else:
                agent.reset("network")

    def _save_last_sample(self, done, eval_mode, sample_manager, state_dict):

        pass

    def _run_episode(self, eval_mode=False):
        done = False
        self._reload_agents(eval_mode=eval_mode)
        while not done:
            for i, agent in enumerate(self.agents):
                pass
        self.env.close_game()

        if self.is_train and not eval_mode:
            pass
        self._print_info()

    def _print_info(self):
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
            except Exception as e:
                traceback.print_exc()
                time.sleep(1)
            if 0 < self._max_episode <= self._episode_num:
                break