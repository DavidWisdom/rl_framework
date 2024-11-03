import time
import traceback


class Actor:
    def __init__(self, actor_id, agents, max_episode: int = 0, env=None, is_train=True):
        self.actor_id = actor_id
        self.agents = agents
        self.max_episode = max_episode
        self.episode_num = 0
        self.env = env
        self.is_train = is_train
        self.sample_manager = None

    def set_env(self, environment):
        self.env = environment

    def set_sample_manager(self, sample_manager):
        self.sample_manager = sample_manager

    def set_agents(self, agents):
        self.agents = agents

    def reload_agents(self, eval_mode=False):
        for i, agent in enumerate(self.agents):
            if eval_mode:
                agent.reset("common_ai")
            else:
                agent.reset("network")

    def run_episode(self, eval_mode=False):
        pass

    def run(self, eval_freq):
        self.episode_num = 0
        while True:
            try:
                self.episode_num += 1
                # TODO:
                pass
            except Exception as e:
                # TODO:
                traceback.print_exc()
                time.sleep(1)
            if 0 < self.max_episode <= self.episode_num:
                break