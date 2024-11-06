from agent import Agent as AgentBase
class Agent(AgentBase):
    def state_process(self, state_dict):
        # TODO:
        return state_dict

    def reward_process(self, state_dict):
        reward = list(state_dict["reward"])
        if reward is None:
            return reward

        reward_sum = reward[-1]
        # TODO:

        reward[-1] = reward_sum
        return tuple(reward)

    def feature_post_process(self, state_dict):
        state_dict = self.state_process(state_dict)
        state_dict["reward"] = self.reward_process(state_dict)
        return state_dict