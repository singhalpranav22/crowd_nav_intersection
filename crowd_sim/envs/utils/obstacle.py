from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Obstacle(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.v_pref = 0
        self.radius = 0.1

    def act(self, ob):
        """
        The state for obstacle is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
