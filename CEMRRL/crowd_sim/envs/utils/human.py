import random
import numpy as np
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super(Human, self).__init__(config, section)
        self.isObstacle = False  # whether the human is a static obstacle (part of wall) or a moving agent

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        self_humanstate: get_full_state()
        otherhuman_state: ob
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
