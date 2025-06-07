import numpy as np

from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super(Robot, self).__init__(config, section)
        self.state = None
        self.config = config

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        self.state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(self.state)

        return action

    def clip_action(self, raw_action, v_pref):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        # quantize the action
        holonomic = True if self.kinematics == 'holonomic' else False
        # clip the action
        if isinstance(raw_action, ActionXY) or isinstance(raw_action, ActionRot):
            return raw_action
        else:
            raw_action = raw_action.cpu().numpy()
            if holonomic:
                raw_action = np.array(raw_action)
                # clip velocity
                v_norm = np.linalg.norm(raw_action)
                # v_pref = 0.5
                if v_norm > v_pref:
                    raw_action[0] = raw_action[0] / v_norm * v_pref
                    raw_action[1] = raw_action[1] / v_norm * v_pref
                    # print(raw_action[0], raw_action[1])
                return ActionXY(raw_action[0], raw_action[1])
            else:
                # for sim2real
                raw_action[0] = (raw_action[0] + 1) / 2  # action[0] is change of v
                raw_action[1] = np.clip(raw_action[1], -1, 1)
                return ActionRot(raw_action[0], raw_action[1])

    def return_state(self):
        return self.get_full_state()


