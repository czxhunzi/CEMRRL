#!/usr/bin/env python2
# coding=utf-8
import logging
import math
import os
import random
import sys
from collections import deque
import gym
import numpy as np
from gym import spaces
from matplotlib.patches import Wedge
from numpy.linalg import norm
from crowd_nav.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.utils import point_to_segment_dist
from math import pow, pi, sqrt, atan2, sin, cos, e
import matplotlib.lines as mlines
from matplotlib import patches

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        # self.state_space = None
        self.group_human = None
        self.end_goal_change_chance = None
        self.end_goal_changing = None
        self.goal_change_chance = None
        self.dis = None
        self.random_v_pref = None
        self.random_radii = None
        self.random_goal_changing = None
        self.potential = None
        self.time_limit = None
        self.time_step = None
        self.robots = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.base_frame = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.human_actions = None
        self.ax = None
        self.fig = None
        self.formation_offsets = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        # configure randomized goal changing of humans midway through episode

        self.random_goal_changing = config.getboolean('env', 'random_goal_changing')
        if self.random_goal_changing:
            self.goal_change_chance = config.getfloat('env', 'goal_change_chance')
        # configure randomized goal changing of humans after reaching their respective goals
        self.end_goal_changing = config.getboolean('env', 'end_goal_changing')
        if self.end_goal_changing:
            self.end_goal_change_chance = config.getfloat('env', 'end_goal_change_chance')
        # configure randomized radii changing when reaching goals
        self.random_radii = config.getboolean('env', 'random_radii')
        # configure randomized v_pref changing when reaching goals
        self.random_v_pref = config.getboolean('env', 'random_v_pref')

        if self.config.get('humans', 'policy') == 'orca' \
                or self.config.get('humans', 'policy') == 'social_force':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000,
                              'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
            self.group_human = config.getboolean('env', 'group_human')
        else:
            raise NotImplementedError

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(
            self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(
            self.square_width, self.circle_radius))

    def set_robot(self, robots):
        self.robots = robots

    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def update_human_goals_randomly(self):
        # Update humans' goals randomly
        for human in self.humans:
            if human.isObstacle or human.v_pref == 0:
                continue
            if np.random.random() <= self.goal_change_chance:
                humans_copy = []
                if not self.group_human:  # to improve the runtime
                    for h in self.humans:
                        if h != human:
                            humans_copy.append(h)

                # Produce valid goal for human in case of circle setting
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases' robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False

                    if self.group_human:
                        pass
                    else:
                        for agent in self.robots + humans_copy:
                            min_dist = human.radius + agent.radius + self.discomfort_dist
                            if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                    norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                                collide = True
                                break
                    if not collide:
                        break
                # Give human new goal
                human.gx = gx
                human.gy = gy
        return

    # Update the specified human's end goals in the environment randomly
    def update_human_goal(self, human):
        # Update human goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            humans_copy = []
            if not self.group_human:
                for h in self.humans:
                    if h != human:
                        humans_copy.append(h)

            # Update human's radius now that it's reached goal
            if self.random_radii:
                human.radius += np.random.uniform(-0.1, 0.1)

            # Update human's v_pref now that it's reached goal
            if self.random_v_pref:
                human.v_pref += np.random.uniform(-0.1, 0.1)

            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                if self.group_human:  # False
                    pass
                else:
                    for agent in self.robots + humans_copy:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                            collide = True
                            break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gx
        return

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                human = self.generate_circle_crossing_human()
                self.humans.append(human)
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 1 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                px, py = -4, -1
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * 6
                        collide = False
                        for agent in self.robots + self.humans:
                            if norm((
                                    px - agent.px,
                                    py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < human_num / 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            self.circle_radius_p = 3.3
            angle = np.random.random() * 2 * np.pi
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius_p * np.cos(angle) + px_noise
            py = self.circle_radius_p * np.sin(angle) + py_noise
            collide = False
            for agent in self.robots + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        human.set(px-1.1, py, -px-1.1, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = np.random.random() * self.square_width * 0.5 * sign
            collide = False
            for agent in self.robots + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in self.robots + self.humans:
                if norm((gx - agent.gx,
                         gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.flags = [0 for i in range(len(self.robots))]
        if self.robots is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case

        self.global_time = 0
        self.human_times = [0] * self.human_num

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}

            # three robot 0: leader ; 1,2: follower
            # self.robots :[leader, follower, follower]
            self.robots[0].set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            self.robots[1].set(2, -self.circle_radius, 1.5, self.circle_radius-math.sqrt(1.75), 0, 0, np.pi / 2)
            self.robots[2].set(-2, -self.circle_radius, -1.5, self.circle_radius-math.sqrt(1.75), 0, 0, np.pi / 2)
            self.formation_offsets = [(1.5, math.sqrt(1.75)), (-1.5, math.sqrt(1.75))]

            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in self.robots + self.humans:
            agent.time_step = self.time_step

        self.states = list()
        robot_obs = self.get_robot_obs()

        return robot_obs

    def get_robot_obs(self):
        human_obs = [human.get_observable_state() for human in self.humans]
        ob_robots = []
        for i in range(len(self.robots)):
            others = [r.get_observable_state() for j, r in enumerate(self.robots) if j != i]
            ob_robots.append(human_obs + others)
        return ob_robots

    def get_robot_states(self):
        human_obs = [human.get_observable_state() for human in self.humans]
        ob_robots = []
        for i in range(len(self.robots)):
            others = [r.get_observable_state() for j, r in enumerate(self.robots)]
            ob_robots.append(human_obs + others)
        return ob_robots



    # def check_collision(self, robot, action, humans):
    #     min_dist = float('inf')
    #     collision = False
    #     for human in humans:
    #         px, py = human.px - robot.px, human.py - robot.py
    #         vx, vy = human.vx - action.vx, human.vy - action.vy
    #         ex, ey = px + vx * self.time_step, py + vy * self.time_step
    #         dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot.radius
    #         if dist < 0:
    #             return True, dist
    #         min_dist = min(min_dist, dist)
    #     return collision, min_dist

    def check_reach_goal(self, robot, action):
        end_position = np.array(robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(robot.get_goal_position())) < robot.radius
        return reaching_goal

    def compute_formation_reward(self, error):
        if 0 <= error <= 0.2:
            return 1
        elif error <= 1:
            return -np.tanh(7.5 * error - 3)
        elif error <= 2:
            return -1
        else:
            return -error

    def get_robot_reward(self, robot, action, dmin, reaching_goal, collision, rewardf, flag):
        if self.global_time >= self.time_limit - 1:
            return 0 + rewardf, Timeout()
        elif collision:
            return self.collision_penalty + rewardf, Collision()
        elif reaching_goal:
            if flag:
                return 0 + rewardf, ReachGoal()
            else:
                return self.success_reward + rewardf, ReachGoal()
        elif dmin < self.discomfort_dist:
            return (dmin - self.discomfort_dist) * 2.5 + rewardf, Danger(dmin)
        else:
            last, end = self.onestep_distance(robot, action)
            return (last - end) * 2 + rewardf, Nothing()

    def step(self, actions):
        """
        Compute actions for all agents, detect collision,
        update environment and return (ob, reward, done, info)
        """
        actions = [robot.clip_action(action, robot.v_pref) for action, robot in zip(actions, self.robots)]
        # global ob
        human_actions = []
        for human in self.humans:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            human_actions.append(human.act(ob))
        self.human_actions = human_actions

        robots_num = len(self.robots)
        collisions = [False for _ in range(robots_num)]
        dmins = [float('inf') for _ in range(robots_num)]
        rewards = [0 for _ in range(robots_num)]
        infos = [None for _ in range(robots_num)]

        # -------------------
        # 机器人与人类碰撞检测
        # -------------------
        for i, robot in enumerate(self.robots):
            action = actions[i]
            for human in self.humans:
                px = human.px - robot.px
                py = human.py - robot.py
                vx = human.vx - action.vx
                vy = human.vy - action.vy
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot.radius
                if dist < 0:
                    collisions[i] = True
                    break
                elif dist < dmins[i]:
                    dmins[i] = dist

        # -------------------
        # 机器人之间碰撞检测
        # -------------------
        robot_collisions = set()
        for i in range(robots_num):
            for j in range(i + 1, robots_num):
                dx = self.robots[i].px - self.robots[j].px
                dy = self.robots[i].py - self.robots[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[i].radius - self.robots[j].radius
                if dist < 0:
                    robot_collisions.add((i, j))

        # 编队奖励 for follower
        formation_rewards = [] #error1, error2
        for i in range(1, robots_num):
            offset = self.formation_offsets[i-1]  # tuple (dx, dy)
            dx = self.robots[i].px - self.robots[0].px - offset[0]  #TODD
            dy = self.robots[i].py - self.robots[0].py - offset[1]
            error = np.linalg.norm([dx, dy])
            if error <= 0.2:
                rewardf = 1
            elif error <= 1:
                rewardf = -np.tanh(7.5 * error - 3)
            elif error <= 2:
                rewardf = -1
            else:
                rewardf = -error
            formation_rewards.append(rewardf)

        reaching_goals = [self.check_reach_goal(robot, action) for robot, action in zip(self.robots, actions)]



        # leader 奖励
        for i in range(robots_num):
            is_leader = (i == 0)
            # 超时处理
            if self.global_time >= self.time_limit - 1:
                reward = 0 if is_leader else formation_rewards[i - 1]
                info = Timeout()
            # 碰撞检测（follower 多考虑和其它机器人的碰撞）
            elif collisions[i] or any(i in pair for pair in robot_collisions):
                base_penalty = self.collision_penalty if is_leader else -33.5
                reward = base_penalty + (0 if is_leader else formation_rewards[i - 1])
                info = Collision()
            # 到达目标点
            elif reaching_goals[i]:
                reward = 0 if self.flags[i] else (self.success_reward if is_leader else formation_rewards[i - 1])
                self.flags[i] += 1
                info = ReachGoal()
            # 距离过近
            elif dmins[i] < self.discomfort_dist:
                reward = (dmins[i] - self.discomfort_dist) * self.discomfort_penalty_factor
                info = Danger(dmins[i])
            # 正常奖励
            else:
                if is_leader:
                    last1, end1 = self.onestep_distance(self.robots[i], actions[i])
                    reward = (last1 - end1) * 2
                else:
                    reward = formation_rewards[i - 1]
                info = Nothing()

            # 赋值
            rewards[i] = reward
            infos[i] = info

        if self.global_time >= self.time_limit - 1:
            done = True
        elif any(collisions) or len(robot_collisions) > 0:
            done = True
        elif reaching_goals[0]:
            done = True
        else:
            done = False

        robot_full_states = [robot.get_full_state() for robot in self.robots]
        human_full_states = [human.get_full_state() for human in self.humans]
        robot_full_states.append(human_full_states)
        self.states.append(robot_full_states)

        # update all agents
        for i, action in enumerate(actions):
            self.robots[i].step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)

        self.global_time += self.time_step

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # compute the observation
        robot_obs = self.get_robot_obs()
        robot_states = self.get_robot_states()

        return robot_obs, rewards, done, infos, robot_states, formation_rewards[0], formation_rewards[1]

    def onestep_distance(self, robot, action):

        robot_state = robot.get_full_state()
        last_position = np.array([robot_state.px, robot_state.py])
        end_position = np.array(robot.compute_position(action, self.time_step))
        last_distance = norm(last_position - np.array(robot.get_goal_position()))
        end_distance = norm(end_position - np.array(robot.get_goal_position()))
        return last_distance, end_distance

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot1_color = 'green'
        robot2_color = 'yellow'
        robot3_color = 'orange'
        goal1_color = 'red'
        goal2_color = 'red'
        goal3_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            for robot in self.robots:
                ax.add_artist(plt.Circle(robot.get_position(), robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=22)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=22)
            ax.set_ylabel('y(m)', fontsize=22)

            robot1_positions = [self.states[i][0].position for i in range(len(self.states))]
            robot2_positions = [self.states[i][1].position for i in range(len(self.states))]
            robot3_positions = [self.states[i][2].position for i in range(len(self.states))]
            goal1 = mlines.Line2D([-1.1], [4], color=goal1_color, marker='*', linestyle='None', markersize=15, label='Goal1')
            goal2 = mlines.Line2D([1.5], [4-math.sqrt(1.75)], color=goal2_color, marker='*', linestyle='None', markersize=15, label='Goal2')
            goal3 = mlines.Line2D([-1.5], [4-math.sqrt(1.75)], color=goal3_color, marker='*', linestyle='None', markersize=15, label='Goal3')
            human_positions = [[self.states[i][3][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot1 = plt.Circle(robot1_positions[k], self.robots[0].radius, fill=True, color=robot1_color)
                    robot2 = plt.Circle(robot2_positions[k], self.robots[1].radius, fill=True, color=robot2_color)
                    robot3 = plt.Circle(robot3_positions[k], self.robots[2].radius, fill=True, color=robot3_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot1)
                    ax.add_artist(robot2)
                    ax.add_artist(robot3)
                    ax.add_artist(goal1)
                    # ax.add_artist(goal2)
                    # ax.add_artist(goal3)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot1, robot2, robot3]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=20) for i in range(self.human_num + 3)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction1 = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot1_color, ls='solid')
                    nav_direction2 = plt.Line2D((self.states[k - 1][1].px, self.states[k][1].px),
                                               (self.states[k - 1][1].py, self.states[k][1].py),
                                               color=robot2_color, ls='solid')
                    nav_direction3 = plt.Line2D((self.states[k - 1][2].px, self.states[k][2].px),
                                                (self.states[k - 1][2].py, self.states[k][2].py),
                                                color=robot3_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][3][i].px, self.states[k][3][i].px),
                                                   (self.states[k - 1][3][i].py, self.states[k][3][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction1)
                    ax.add_artist(nav_direction2)
                    ax.add_artist(nav_direction3)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot1, robot2, robot3, goal1], ['Leader', 'Follower1', 'Follower2', 'Goal'], fontsize=18.5)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal 添加机器人及其目标
            robot1_positions = [state[0].position for state in self.states]
            robot2_positions = [state[1].position for state in self.states]
            robot3_positions = [state[2].position for state in self.states]
            goal1 = mlines.Line2D([0], [4], color=goal1_color, marker='*', linestyle='None', markersize=15, label='Goal1')
            goal2 = mlines.Line2D([1.5], [4-math.sqrt(1.75)], color=goal2_color, marker='*', linestyle='None', markersize=15, label='Goal2')
            goal3 = mlines.Line2D([-1.5], [4-math.sqrt(1.75)], color=goal3_color, marker='*', linestyle='None', markersize=15, label='Goal3')
            robot1 = plt.Circle(robot1_positions[0], self.robots[0].radius, fill=True, color=robot1_color)
            robot2 = plt.Circle(robot2_positions[0], self.robots[1].radius, fill=True, color=robot2_color)
            robot3 = plt.Circle(robot3_positions[0], self.robots[2].radius, fill=True, color=robot3_color)
            ax.add_artist(robot1)
            ax.add_artist(robot2)
            ax.add_artist(robot3)
            ax.add_artist(goal1)
            ax.add_artist(goal2)
            ax.add_artist(goal3)
            plt.legend([robot1, robot2, robot3, goal1, goal2, goal3], ['Robot1', 'Robot2', 'Robot3', 'Goal1', 'Goal2',
                                                                       'Goal3'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[3][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction 计算每个步骤的方向，并使用箭头指示方向
            radius = self.robots[0].radius
            if self.robots[0].kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 3):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        elif i == 1:
                            agent_state = state[1]
                        elif i == 2:
                            agent_state = state[2]
                        else:
                            agent_state = state[3][i - 3]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot1.center = robot1_positions[frame_num]
                robot2.center = robot2_positions[frame_num]
                robot3.center = robot3_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation1[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation1 in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)

                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robots[0].kinematics == 'holonomic'
                assert self.robots[1].kinematics == 'holonomic'
                assert self.robots[2].kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1] \
                             + self.states[global_step][2] + self.states[global_step][3]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot 当按下任何键时，绘制动作值图
                fig, axis = plt.subplots()
                speeds = [0] + self.robots[0].policy.speeds
                rotations = self.robots[0].policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][3:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robots[0].policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError

