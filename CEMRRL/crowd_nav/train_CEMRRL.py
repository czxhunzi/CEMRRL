import os
import sys
import gym
import torch
import shutil
import logging
import argparse
import configparser
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from crowd_nav.policy.CEMRRL import CEMRRL
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.utils import obs2state
from crowd_nav.utils.utils import ReplayBuffer2 as Memory

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='CEMRRL')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument("--env_id", type=str, default="CrowdSim-v0", help="Environment Id")
    args = parser.parse_args()

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')

    # configure logging
    mode = 'w' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    env = gym.make(args.env_id)
    env.configure(env_config)

    robot_num = env_config.getint('sim', 'robot_num')
    max_capacity = train_config.getint('memory', 'max_capacity')
    state_dim = train_config.getint('dim', 'joint_state_dim')
    action_dim = train_config.getint('dim', 'action_dim')
    coeff = train_config.getfloat('trainer', 'coeff')
    initial_random_steps = train_config.getint('train', 'initial_random_steps')
    max_episode = train_config.getint('train', 'max_episode')
    gamma = train_config.getfloat('trainer', 'gamma')
    tau = train_config.getfloat('trainer', 'tau')
    batch_size = train_config.getint('trainer', 'batch_size')
    hidden_size =  train_config.getint('net', 'hidden_size')
    hidden_dim = train_config.getint('net', 'hidden_dim')
    int_rew_enc_dim = train_config.getint('net', 'int_rew_enc_dim')
    int_rew_hiddem_dim = train_config.getint('net', 'int_rew_hiddem_dim')

    robots = [Robot(env_config, 'robot') for _ in range(robot_num)]

    memory = Memory(max_capacity, device = device)
    trainer = CEMRRL(robot_num, memory, state_dim, action_dim,
                     gamma=gamma, tau=tau, batch_size=batch_size, hidden_size=hidden_size,
                     hidden_dim=hidden_dim, int_rew_enc_dim=int_rew_enc_dim, int_rew_hidden_dim=int_rew_hiddem_dim,
                     device=device)

    assert len(robots) == len(trainer.actors)
    for robot, policy in zip(robots, trainer.actors):
        robot.set_policy(policy)
    env.set_robot(robots)

    training_step = 0
    for episode in range(max_episode):
        done = False
        obs = env.reset('train')

        tol_rewards = [0] * robot_num
        infos = [None] * robot_num
        states = [obs2state(robots[i], obs[i]) for i in range(robot_num)]

        error1_list, error2_list = [], []
        step = 0

        while not done:

            actions = [trainer.predict(i, states[i], deterministic=False).squeeze(0) for i in range(robot_num)]

            obs, ex_rewards, done, infos, _, error1, error2 = env.step(actions)

            tol_rewards = [t + r for t, r in zip(tol_rewards, ex_rewards)]

            next_states = [obs2state(robots[i], obs[i]) for i in range(robot_num)]

            int_rewards = trainer.intrinsic_reward.get_intrinsic_rewards(states)

            rewards = np.array([ex_rewards]) + coeff * np.array([int_rewards])

            rewards = rewards.T

            memory.push(states, actions, rewards, next_states, done)

            if episode > initial_random_steps:
                trainer.train_intrinsic_reward()

            states = next_states

            error1_list.append(error1)
            error2_list.append(error2)

            step += 1

        error = (sum(error1_list) + sum(error2_list)) / step

        if episode > initial_random_steps:
            trainer.train()

        logging.info('episode:{}, reward1:{}, reward2:{}, reward3:{}, memory size1:{}, time:{},'
                     ' info1:{}, error:{}'.format(episode, ('%.2f' % tol_rewards[0]), ('%.2f' % tol_rewards[1]), ('%.2f' % tol_rewards[2]), len(memory), ('%.2f' % env.global_time), infos[0], ('%.2f' % error)))

        if episode % args.save_interval == 0 or episode == max_episode - 1:
            trainer.save_load_model('save', args.output_dir)
        training_step += 1

if __name__ == '__main__':
    main()
