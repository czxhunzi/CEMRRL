import numpy as np
from crowd_nav.models.models import Actor, Critic
import torch
import torch.nn.functional as F
from copy import deepcopy
from crowd_nav.models.intrinsic_reward import IntrinsicReward


class CEMRRL:
    def __init__(self, N, memory, state_dim, action_dim, self_state_dim=7,
                 gamma=0.99, tau=0.005, batch_size=256,
                 reward_scale=10.0, hidden_size=256, hidden_dim=64, int_rew_enc_dim=16,
                 int_rew_hidden_dim=128,device='cpu'):
        self.N = N
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.episode = 0
        self.rho = 0.2
        self.e_f = 5001
        self.mu_f = 0.78

        # 多智能体 Actor-Critic 网络初始化
        self.actors = []
        self.critics = []
        self.critics_target = []
        self.critic_optimizers = []
        self.actor_optimizers = []
        self.log_alphas = []
        self.alpha_optimizers = []

        lr = self.get_lr()
        for _ in range(self.N):
            action_scale = 1.0 if N == 0 else 1.3
            actor = Actor(state_dim, action_dim, self_state_dim, hidden_size, hidden_dim, action_scale, device)
            critic = Critic(N, state_dim, action_dim, self_state_dim, hidden_size, hidden_dim, device)
            critic_target = deepcopy(critic).requires_grad_(False)
            log_alpha = torch.tensor(np.log(1.0), requires_grad=True, device=device)

            self.actors.append(actor)
            self.critics.append(critic)
            self.critics_target.append(critic_target)
            self.log_alphas.append(log_alpha)

            self.critic_optimizers.append(torch.optim.Adam(critic.parameters(), lr=lr))
            self.actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=lr))
            self.alpha_optimizers.append(torch.optim.Adam([log_alpha], lr=lr))

        self.target_entropy = -action_dim
        self.update_interval = 1

        self.intrinsic_reward = IntrinsicReward(N, state_dim, action_dim, int_rew_enc_dim=int_rew_enc_dim,
              int_rew_hidden_dim=int_rew_hidden_dim, intrinsic_reward_mode="central", device=device)

    def get_lr(self):
        return self.rho / ((self.e_f + self.episode) ** self.mu_f) + 5e-10

    def update_lr(self, optimizer):
        lr = self.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def predict(self, idx, state, deterministic=False):
        """返回第 idx 个智能体的动作"""
        actor = self.actors[idx]
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _, _ = actor.sample(state, deterministic)
        return action

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_critic(self, agent_id, states, actions, rewards, next_states, dones):
        next_actions, log_probs = [], []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                next_action, log_prob, _ = actor.sample(next_states[i], deterministic=False)
                next_actions.append(next_action)
                log_probs.append(log_prob)
            q_min = torch.min(*self.critics_target[agent_id](next_states, next_actions))
            q_target = self.reward_scale * rewards[agent_id] + (1 - dones) * self.gamma * (
                    q_min - self.log_alphas[agent_id].exp() * log_probs[agent_id])

        q1, q2 = self.critics[agent_id](states, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizers[agent_id].zero_grad()
        q_loss.backward()
        self.critic_optimizers[agent_id].step()
        self.update_lr(self.critic_optimizers[agent_id])

    def update_actor(self, agent_id, states):
        current_actions, log_probs = [], []
        for i, actor in enumerate(self.actors):
            current_action, log_prob, _ = actor.sample(states[i], deterministic=False)
            current_actions.append(current_action)
            log_probs.append(log_prob)

        q_val_min = torch.min(*self.critics[agent_id](states, current_actions))

        pi_loss = (self.log_alphas[agent_id].exp() * log_probs[agent_id] - q_val_min).mean()
        alpha_loss = -(self.log_alphas[agent_id].exp() * (log_probs[agent_id] + self.target_entropy).detach()).mean()

        self.actor_optimizers[agent_id].zero_grad()
        pi_loss.backward()
        self.actor_optimizers[agent_id].step()
        self.update_lr(self.actor_optimizers[agent_id])

        self.alpha_optimizers[agent_id].zero_grad()
        alpha_loss.backward()
        self.alpha_optimizers[agent_id].step()
        self.update_lr(self.alpha_optimizers[agent_id])

    def train_intrinsic_reward(self):
        batchs = self.memory.sample(self.batch_size)
        self.intrinsic_reward.train(batchs, self.episode)

    def train(self):
        self.episode += 1
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        for agent_id in range(self.N):
            self.update_critic(agent_id, states, actions, rewards, next_states, dones)
            self.update_actor(agent_id, states)
            if self.episode % self.update_interval == 0:
                self.soft_update(self.critics[agent_id], self.critics_target[agent_id])


    def save_load_model(self, op, path):
        for i, (actor, critic, target_critic) in enumerate(zip(self.actors, self.critics, self.critics_target), start=1):
            actor_path = f"{path}/Actor{i}.pt"
            critic_path = f"{path}/Critic{i}.pt"
            if op == "save":
                torch.save(actor.state_dict(), actor_path)
                torch.save(critic.state_dict(), critic_path)
            elif op == "load":
                actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
                critic_dict = torch.load(critic_path, map_location=torch.device('cpu'))
                critic.load_state_dict(critic_dict)
                target_critic.load_state_dict(critic_dict)
            else:
                raise ValueError(f"Unknown operation '{op}'. Use 'save' or 'load'.")


