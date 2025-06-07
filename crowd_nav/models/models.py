import torch
import torch.nn as nn
from torch.distributions import Normal

MIN_LOG_STD = -20
MAX_LOG_STD = 2


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, source_dims, hidden_cnt):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=source_dims, out_channels=hidden_cnt, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_cnt, out_channels=source_dims, kernel_size=1)
        self.layer_norm = nn.LayerNorm(source_dims)

    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class Critic(nn.Module):
    def __init__(self, N, state_dim, action_dim, self_state_dim, hidden_size, hidden_dim, device):
        super().__init__()
        self.N = N
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.self_state_dim = self_state_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.linear = nn.Linear(state_dim - self_state_dim, hidden_dim).to(self.device)
        self.multi_self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4).to(self.device)
        self.feed_forward_net = PoswiseFeedForwardNet(source_dims=hidden_dim, hidden_cnt=hidden_dim * 4).to(self.device)

        self.net = nn.Sequential(
            nn.Linear(N * (action_dim + hidden_dim + self_state_dim), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()).to(self.device)
        self.q_net1 = nn.Linear(hidden_size, 1).to(self.device)
        self.q_net2 = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, states, actions):
        # states: list of len N, each (B, T, D); actions: list of len N, each (B, A)
        joint_states = []
        joint_actions = []
        for i in range(self.N):
            s = states[i]
            a = actions[i]
            self_s = s[:, 0, :self.self_state_dim]
            other_s = s[:, :, self.self_state_dim:]
            other_s = torch.relu(self.linear(other_s))
            # out, _ = self.multi_self_attention(other_s, other_s, other_s)
            other_s = other_s.transpose(0, 1)
            out, _ = self.multi_self_attention(other_s, other_s, other_s)
            # 转回 (batch_size, seq_len, embed_dim)
            out = out.transpose(0, 1)
            out = self.feed_forward_net(out)
            out = torch.mean(out, dim=1)
            joint = torch.cat([self_s, out], dim=-1)
            joint_states.append(joint)
            joint_actions.append(a.reshape(-1, self.action_dim))

        x = torch.cat(joint_states + joint_actions, dim=1)

        x = self.net(x)
        q1 = self.q_net1(x)
        q2 = self.q_net2(x)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, self_state_dim, hidden_size, hidden_dim, action_scale, device,
                 kinematics='holonomic'):
        super().__init__()

        self.state_dim = state_dim
        self.action = action_dim
        self.self_state_dim = self_state_dim
        self.hidden_dim = hidden_dim
        self.action_scale = action_scale
        self.device = device
        self.kinematics = kinematics
        self.multiagent_training = True

        self.linear = nn.Linear(state_dim - self_state_dim, hidden_dim).to(self.device)
        self.multi_self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4).to(self.device)
        self.feed_forward_net = PoswiseFeedForwardNet(source_dims=hidden_dim, hidden_cnt=hidden_dim*4).to(self.device)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + self_state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()).to(self.device)
        self.mean_linear = nn.Linear(hidden_size, action_dim).to(self.device)
        self.log_std_linear = nn.Linear(hidden_size, action_dim).to(self.device)

    def forward(self, state):
        self_s = state[:, 0, :self.self_state_dim]
        other_s = state[:, :, self.self_state_dim:]
        other_s = torch.relu(self.linear(other_s))
        # out, _ = self.multi_self_attention(other_s, other_s, other_s)
        other_s = other_s.transpose(0, 1)
        out, _ = self.multi_self_attention(other_s, other_s, other_s)
        # 转回 (batch_size, seq_len, embed_dim)
        out = out.transpose(0, 1)
        out = self.feed_forward_net(out)
        out = torch.mean(out, dim=1)
        joint_state = torch.cat([self_s, out], dim=-1)

        x = self.net(joint_state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(MIN_LOG_STD, MAX_LOG_STD)

        return mean, log_std

    def sample(self, state, deterministic):
        mean, log_std = self.forward(state)

        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            position_x = dist.mean
        else:
            position_x = dist.rsample()
        A_ = torch.tanh(position_x)
        log_prob = dist.log_prob(position_x) - torch.log(1 - A_.pow(2) + 1e-6)

        return A_ * self.action_scale, log_prob.sum(1, keepdim=True), torch.tanh(mean) * self.action_scale

