import torch
from .modules.e2s_noveld import E2S_NovelD


class IntrinsicReward:
    """
    Class Intrinsic Rewards.
    """
    def __init__(self, N, state_dim, action_dim, int_rew_enc_dim=16, int_rew_hidden_dim=128,
                 scale_fac=0.5, ridge=0.1, intrinsic_reward_mode="central", device="cpu"):
        self.N = N
        self.device = device
        self.rho = 0.2
        self.e_s = 2001
        self.mu_s = 0.95
        self.episode = 0
        int_rew_lr = self.get_lr()

        intrinsic_reward_params = {
            "enc_dim": int_rew_enc_dim,
            "act_dim": action_dim * N,
            "hidden_dim": int_rew_hidden_dim,
            "ridge": ridge,
            "scale_fac": scale_fac,
            "lr": int_rew_lr,
            "device": device}

        # Models
        self.ir_mode = intrinsic_reward_mode
        if self.ir_mode == "central":
            self.int_rew = E2S_NovelD(N * state_dim, **intrinsic_reward_params)
        elif self.ir_mode == "local":
            self.int_rew_list = [E2S_NovelD(state_dim, **intrinsic_reward_params) for _ in range(N)]

        self.last_nov = None

    def get_lr(self):
        return self.rho / ((self.e_s + self.episode) ** self.mu_s) + 5e-10

    def update_lr(self, optimizer):
        lr = self.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_intrinsic_rewards(self, states):
        int_rewards = [0] * self.N
        if self.ir_mode == "central":
            cat_obs = torch.cat([states[i].unsqueeze(0).mean(1) for i in range(len(states))], dim=-1).to(self.device)
            int_reward = self.int_rew.get_reward(cat_obs)
            int_rewards = [int_reward] * self.N
        elif self.ir_mode == "local":
            int_rewards = []
            for i in range(self.N):
                obs = states[i].unsqueeze(0).mean(1).to(self.device)
                int_rewards.append(self.int_rew_list[i].get_reward(obs))

        return int_rewards

    def train(self, batch, episode):
        self.episode = episode
        states, actions, _, _, _,  = batch
        cat_states = torch.cat([states[i].mean(1) for i in range(len(states))], dim=-1)
        cat_actions = torch.cat(actions, dim=-1)

        if self.ir_mode == "central":
            self.int_rew.train(cat_states, cat_actions)
            self.update_lr(self.int_rew.rnd.optim)
            self.update_lr(self.int_rew.e3b.encoder_optim)
            self.update_lr(self.int_rew.e3b.inv_dyn_optim)
        elif self.ir_mode == "local":
            for i in range(self.N):
                self.int_rew_list[i].train(states[i].mean(1), actions[i])
                self.update_lr(self.int_rew_list[i].rnd.optim)
                self.update_lr(self.int_rew_list[i].e3b.encoder_optim)
                self.update_lr(self.int_rew_list[i].e3b.inv_dyn_optim)

    def reset_int_reward(self, states):
        if self.ir_mode == "central":
            self.int_rew.init_new_episode()
            # Initialise intrinsic reward model with first observation
            cat_obs = torch.cat([states[i].unsqueeze(0).mean(1) for i in range(len(states))], dim=-1).to(self.device)
            self.int_rew.get_reward(cat_obs.view(1, -1))
        elif self.ir_mode == "local":
            for i in range(self.N):
                # Reset intrinsic reward model
                self.int_rew_list[i].init_new_episode()
                # Initialise intrinsic reward model with first observation
                obs = torch.Tensor(states[i]).unsqueeze(0).mean(1) .to(self.device)
                self.int_rew_list[i].get_reward(obs)

    def prep_training(self, device='cpu'):
        super().prep_training(device)
        if self.ir_mode == "central":
            self.int_rew.set_train(device)
        elif self.ir_mode == "local":
            for a_int_rew in self.int_rew:
                a_int_rew.set_train(device)

    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device)
        if self.ir_mode == "central":
            self.int_rew.set_eval(device)
        elif self.ir_mode == "local":
            for a_int_rew in self.int_rew:
                a_int_rew.set_eval(device)

    def _get_ir_params(self):
        if self.ir_mode == "central":
            return self.int_rew.get_params()
        elif self.ir_mode == "local":
            return [a_int_rew.get_params() for a_int_rew in self.int_rew]

    def _load_ir_params(self, params):
        if self.ir_mode == "central":
            self.int_rew.load_params(params)
        elif self.ir_mode == "local":
            for a_int_rew, param in zip(self.int_rew, params):
                a_int_rew.load_params(param)





