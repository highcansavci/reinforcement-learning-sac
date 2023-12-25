import numpy as np
import torch
import torch.nn.functional as F
from replay_buffer.replay_buffer import ReplayBuffer
from model.actor import Actor
from model.critic import Critic
from model.value import Value


class Agent:
    def __init__(self, actor_lr=3e-4, critic_lr=3e-4, input_dims=[8], tau=5e-3, env=None, gamma=0.99, reward_scale=2, n_actions=2, max_size=int(1e6), fc1_dim=256, fc2_dim=256, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.n_actions = n_actions
        self.scale = reward_scale
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.actor = Actor(actor_lr, input_dims, self.min_action, self.max_action, "actor", fc1_dim, fc2_dim, n_actions)
        self.critic_a = Critic(critic_lr, input_dims, n_actions, "critic_a", fc1_dim, fc2_dim)
        self.critic_b = Critic(critic_lr, input_dims, n_actions, "critic_b", fc1_dim, fc2_dim)
        self.value = Value(critic_lr, input_dims, "value", fc1_dim, fc2_dim)
        self.target_value = Value(critic_lr, input_dims, "value_target", fc1_dim, fc2_dim)
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float32, device=self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.detach().cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def learn(self):
        torch.autograd.set_detect_anomaly(True)
        if self.memory.mem_ctr < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.critic_a.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.critic_a.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.critic_a.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.critic_a.device)
        states = torch.tensor(states, dtype=torch.float32, device=self.critic_a.device)

        # Optimize Value Network by Equation 5
        value = self.value(states)
        value_ = self.target_value(next_states)
        value_[dones] = 0.0

        actions_, log_probs = self.actor.sample_normal(states, reparameterize=False)
        q_a_policy = self.critic_a(states, actions_)
        q_b_policy = self.critic_b(states, actions_)
        critic_value = torch.min(q_a_policy, q_b_policy)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Optimize Actor Network by Equation 12
        actions_, log_probs = self.actor.sample_normal(states, reparameterize=True)
        q_a_policy = self.critic_a(states, actions_)
        q_b_policy = self.critic_b(states, actions_)
        critic_value = torch.min(q_a_policy, q_b_policy)

        self.actor.optimizer.zero_grad()
        actor_loss = log_probs - critic_value
        actor_loss = actor_loss.mean()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Optimize Critic Network by Equation 7
        self.critic_a.optimizer.zero_grad()
        self.critic_b.optimizer.zero_grad()
        q_hat = self.scale * rewards.reshape(self.batch_size, 1) + self.gamma * value_
        q_a_old_policy = self.critic_a(states, actions)
        q_b_old_policy = self.critic_b(states, actions)
        critic_a_loss = 0.5 * F.mse_loss(q_a_old_policy, q_hat)
        critic_b_loss = 0.5 * F.mse_loss(q_b_old_policy, q_hat)
        critic_loss = critic_a_loss + critic_b_loss
        critic_loss.backward()
        self.critic_a.optimizer.step()
        self.critic_b.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.soft_update(self.target_value, self.value, tau)

    def save_models(self):
        print("Saving Models...")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.critic_a.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_b.save_checkpoint()

    def load_models(self):
        print("Loading Models...")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.critic_a.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_b.load_checkpoint()
