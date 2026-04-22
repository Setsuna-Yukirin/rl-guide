"""
DDPG 和 TD3 算法

深度确定性策略梯度和 Twin Delayed DDPG
用于连续动作空间控制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """Actor 网络（策略）"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 action_bounds: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.action_bounds = action_bounds
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # 输出范围 [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.network(state)
        # 缩放到实际动作范围
        low, high = self.action_bounds
        action = (high - low) / 2 * action + (high + low) / 2
        return action


class Critic(nn.Module):
    """Critic 网络（Q 函数）"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        
        # Q(s, a)
        layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class DDPGAgent:
    """
    DDPG (Deep Deterministic Policy Gradient) 智能体
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,  # 软更新系数
        buffer_size: int = 10000,
        batch_size: int = 64,
        noise_std: float = 0.1,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Actor
        self.actor = Actor(state_dim, action_dim, action_bounds=action_bounds).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, action_bounds=action_bounds).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 噪声（Ornstein-Uhlenbeck）
        self.noise = np.zeros(action_dim)
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """选择动作（带噪声探索）"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy().squeeze(0)
        
        if explore:
            # 添加探索噪声
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
            # 裁剪到动作范围
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, target: nn.Module, source: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def update(self) -> Tuple[float, float]:
        """更新 Actor 和 Critic"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Critic 更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor 更新
        actor_actions = self.actor(states)
        actor_q = self.critic(states, actor_actions)
        actor_loss = -actor_q.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        return actor_loss.item(), critic_loss.item()
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """训练一个 episode"""
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.store_transition(state, action, reward, next_state, done)
            self.update()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        return total_reward, steps


class TD3Agent(DDPGAgent):
    """
    TD3 (Twin Delayed DDPG) 智能体
    
    DDPG 的改进版本：
    1. Clipped Double Q-Learning
    2. Delayed Policy Updates
    3. Target Policy Smoothing
    """
    
    def __init__(self, *args, policy_delay: int = 2, noise_clip: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        self.total_steps = 0
        
        # 第二个 Critic
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)
    
    def update(self) -> Tuple[float, float]:
        """TD3 更新"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        self.total_steps += 1
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Critic 更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            
            # Target Policy Smoothing
            noise = torch.randn_like(next_actions) * self.noise_std
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(
                self.action_bounds[0], self.action_bounds[1]
            )
            
            # Clipped Double Q-Learning
            next_q1 = self.critic_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 更新两个 Critic
        current_q1 = self.critic(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Delayed Policy Update
        if self.total_steps % self.policy_delay == 0:
            actor_actions = self.actor(states)
            actor_q = self.critic(states, actor_actions)
            actor_loss = -actor_q.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
            self.soft_update(self.critic2_target, self.critic2)
            
            return actor_loss.item(), (critic1_loss.item() + critic2_loss.item()) / 2
        
        return 0.0, (critic1_loss.item() + critic2_loss.item()) / 2
