"""
Actor-Critic 和 A2C 算法

Actor-Critic 基础架构和 Advantage Actor-Critic 实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络
    
    共享特征提取层，分离策略和价值输出
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (64, 64)
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        
        # 共享特征层
        feature_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*feature_layers)
        
        # Actor（策略）头
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic（价值）头
        self.critic = nn.Linear(prev_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_probs: 动作概率
            value: 状态价值
        """
        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class ActorCriticAgent:
    """
    Actor-Critic 智能体
    
    使用 TD 误差更新
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        hidden_dims: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            hidden_dims: 隐藏层维度
            device: 计算设备
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Actor-Critic 网络
        self.network = ActorCriticNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.network.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.network.critic.parameters(), lr=critic_lr)
        
        # 轨迹
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
    
    def get_action(self, state: np.ndarray) -> int:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.network(state_tensor)
            
            # 采样动作
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            self.log_probs.append(log_prob)
            self.values.append(value)
            
            return int(action.item())
    
    def update(self, next_state: np.ndarray, done: bool) -> float:
        """
        更新 Actor 和 Critic
        
        Args:
            next_state: 下一状态
            done: 是否终止
        
        Returns:
            损失值
        """
        # 计算下一状态价值
        with torch.no_grad():
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.network(state_tensor)
            next_value = 0.0 if done else next_value.squeeze(0)
        
        # 计算 TD 误差
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        values = torch.cat(self.values)
        
        # 计算回报
        returns = []
        R = next_value
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # TD 误差（优势）
        advantages = returns - values
        
        # Actor 损失
        actor_loss = []
        for log_prob, A in zip(self.log_probs, advantages):
            actor_loss.append(-log_prob * A)
        actor_loss = torch.cat(actor_loss).sum()
        
        # Critic 损失
        critic_loss = nn.functional.mse_loss(values, returns)
        
        # 反向传播
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 清空轨迹
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return (actor_loss + critic_loss).item()


class A2CAgent:
    """
    A2C (Advantage Actor-Critic) 智能体
    
    使用 n 步回报估计优势
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        hidden_dims: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            n_steps: n 步回报的 n
            entropy_coef: 熵正则化系数
            value_loss_coef: 价值损失系数
            hidden_dims: 隐藏层维度
            device: 计算设备
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Actor-Critic 网络
        self.network = ActorCriticNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 轨迹存储
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
    
    def get_action(self, state: np.ndarray) -> int:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.network(state_tensor)
            
            # 采样动作
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            self.states.append(state)
            self.actions.append(int(action.item()))
            self.log_probs.append(log_prob)
            self.values.append(value.squeeze(0))
            
            return int(action.item())
    
    def store_reward(self, reward: float):
        """存储奖励"""
        self.rewards.append(reward)
    
    def compute_nstep_returns(self, next_value: torch.Tensor) -> torch.Tensor:
        """
        计算 n 步回报
        
        Args:
            next_value: n 步后的价值估计
        
        Returns:
            n 步回报
        """
        returns = []
        R = next_value
        
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def update(self, next_state: np.ndarray, done: bool) -> Tuple[float, float, float]:
        """
        更新网络
        
        Args:
            next_state: 下一状态
            done: 是否终止
        
        Returns:
            actor_loss, critic_loss, entropy_loss
        """
        # 计算下一状态价值
        with torch.no_grad():
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.network(state_tensor)
            next_value = 0.0 if done else next_value.squeeze(0)
        
        # 计算 n 步回报
        returns = self.compute_nstep_returns(next_value)
        values = torch.stack(self.values)
        
        # 优势
        advantages = returns - values
        
        # Actor 损失（策略梯度）
        log_probs = torch.stack(self.log_probs)
        actor_loss = -(log_probs * advantages.detach()).sum()
        
        # Critic 损失
        critic_loss = nn.functional.mse_loss(values, returns)
        
        # 熵正则化（鼓励探索）
        action_probs, _ = self.network(torch.FloatTensor(np.array(self.states)).to(self.device))
        dist = torch.distributions.Categorical(action_probs)
        entropy = dist.entropy().mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # 总损失
        total_loss = actor_loss + self.value_loss_coef * critic_loss + entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
        return actor_loss.item(), critic_loss.item(), entropy.item()
    
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
            
            self.store_reward(reward)
            
            # 每 n 步更新一次
            if (steps + 1) % self.n_steps == 0 or done:
                self.update(next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        return total_reward, steps
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = True) -> list:
        """训练多个 episode"""
        rewards = []
        
        for episode in range(n_episodes):
            total_reward, steps = self.train_episode(env)
            rewards.append(total_reward)
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}")
        
        return rewards
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
