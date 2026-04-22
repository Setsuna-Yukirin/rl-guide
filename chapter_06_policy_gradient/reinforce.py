"""
REINFORCE 算法

蒙特卡洛策略梯度算法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List


class PolicyNetwork(nn.Module):
    """
    策略网络
    
    输出动作的概率分布
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
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_actions))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 (batch, state_dim)
        
        Returns:
            动作概率 (batch, n_actions)
        """
        return self.network(state)
    
    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: 状态
        
        Returns:
            action: 动作索引
            log_prob: 动作的对数概率
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.forward(state_tensor)
            
            # 采样动作
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return int(action.item()), log_prob


class GaussianPolicyNetwork(nn.Module):
    """
    高斯策略网络（用于连续动作空间）
    
    输出动作的均值和标准差
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        log_std_init: float = 0.0
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            log_std_init: 初始对数标准差
        """
        super().__init__()
        
        self.action_dim = action_dim
        
        # 均值网络
        mean_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            mean_layers.append(nn.Linear(prev_dim, hidden_dim))
            mean_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        mean_layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mean_network = nn.Sequential(*mean_layers)
        
        # 对数标准差（可学习参数）
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            mean: 动作均值
            std: 动作标准差
        """
        mean = self.mean_network(state)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: 状态
        
        Returns:
            action: 动作
            log_prob: 对数概率
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, std = self.forward(state_tensor)
            
            # 采样
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action.numpy().squeeze(0), log_prob


class REINFORCEAgent:
    """
    REINFORCE 智能体
    
    蒙特卡洛策略梯度算法
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        hidden_dims: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            learning_rate: 学习率
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
        
        # 策略网络
        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 轨迹存储
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
    
    def get_action(self, state: np.ndarray) -> int:
        """选择动作"""
        action, log_prob = self.policy.get_action(state)
        self.log_probs.append(log_prob)
        return action
    
    def store_reward(self, reward: float):
        """存储奖励"""
        self.rewards.append(reward)
    
    def compute_returns(self) -> torch.Tensor:
        """
        计算折扣回报
        
        Returns:
            折扣回报张量
        """
        returns = []
        R = 0
        
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # 标准化（减少方差）
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self) -> float:
        """
        更新策略
        
        Returns:
            损失值
        """
        if len(self.log_probs) == 0:
            return 0.0
        
        # 计算回报
        returns = self.compute_returns()
        
        # 策略梯度损失
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.cat(policy_loss).sum()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 清空轨迹
        self.log_probs = []
        self.rewards = []
        
        return loss.item()
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """
        训练一个 episode
        
        Returns:
            total_reward: 总奖励
            steps: 步数
        """
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.store_reward(reward)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # 更新
        loss = self.update()
        
        return total_reward, steps
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = True) -> list:
        """
        训练多个 episode
        
        Returns:
            rewards: 每个 episode 的奖励
        """
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
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class REINFORCEBaseline(REINFORCEAgent):
    """
    REINFORCE with Baseline
    
    使用价值函数作为基线减少方差
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 价值网络（基线）
        self.value_network = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.01)
        self.values: List[torch.Tensor] = []
    
    def get_action(self, state: np.ndarray) -> int:
        """选择动作并存储价值"""
        action, log_prob = self.policy.get_action(state)
        self.log_probs.append(log_prob)
        
        # 计算价值（基线）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.value_network(state_tensor)
            self.values.append(value)
        
        return action
    
    def compute_advantages(self) -> torch.Tensor:
        """
        计算优势函数 A = G - V(s)
        
        Returns:
            优势张量
        """
        returns = []
        R = 0
        
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).unsqueeze(1)
        values = torch.cat(self.values)
        
        # 优势 = 回报 - 价值
        advantages = returns - values
        
        # 标准化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self) -> float:
        """更新策略和价值网络"""
        if len(self.log_probs) == 0:
            return 0.0
        
        # 计算优势
        advantages = self.compute_advantages()
        
        # 策略损失
        policy_loss = []
        for log_prob, A in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * A)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # 价值损失
        returns = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        value_loss = nn.functional.mse_loss(torch.cat(self.values), returns)
        
        # 总损失
        loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.value_optimizer.step()
        
        # 清空
        self.log_probs = []
        self.rewards = []
        self.values = []
        
        return loss.item()
