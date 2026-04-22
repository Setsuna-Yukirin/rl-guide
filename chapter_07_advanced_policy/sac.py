"""
SAC (Soft Actor-Critic) 算法实现

SAC 是一种 off-policy 最大熵强化学习算法，通过最大化期望回报和策略熵，
实现更好的探索和稳定性。

核心特性：
- Maximum Entropy：最大化期望回报 + 策略熵
- Off-Policy：可使用经验回放
- Twin Q-Networks：两个 Q 网络防止过估计
- Automatic Temperature Tuning：自动调整熵系数

参考文献：
Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
https://arxiv.org/abs/1801.01290
Haarnoja et al. "SAC Applications and Analysis" (2019)
https://arxiv.org/abs/1812.05905
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class SACConfig:
    """SAC 配置"""
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # 软更新系数
    batch_size: int = 256
    buffer_size: int = int(1e6)
    hidden_dim: int = 256
    target_entropy: Optional[float] = None  # 目标熵（None 则自动设置）
    alpha: float = 0.2  # 初始熵系数
    automatic_entropy_tuning: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储 (state, action, reward, next_state, done) 元组
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """随机采样批次"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    """
    Soft Q 网络
    
    输入：状态 + 动作
    输出：Q 值
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        return self.q_network(x)


class PolicyNetwork(nn.Module):
    """
    策略网络（高斯策略）
    
    输出：高斯分布的均值和对数标准差
    使用 Tanh 变换保证动作在有界范围内
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.action_bounds = action_bounds
        self.action_dim = action_dim
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.logstd_layer = nn.Linear(hidden_dim, action_dim)
        
        # 初始化对数标准差
        self.logstd_layer.weight.data.fill_(0)
        self.logstd_layer.bias.data.fill_(-0.5)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            mean: 高斯分布均值
            logstd: 对数标准差
        """
        x = self.policy_network(state)
        mean = self.mean_layer(x)
        logstd = self.logstd_layer(x)
        logstd = torch.clamp(logstd, -20, 2)
        return mean, logstd
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作
        
        Args:
            state: 状态
            deterministic: 是否使用确定性策略（取均值）
        
        Returns:
            action: 动作（Tanh 变换后）
            log_prob: 对数概率
        """
        mean, logstd = self(state)
        std = torch.exp(logstd)
        
        if deterministic:
            action = torch.tanh(mean)
            # 确定性策略的对数概率为 0
            log_prob = torch.zeros(mean.size(0), 1, device=mean.device)
        else:
            # 使用 Tanh 变换的高斯分布
            dist = TransformedDistribution(
                Normal(mean, std),
                TanhTransform(cache_size=1),
            )
            action = dist.rsample()  # 重参数化采样
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            
            # 修正 Tanh 变换的雅可比行列式
            log_prob = log_prob - (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(-1, keepdim=True)
        
        # 缩放到动作范围
        low, high = self.action_bounds
        action = low + (action + 1) * 0.5 * (high - low)
        action = torch.clamp(action, low, high)
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算给定动作的对数概率（用于策略评估）
        
        Args:
            state: 状态
            action: 动作（已缩放）
        
        Returns:
            log_prob: 对数概率
        """
        # 将动作反变换到 [-1, 1] 范围
        low, high = self.action_bounds
        action_normalized = (action - low) / (high - low) * 2 - 1
        action_normalized = torch.clamp(action_normalized, -0.9999, 0.9999)
        
        # 反 Tanh
        pre_tanh = torch.atanh(action_normalized)
        
        mean, logstd = self(state)
        std = torch.exp(logstd)
        
        # 计算对数概率
        dist = Normal(mean, std)
        log_prob = dist.log_prob(pre_tanh).sum(-1, keepdim=True)
        
        # 修正 Tanh 变换
        log_prob = log_prob - (2 * (np.log(2) - action_normalized - F.softplus(-2 * action_normalized))).sum(-1, keepdim=True)
        
        return log_prob


class SACAgent:
    """
    SAC 智能体
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        config: Optional[SACConfig] = None,
    ):
        self.config = config or SACConfig()
        self.action_bounds = action_bounds
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # 创建网络
        self.q_network1 = SoftQNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.config.device)
        self.q_network2 = SoftQNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.config.device)
        self.target_q_network1 = SoftQNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.config.device)
        self.target_q_network2 = SoftQNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.config.device)
        self.policy_network = PolicyNetwork(state_dim, action_dim, self.config.hidden_dim, action_bounds).to(self.config.device)
        
        # 复制权重到目标网络
        self._soft_update(1.0)
        
        # 优化器
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=self.config.lr
        )
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.lr)
        
        # 自动熵调节
        if self.config.automatic_entropy_tuning:
            if self.config.target_entropy is None:
                self.config.target_entropy = -action_dim  # 启发式设置
            self.log_alpha = torch.tensor(np.log(self.config.alpha), requires_grad=True, device=self.config.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.config.alpha), device=self.config.device)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # 训练统计
        self.total_updates = 0
    
    def _soft_update(self, tau: float):
        """软更新目标网络"""
        for target_param, param in zip(self.target_q_network1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
        for target_param, param in zip(self.target_q_network2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            action, _ = self.policy_network.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        执行 SAC 更新
        
        Returns:
            stats: 训练统计信息
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # 采样批次
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor(batch['states']).to(self.config.device)
        actions = torch.FloatTensor(batch['actions']).to(self.config.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.config.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.config.device)
        
        # ==================== 更新 Q 网络 ====================
        
        # 计算目标 Q 值
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.get_action(next_states)
            q1_target = self.target_q_network1(next_states, next_actions)
            q2_target = self.target_q_network2(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            # 最大熵目标
            alpha = self.log_alpha.exp()
            q_target = rewards + self.config.gamma * (1 - dones) * (min_q_target - alpha * next_log_probs)
        
        # 计算当前 Q 值
        q1_current = self.q_network1(states, actions)
        q2_current = self.q_network2(states, actions)
        
        # Q 网络损失
        q1_loss = F.mse_loss(q1_current, q_target)
        q2_loss = F.mse_loss(q2_current, q_target)
        q_loss = q1_loss + q2_loss
        
        # 优化 Q 网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ==================== 更新策略网络 ====================
        
        # 采样新动作
        new_actions, new_log_probs = self.policy_network.get_action(states)
        
        # 计算策略损失（最大化熵 + Q 值）
        q1_new = self.q_network1(states, new_actions)
        q2_new = self.q_network2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * new_log_probs - min_q_new).mean()
        
        # 优化策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ==================== 自动熵调节 ====================
        
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_log_probs + self.config.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ==================== 软更新目标网络 ====================
        
        self._soft_update(self.config.tau)
        
        self.total_updates += 1
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': alpha.item() if self.config.automatic_entropy_tuning else self.config.alpha,
            'mean_q': q1_current.mean().item(),
        }
    
    def train(
        self,
        env,
        n_episodes: int = 100,
        max_steps: int = 1000,
        warmup_steps: int = 1000,
        gradient_steps: int = 1,
        verbose: bool = True,
    ) -> List[float]:
        """
        训练 SAC 智能体
        
        Args:
            env: Gymnasium 环境
            n_episodes: 训练 episode 数
            max_steps: 每个 episode 最大步数
            warmup_steps: 随机探索步数
            gradient_steps: 每次更新的梯度步数
            verbose: 是否打印训练信息
        
        Returns:
            episode_rewards: 每个 episode 的总奖励
        """
        episode_rewards = []
        total_steps = 0
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # 随机探索或选择动作
                if total_steps < warmup_steps:
                    low, high = self.action_bounds
                    action = np.random.uniform(low, high, self.action_dim)
                else:
                    action = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                # 存储经验
                self.store_transition(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                # 更新
                if total_steps >= warmup_steps:
                    for _ in range(gradient_steps):
                        stats = self.update()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Steps: {total_steps} | "
                      f"Buffer Size: {len(self.replay_buffer)}")
        
        return episode_rewards
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q_network1': self.target_q_network1.state_dict(),
            'target_q_network2': self.target_q_network2.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_updates': self.total_updates,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.q_network1.load_state_dict(checkpoint['q_network1'])
        self.q_network2.load_state_dict(checkpoint['q_network2'])
        self.target_q_network1.load_state_dict(checkpoint['target_q_network1'])
        self.target_q_network2.load_state_dict(checkpoint['target_q_network2'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha'].to(self.config.device)
        self.total_updates = checkpoint['total_updates']
