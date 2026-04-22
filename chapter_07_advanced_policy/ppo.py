"""
PPO (Proximal Policy Optimization) 算法实现

PPO 是一种稳定的策略梯度算法，通过限制策略更新幅度来避免性能崩溃。

核心特性：
- Clipped Surrogate Objective：截断策略比率
- Generalized Advantage Estimation (GAE)：低方差优势估计
- Value Function Clipping：价值函数裁剪

参考文献：
Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO 配置"""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络
    
    输出：
    - 策略分布参数（离散：概率分布；连续：高斯分布均值和标准差）
    - 状态价值估计
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: Optional[int] = None,
        action_dim: Optional[int] = None,
        continuous: bool = False,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.continuous = continuous
        self.n_actions = n_actions
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor 头
        if continuous:
            # 连续动作：输出高斯分布的均值和标准差
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            # 离散动作：输出动作概率分布
            self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Critic 头
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.shared(state)
        
        if self.continuous:
            mean = self.actor_mean(features)
            logstd = self.actor_logstd.expand(mean.size(0), -1)
            return mean, logstd
        else:
            logits = self.actor(features)
            return logits
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        features = self.shared(state)
        return self.critic(features)
    
    def get_distribution(self, state: torch.Tensor):
        """获取策略分布"""
        if self.continuous:
            mean, logstd = self(state)
            std = torch.exp(logstd)
            return Normal(mean, std)
        else:
            logits = self(state)
            return Categorical(logits=logits)
    
    def get_action_logprob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作和对数概率
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if self.continuous:
            log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于 PPO 更新）
        
        Returns:
            log_prob: 动作的对数概率
            entropy: 分布熵
            value: 状态价值
        """
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action)
        
        if self.continuous:
            log_prob = log_prob.sum(-1, keepdim=True)
        
        entropy = dist.entropy().mean()
        value = self.get_value(state)
        
        return log_prob, entropy, value


class RolloutBuffer:
    """
    PPO 轨迹缓冲区
    
    存储一个 rollout 的经验，用于 PPO 更新
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """添加一步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def get(self) -> Dict[str, np.ndarray]:
        """获取所有数据"""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
        }
    
    def __len__(self) -> int:
        return len(self.states)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: 奖励序列 [T]
        values: 价值估计 [T+1] (包含最后状态的 value)
        dones: 终止标志 [T]
        gamma: 折扣因子
        gae_lambda: GAE lambda 参数
    
    Returns:
        advantages: 优势估计 [T]
        returns: 回报估计 [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE: A_t = δ_t + γ * λ * δ_{t+1} + ...
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # Returns = advantages + values
    returns = advantages + values[:T]
    
    return advantages, returns


class PPOAgent:
    """
    PPO 智能体
    
    支持离散和连续动作空间
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: Optional[int] = None,
        action_dim: Optional[int] = None,
        continuous: bool = False,
        config: Optional[PPOConfig] = None,
    ):
        self.config = config or PPOConfig()
        self.continuous = continuous
        self.n_actions = n_actions
        self.action_dim = action_dim
        
        # 创建网络
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            action_dim=action_dim,
            continuous=continuous,
        ).to(self.config.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.lr)
        
        # 轨迹缓冲区
        self.buffer = RolloutBuffer()
        
        # 训练统计
        self.total_updates = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略（取均值/最大概率）
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            if deterministic:
                if self.continuous:
                    mean, _ = self.network(state_tensor)
                    action = mean
                else:
                    logits = self.network(state_tensor)
                    action = torch.argmax(logits, dim=-1, keepdim=True)
                log_prob = torch.zeros(1)
            else:
                action, log_prob = self.network.get_action_logprob(state_tensor)
            
            value = self.network.get_value(state_tensor)
        
        action_np = action.cpu().numpy()[0]
        if not deterministic:
            if log_prob.dim() > 0:
                log_prob_np = float(log_prob.cpu().numpy().flatten()[0])
            else:
                log_prob_np = float(log_prob.cpu().numpy())
        else:
            log_prob_np = 0.0
        value_np = value.cpu().numpy()[0, 0]
        
        return action_np, log_prob_np, value_np
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """存储一步过渡"""
        self.buffer.add(state, action, reward, done, log_prob, value)
    
    def compute_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算优势函数和回报
        
        Args:
            last_value: 最后状态的价值估计
        
        Returns:
            advantages: 优势估计
            returns: 回报估计
        """
        data = self.buffer.get()
        
        # 添加最后状态的 value
        values = np.append(data['values'], last_value)
        
        # 计算 GAE
        advantages, returns = compute_gae(
            rewards=data['rewards'],
            values=values,
            dones=data['dones'],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return (
            torch.FloatTensor(advantages).to(self.config.device),
            torch.FloatTensor(returns).to(self.config.device),
        )
    
    def update(self) -> Dict[str, float]:
        """
        执行 PPO 更新
        
        Returns:
            stats: 训练统计信息
        """
        data = self.buffer.get()
        advantages, returns = self.compute_advantages(0.0)
        
        # 转换为 tensor
        states = torch.FloatTensor(data['states']).to(self.config.device)
        actions = torch.FloatTensor(data['actions']).to(self.config.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.config.device)
        
        # 打乱数据
        dataset_size = len(states)
        indices = np.random.permutation(dataset_size)
        
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'clip_fraction': 0.0,
        }
        
        # PPO 更新
        n_batches = max(1, dataset_size // self.config.batch_size)
        
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略
                log_probs, entropy, values = self.network.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 计算策略比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 截断策略损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失（裁剪）
                values_clipped = torch.clamp(
                    values,
                    batch_returns - self.config.clip_epsilon,
                    batch_returns + self.config.clip_epsilon,
                )
                value_loss1 = (values - batch_returns).pow(2)
                value_loss2 = (values_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # 统计
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
                stats['clip_fraction'] += (torch.abs(ratio - 1) > self.config.clip_epsilon).float().mean().item()
        
        # 平均统计
        n_updates = self.config.ppo_epochs * n_batches
        for key in stats:
            stats[key] /= n_updates
        
        stats['n_updates'] = n_updates
        self.total_updates += n_updates
        
        # 清空缓冲区
        self.buffer.clear()
        
        return stats
    
    def train(
        self,
        env,
        n_episodes: int = 100,
        max_steps: int = 1000,
        verbose: bool = True,
    ) -> List[float]:
        """
        训练 PPO 智能体
        
        Args:
            env: Gymnasium 环境
            n_episodes: 训练 episode 数
            max_steps: 每个 episode 最大步数
            verbose: 是否打印训练信息
        
        Returns:
            episode_rewards: 每个 episode 的总奖励
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # 选择动作
                action, log_prob, value = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                # 存储经验
                self.store_transition(state, action, reward, done, log_prob, value)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # 获取最后状态的价值（用于 GAE）
            _, _, last_value = self.select_action(state, deterministic=True)
            self.store_transition(state, np.zeros_like(action), 0, True, 0, last_value)
            
            # PPO 更新
            stats = self.update()
            
            episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Value Loss: {stats['value_loss']:.4f}")
        
        return episode_rewards
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_updates = checkpoint['total_updates']
