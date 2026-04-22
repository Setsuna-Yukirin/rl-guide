"""
DQN 算法及变体

Deep Q-Network 完整实现
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, Optional, Dict
import random


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    根据 TD 误差优先级采样
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: 容量
            alpha: 优先级指数 (0=随机，1=完全优先)
            beta: 重要性采样指数
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """存储经验（最大优先级）"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """按优先级采样"""
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights),
            indices
        )
    
    def update_priorities(self, indices: list, priorities: list):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    DQN 智能体
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dims: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            buffer_size: 回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
            hidden_dims: 隐藏层维度
            device: 计算设备
        """
        from chapter_05_function_approximation.neural_network_q import QNetwork, NeuralQFunction
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Q 网络
        self.q_network = QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练统计
        self.steps = 0
        self.episode_count = 0
    
    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标 Q 值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 损失
        loss = self.loss_fn(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        
        # 更新目标网络
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减 ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
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
            
            self.store_transition(state, action, reward, next_state, done)
            self.train_step()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        self.episode_count += 1
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
                print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, ε={self.epsilon:.3f}")
        
        return rewards
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN 智能体
    
    解决 Q-Learning 的最大值偏差
    """
    
    def train_step(self) -> float:
        """Double DQN 训练"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN：用当前网络选择，目标网络评估
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN 智能体
    
    使用 Dueling 网络架构
    """
    
    def __init__(self, *args, **kwargs):
        from chapter_05_function_approximation.neural_network_q import DuelingQNetwork
        
        super().__init__(*args, **kwargs)
        
        # 替换为 Dueling 网络
        self.q_network = DuelingQNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_network = DuelingQNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
