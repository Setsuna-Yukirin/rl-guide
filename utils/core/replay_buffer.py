"""
经验回放缓冲区 (Replay Buffer)

用于存储和采样经验数据，是 DQN 等算法的核心组件。

支持：
- 均匀随机采样
- 优先级采样（待实现）
- 批量存储

Reference:
    Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from collections import deque


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储经验数据 (state, action, reward, next_state, done) 并支持批量采样。
    
    Attributes:
        capacity: 缓冲区容量
        buffer: 存储经验的 deque
    
    Example:
        >>> from utils.core import ReplayBuffer
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.add(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
    """
    
    def __init__(self, capacity: int):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否终止
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(
        self,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        随机采样一批经验
        
        Args:
            batch_size: 批量大小
            seed: 随机种子（可选）
            
        Returns:
            states: 状态数组 (batch_size, state_dim)
            actions: 动作数组 (batch_size,)
            rewards: 奖励数组 (batch_size,)
            next_states: 下一状态数组 (batch_size, state_dim)
            dones: 终止标志数组 (batch_size,)
            
        Raises:
            ValueError: 当缓冲区经验不足时
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"缓冲区经验数量 ({len(self.buffer)}) 小于批量大小 ({batch_size})"
            )
        
        if seed is not None:
            np.random.seed(seed)
        
        # 随机采样索引
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # 批量获取经验
        batch = [self.buffer[idx] for idx in indices]
        
        # 分离为各个组件
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为 numpy 数组
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """获取缓冲区当前经验数量"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0
    
    def save(self, path: str):
        """
        保存缓冲区到文件
        
        Args:
            path: 保存路径
        """
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump({
                'capacity': self.capacity,
                'buffer': list(self.buffer),
            }, f)
    
    def load(self, path: str):
        """
        从文件加载缓冲区
        
        Args:
            path: 加载路径
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data['buffer'], maxlen=self.capacity)


class PriorityReplayBuffer:
    """
    优先级经验回放缓冲区 (Prioritized Experience Replay)
    
    根据 TD 误差给经验分配优先级，优先采样高 TD 误差的经验。
    
    Reference:
        Schaul, T., et al. (2016). Prioritized Experience Replay. ICLR.
    
    Note:
        简化实现，使用 SumTree 的完整版本可以参考 DQN 实现
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        初始化优先级回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0 = 均匀采样，1 = 完全按优先级)
            beta: 重要性采样权重指数
            beta_increment: beta 每步增量
            epsilon: 小常数，避免优先级为 0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 使用简单列表实现（SumTree 更高效但更复杂）
        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        添加经验到缓冲区
        
        新经验使用最大优先级
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否终止
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority ** self.alpha)
        else:
            # 替换最早的经验
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = experience
            self.priorities[idx] = self.max_priority ** self.alpha
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        np.ndarray,
        np.ndarray,
    ]:
        """
        按优先级采样
        
        Args:
            batch_size: 批量大小
            
        Returns:
            batch: 经验元组 (states, actions, rewards, next_states, dones)
            indices: 采样索引
            weights: 重要性采样权重
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"缓冲区经验数量 ({len(self.buffer)}) 小于批量大小 ({batch_size})"
            )
        
        # 计算采样概率
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities / priorities.sum()
        
        # 按概率采样
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probabilities,
            replace=False,
        )
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 增加 beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取批量数据
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新经验优先级
        
        Args:
            indices: 经验索引
            priorities: 新优先级（通常是 TD 误差的绝对值）
        """
        # 更新优先级
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + self.epsilon) ** self.alpha
        
        # 更新最大优先级
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        """获取缓冲区当前经验数量"""
        return len(self.buffer)


if __name__ == "__main__":
    # 简单测试
    print("测试 ReplayBuffer 类...\n")
    
    # 测试基础回放缓冲区
    print("1. 测试 ReplayBuffer")
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加经验
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.rand() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"   缓冲区大小：{len(buffer)}")
    assert len(buffer) == 100, "缓冲区应该有 100 条经验"
    
    # 采样
    batch = buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = batch
    
    print(f"   采样批量大小：{states.shape[0]}")
    assert states.shape == (32, 4), "状态批量形状错误"
    assert actions.shape == (32,), "动作批量形状错误"
    assert rewards.shape == (32,), "奖励批量形状错误"
    
    print(f"   ✓ ReplayBuffer 测试通过\n")
    
    # 测试优先级回放缓冲区
    print("2. 测试 PriorityReplayBuffer")
    pri_buffer = PriorityReplayBuffer(capacity=1000)
    
    # 添加经验
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.rand() < 0.1
        
        pri_buffer.add(state, action, reward, next_state, done)
    
    print(f"   缓冲区大小：{len(pri_buffer)}")
    
    # 采样
    batch, indices, weights = pri_buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = batch
    
    print(f"   采样批量大小：{states.shape[0]}")
    print(f"   权重范围：[{weights.min():.4f}, {weights.max():.4f}]")
    
    # 更新优先级
    td_errors = np.abs(np.random.randn(32))
    pri_buffer.update_priorities(indices, td_errors)
    
    print(f"   ✓ PriorityReplayBuffer 测试通过\n")
    
    print("✅ 所有 ReplayBuffer 测试通过！")
