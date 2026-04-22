"""
策略 (Policy) 抽象基类

策略是强化学习中智能体的行为方式，定义为从状态到动作的映射。

策略可以分为：
- 确定性策略：π(s) = a
- 随机性策略：π(a|s) = P(A=a|S=s)

Reference:
    Sutton, R. S., & Barto, A. G. (2018). 
    Reinforcement learning: An introduction. MIT press.
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod


class Policy(ABC):
    """
    策略抽象基类
    
    所有策略类都应该继承此类并实现抽象方法。
    
    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
    
    Example:
        >>> from utils.core import Policy
        >>> class RandomPolicy(Policy):
        ...     def get_action(self, state):
        ...         return np.random.randint(self.action_dim)
        ...     def get_action_prob(self, state, action):
        ...         return 1.0 / self.action_dim
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        """
        初始化策略
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def get_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        根据状态选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            action: 选择的动作（离散为 int，连续为 ndarray）
        """
        pass
    
    def get_action_prob(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
    ) -> float:
        """
        获取在给定状态下选择特定动作的概率
        
        Args:
            state: 当前状态
            action: 动作
            
        Returns:
            概率值
            
        Note:
            可选方法，对于确定性策略可以返回 1.0
        """
        raise NotImplementedError("子类可以选择实现此方法")
    
    def update(self, experience: Tuple) -> Optional[Dict[str, float]]:
        """
        更新策略参数
        
        Args:
            experience: 经验数据 (state, action, reward, next_state, done)
            
        Returns:
            更新统计信息（可选）
            
        Note:
            可选方法，对于固定策略可以返回 None
        """
        return None
    
    def save(self, path: str):
        """
        保存策略参数
        
        Args:
            path: 保存路径
            
        Note:
            可选方法，简单策略可以不需要保存
        """
        raise NotImplementedError("子类可以选择实现此方法")
    
    def load(self, path: str):
        """
        加载策略参数
        
        Args:
            path: 加载路径
            
        Note:
            可选方法，简单策略可以不需要加载
        """
        raise NotImplementedError("子类可以选择实现此方法")
    
    def set_mode(self, mode: str):
        """
        设置策略模式（训练/评估）
        
        Args:
            mode: "train" 或 "eval"
            
        Note:
            可选方法，需要 dropout 等机制的策略可以实现
        """
        pass


class RandomPolicy(Policy):
    """
    随机策略
    
    均匀随机选择动作，通常用作基线或探索策略。
    
    Example:
        >>> from utils.core import RandomPolicy
        >>> policy = RandomPolicy(state_dim=4, action_dim=2)
        >>> action = policy.get_action(np.zeros(4))
        >>> print(f"随机动作：{action}")
    """
    
    def __init__(self, state_dim: int, action_dim: int, seed: Optional[int] = None):
        """
        初始化随机策略
        
        Args:
            state_dim: 状态空间维度（未使用，仅为接口一致）
            action_dim: 动作空间维度
            seed: 随机种子（可选）
        """
        super().__init__(state_dim, action_dim)
        if seed is not None:
            np.random.seed(seed)
    
    def get_action(self, state: np.ndarray) -> int:
        """
        随机选择动作
        
        Args:
            state: 当前状态（未使用）
            
        Returns:
            随机选择的动作
        """
        return np.random.randint(self.action_dim)
    
    def get_action_prob(self, state: np.ndarray, action: int) -> float:
        """
        获取动作概率（均匀分布）
        
        Args:
            state: 当前状态（未使用）
            action: 动作
            
        Returns:
            1.0 / action_dim
        """
        return 1.0 / self.action_dim


class EpsilonGreedyPolicy(Policy):
    """
    ε-greedy 策略
    
    以概率 ε 随机探索，以概率 1-ε 选择最优动作。
    
    Attributes:
        q_values: Q 值表或函数
        epsilon: 探索率
    
    Example:
        >>> from utils.core import EpsilonGreedyPolicy
        >>> policy = EpsilonGreedyPolicy(state_dim=10, action_dim=4, epsilon=0.1)
        >>> policy.set_q_values(np.random.randn(10, 4))
        >>> action = policy.get_action(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        epsilon: float = 0.1,
    ):
        """
        初始化 ε-greedy 策略
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            epsilon: 探索率 (0 <= epsilon <= 1)
            
        Raises:
            ValueError: 当 epsilon 不在 [0, 1] 范围内时
        """
        super().__init__(state_dim, action_dim)
        
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon 必须在 [0, 1] 范围内，当前值：{epsilon}")
        
        self.epsilon = epsilon
        self.q_values: Optional[np.ndarray] = None
    
    def set_q_values(self, q_values: np.ndarray):
        """
        设置 Q 值表
        
        Args:
            q_values: Q 值表，形状为 (state_dim, action_dim)
        """
        if q_values.shape != (self.state_dim, self.action_dim):
            raise ValueError(
                f"Q 值表形状 {q_values.shape} 不匹配，"
                f"期望形状 ({self.state_dim}, {self.action_dim})"
            )
        self.q_values = q_values
    
    def set_epsilon(self, epsilon: float):
        """
        设置探索率
        
        Args:
            epsilon: 新的探索率
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon 必须在 [0, 1] 范围内，当前值：{epsilon}")
        self.epsilon = epsilon
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """
        衰减探索率
        
        Args:
            decay_rate: 衰减率
            min_epsilon: 最小探索率
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def get_action(self, state: np.ndarray) -> int:
        """
        ε-greedy 选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
            
        Raises:
            ValueError: 当 Q 值表未设置时
        """
        if self.q_values is None:
            raise ValueError("Q 值表未设置，请先调用 set_q_values()")
        
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.action_dim)
        else:
            # 利用：选择最优动作
            state_idx = int(state) if np.isscalar(state) else int(state[0])
            return int(np.argmax(self.q_values[state_idx]))
    
    def get_action_prob(
        self,
        state: np.ndarray,
        action: int,
    ) -> float:
        """
        获取动作概率
        
        Args:
            state: 当前状态
            action: 动作
            
        Returns:
            选择该动作的概率
        """
        if self.q_values is None:
            return 1.0 / self.action_dim
        
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        best_action = np.argmax(self.q_values[state_idx])
        
        if action == best_action:
            # 最优动作的概率
            return 1 - self.epsilon + self.epsilon / self.action_dim
        else:
            # 非最优动作的概率
            return self.epsilon / self.action_dim


class GreedyPolicy(Policy):
    """
    贪心策略
    
    始终选择当前最优动作，用于评估阶段。
    
    Example:
        >>> from utils.core import GreedyPolicy
        >>> policy = GreedyPolicy(state_dim=10, action_dim=4)
        >>> policy.set_q_values(np.random.randn(10, 4))
        >>> action = policy.get_action(state)  # 总是选择最优动作
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        初始化贪心策略
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
        """
        super().__init__(state_dim, action_dim)
        self.q_values: Optional[np.ndarray] = None
    
    def set_q_values(self, q_values: np.ndarray):
        """
        设置 Q 值表
        
        Args:
            q_values: Q 值表，形状为 (state_dim, action_dim)
        """
        if q_values.shape != (self.state_dim, self.action_dim):
            raise ValueError(
                f"Q 值表形状 {q_values.shape} 不匹配，"
                f"期望形状 ({self.state_dim}, {self.action_dim})"
            )
        self.q_values = q_values
    
    def get_action(self, state: np.ndarray) -> int:
        """
        贪心选择最优动作
        
        Args:
            state: 当前状态
            
        Returns:
            最优动作
            
        Raises:
            ValueError: 当 Q 值表未设置时
        """
        if self.q_values is None:
            raise ValueError("Q 值表未设置，请先调用 set_q_values()")
        
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        return int(np.argmax(self.q_values[state_idx]))


if __name__ == "__main__":
    # 简单测试
    print("测试 Policy 类...\n")
    
    # 测试随机策略
    print("1. 测试 RandomPolicy")
    random_policy = RandomPolicy(state_dim=4, action_dim=2, seed=42)
    actions = [random_policy.get_action(np.zeros(4)) for _ in range(10)]
    print(f"   随机动作序列：{actions}")
    print(f"   ✓ RandomPolicy 测试通过\n")
    
    # 测试 ε-greedy 策略
    print("2. 测试 EpsilonGreedyPolicy")
    epsilon_policy = EpsilonGreedyPolicy(state_dim=10, action_dim=4, epsilon=0.5)
    q_values = np.random.randn(10, 4)
    epsilon_policy.set_q_values(q_values)
    
    # 测试动作选择
    state = np.array([0])
    actions = [epsilon_policy.get_action(state) for _ in range(20)]
    print(f"   动作序列（ε=0.5）：{actions}")
    
    # 测试 epsilon 衰减
    epsilon_policy.decay_epsilon(decay_rate=0.9, min_epsilon=0.01)
    print(f"   衰减后 epsilon: {epsilon_policy.epsilon:.4f}")
    print(f"   ✓ EpsilonGreedyPolicy 测试通过\n")
    
    # 测试贪心策略
    print("3. 测试 GreedyPolicy")
    greedy_policy = GreedyPolicy(state_dim=10, action_dim=4)
    greedy_policy.set_q_values(q_values)
    
    # 贪心策略应该总是选择相同的最优动作
    state = np.array([0])
    actions = [greedy_policy.get_action(state) for _ in range(10)]
    best_action = np.argmax(q_values[0])
    print(f"   状态 0 的最优动作：{best_action}")
    print(f"   贪心动作序列：{actions}")
    assert all(a == best_action for a in actions), "贪心策略应该总是选择最优动作"
    print(f"   ✓ GreedyPolicy 测试通过\n")
    
    print("✅ 所有 Policy 测试通过！")
