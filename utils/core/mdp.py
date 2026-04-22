"""
MDP 核心类

实现马尔可夫决策过程 (Markov Decision Process) 的基础抽象。

MDP 由五元组 (S, A, P, R, γ) 定义：
- S: 状态空间
- A: 动作空间
- P: 状态转移概率
- R: 奖励函数
- γ: 折扣因子

Reference:
    Sutton, R. S., & Barto, A. G. (2018). 
    Reinforcement learning: An introduction. MIT press.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union, List, Any
from abc import ABC, abstractmethod


class MDP:
    """
    马尔可夫决策过程 (MDP) 基类
    
    MDP 是强化学习的数学框架，由以下组件构成：
    - 状态空间 (state space)
    - 动作空间 (action space)
    - 状态转移概率 (transition probability)
    - 奖励函数 (reward function)
    - 折扣因子 (discount factor)
    
    Attributes:
        state_space: 状态空间（Gymnasium Space 或维度）
        action_space: 动作空间（Gymnasium Space 或维度）
        gamma: 折扣因子 (0 < gamma <= 1)
    
    Example:
        >>> from utils.core import MDP
        >>> mdp = MDP(state_dim=10, action_dim=4, gamma=0.99)
        >>> print(mdp)
        MDP(state_dim=10, action_dim=4, gamma=0.99)
    """
    
    def __init__(
        self,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        state_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        gamma: float = 0.99,
    ):
        """
        初始化 MDP
        
        Args:
            state_dim: 状态空间维度（当未提供 state_space 时使用）
            action_dim: 动作空间维度（当未提供 action_space 时使用）
            state_space: Gymnasium 状态空间
            action_space: Gymnasium 动作空间
            gamma: 折扣因子，默认 0.99
            
        Raises:
            ValueError: 当 state_dim 和 state_space 都未提供时
            ValueError: 当 action_dim 和 action_space 都未提供时
            ValueError: 当 gamma 不在 (0, 1] 范围内时
        """
        # 验证 gamma
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma 必须在 (0, 1] 范围内，当前值：{gamma}")
        
        # 处理状态空间
        if state_space is not None:
            self.state_space = state_space
            self.state_dim = self._get_space_dim(state_space)
        elif state_dim is not None:
            self.state_dim = state_dim
            self.state_space = None
        else:
            raise ValueError("必须提供 state_dim 或 state_space")
        
        # 处理动作空间
        if action_space is not None:
            self.action_space = action_space
            self.action_dim = self._get_space_dim(action_space)
        elif action_dim is not None:
            self.action_dim = action_dim
            self.action_space = None
        else:
            raise ValueError("必须提供 action_dim 或 action_space")
        
        self.gamma = gamma
    
    def _get_space_dim(self, space: Any) -> int:
        """
        从 Gymnasium Space 获取维度
        
        Args:
            space: Gymnasium Space
            
        Returns:
            空间维度
        """
        from gymnasium import spaces
        
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.Box):
            if len(space.shape) == 1:
                return space.shape[0]
            else:
                return int(np.prod(space.shape))
        else:
            # 其他类型尝试直接获取
            return int(space.n) if hasattr(space, 'n') else 1
    
    def step(
        self,
        state: np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步状态转移
        
        Args:
            state: 当前状态
            action: 动作
            
        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否终止
            info: 额外信息
            
        Note:
            这是一个抽象方法，具体实现应该在子类中完成
        """
        raise NotImplementedError(
            "MDP.step() 是抽象方法，需要在子类中实现"
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态
        
        Args:
            seed: 随机种子（可选）
            
        Returns:
            state: 初始状态
            info: 额外信息
            
        Note:
            这是一个抽象方法，具体实现应该在子类中完成
        """
        raise NotImplementedError(
            "MDP.reset() 是抽象方法，需要在子类中完成"
        )
    
    def get_transition_prob(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """
        获取状态转移概率 P(s'|s, a)
        
        Args:
            state: 当前状态
            action: 动作
            next_state: 下一状态
            
        Returns:
            转移概率
            
        Note:
            默认返回 1.0（确定性转移），子类可以重写
        """
        return 1.0
    
    def get_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """
        获取奖励 R(s, a, s')
        
        Args:
            state: 当前状态
            action: 动作
            next_state: 下一状态
            
        Returns:
            奖励值
            
        Note:
            默认返回 0.0，子类应该重写
        """
        return 0.0
    
    def is_terminal(self, state: np.ndarray) -> bool:
        """
        判断状态是否为终止状态
        
        Args:
            state: 状态
            
        Returns:
            是否为终止状态
        """
        return False
    
    def render(self, mode: str = "human"):
        """
        渲染环境
        
        Args:
            mode: 渲染模式 ("human", "rgb_array", etc.)
            
        Note:
            可选方法，子类可以选择实现
        """
        pass
    
    def close(self):
        """关闭环境"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state_dim={self.state_dim}, action_dim={self.action_dim}, gamma={self.gamma})"


class TabularMDP(MDP):
    """
    表格型 MDP
    
    适用于状态和动作空间较小的离散 MDP。
    使用表格存储转移概率和奖励。
    
    Attributes:
        P: 转移概率字典 P[s][a] = [(prob, next_state, reward, done), ...]
        nS: 状态数量
        nA: 动作数量
    
    Example:
        >>> from utils.core import TabularMDP
        >>> mdp = TabularMDP(n_states=4, n_actions=2)
        >>> # 定义转移概率
        >>> mdp.P = {
        ...     0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, -1.0, False)]},
        ...     ...
        ... }
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
    ):
        """
        初始化表格型 MDP
        
        Args:
            n_states: 状态数量
            n_actions: 动作数量
            gamma: 折扣因子
        """
        super().__init__(state_dim=n_states, action_dim=n_actions, gamma=gamma)
        self.nS = n_states
        self.nA = n_actions
        self.P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
    
    def set_transition(
        self,
        state: int,
        action: int,
        transitions: List[Tuple[float, int, float, bool]],
    ):
        """
        设置状态转移概率
        
        Args:
            state: 当前状态
            action: 动作
            transitions: 转移列表 [(prob, next_state, reward, done), ...]
            
        Example:
            >>> mdp.set_transition(0, 0, [
            ...     (0.8, 1, -1.0, False),  # 80% 概率到状态 1
            ...     (0.2, 0, -1.0, False),  # 20% 概率留在状态 0
            ... ])
        """
        if state not in self.P:
            self.P[state] = {}
        
        # 验证概率和为 1
        prob_sum = sum(t[0] for t in transitions)
        if not np.isclose(prob_sum, 1.0):
            raise ValueError(
                f"状态 {state} 动作 {action} 的转移概率和为 {prob_sum}，应该为 1.0"
            )
        
        self.P[state][action] = transitions
    
    def step(
        self,
        state: int,
        action: int,
    ) -> Tuple[int, float, bool, Dict]:
        """
        执行一步状态转移
        
        Args:
            state: 当前状态（整数索引）
            action: 动作（整数索引）
            
        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否终止
            info: 额外信息
        """
        if state not in self.P or action not in self.P[state]:
            raise ValueError(f"状态 {state} 动作 {action} 的转移未定义")
        
        transitions = self.P[state][action]
        
        # 根据概率采样下一状态
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        _, next_state, reward, done = transitions[idx]
        
        info = {}
        return next_state, reward, done, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict]:
        """
        重置环境到随机初始状态
        
        Args:
            seed: 随机种子
            
        Returns:
            state: 初始状态（随机选择）
            info: 额外信息
        """
        if seed is not None:
            np.random.seed(seed)
        
        state = np.random.randint(self.nS)
        return state, {}
    
    def get_transition(self, state: int, action: int) -> List[Tuple[float, int, float, bool]]:
        """
        获取状态转移概率
        
        Args:
            state: 当前状态
            action: 动作
            
        Returns:
            [(prob, next_state, reward, done), ...]
        """
        if state in self.P and action in self.P[state]:
            return self.P[state][action]
        return []


if __name__ == "__main__":
    # 简单测试
    print("测试 MDP 类...")
    
    # 测试基础 MDP
    mdp = MDP(state_dim=10, action_dim=4, gamma=0.99)
    print(f"✓ MDP 创建成功：{mdp}")
    
    # 测试表格型 MDP
    tabular_mdp = TabularMDP(n_states=4, n_actions=2)
    print(f"✓ TabularMDP 创建成功：{tabular_mdp}")
    
    # 设置转移概率
    tabular_mdp.set_transition(0, 0, [
        (0.8, 1, -1.0, False),
        (0.2, 0, -1.0, False),
    ])
    print("✓ 转移概率设置成功")
    
    # 测试 step
    state, reward, done, info = tabular_mdp.step(0, 0)
    print(f"✓ Step 测试成功：state={state}, reward={reward}, done={done}")
    
    # 测试 reset
    state, info = tabular_mdp.reset()
    print(f"✓ Reset 测试成功：state={state}")
    
    print("\n✅ 所有测试通过！")
