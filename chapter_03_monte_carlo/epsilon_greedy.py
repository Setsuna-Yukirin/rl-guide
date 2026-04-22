"""
ε-贪婪策略
"""

import numpy as np
from typing import Dict, Optional


class EpsilonGreedyPolicy:
    """
    ε-贪婪策略
    
    以概率 1-ε 选择最优动作，以概率 ε 随机选择
    """
    
    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        Q: Optional[Dict] = None
    ):
        """
        Args:
            n_actions: 动作数量
            epsilon: 探索概率 (0-1)
            Q: 动作价值函数 {state: [Q(s,0), Q(s,1), ...]}
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.Q = Q if Q is not None else {}
    
    def set_epsilon(self, epsilon: float):
        """设置 ε 值"""
        self.epsilon = epsilon
    
    def decay_epsilon(self, decay_rate: float = 0.99, min_epsilon: float = 0.01):
        """
        衰减 ε 值
        
        Args:
            decay_rate: 衰减率
            min_epsilon: 最小 ε 值
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def get_action(self, state, explore: bool = True) -> int:
        """
        根据策略选择动作
        
        Args:
            state: 当前状态
            explore: 是否进行探索
        
        Returns:
            action: 选择的动作
        """
        if explore and np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择最优动作
            if state in self.Q:
                q_values = self.Q[state]
                # 处理多个最优动作的情况
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                return np.random.choice(best_actions)
            else:
                # 未见过的状态，随机选择
                return np.random.randint(self.n_actions)
    
    def get_action_prob(self, state: int, action: int) -> float:
        """
        获取选择某个动作的概率
        
        Args:
            state: 状态
            action: 动作
        
        Returns:
            概率
        """
        if state not in self.Q:
            return 1.0 / self.n_actions
        
        q_values = self.Q[state]
        best_action = np.argmax(q_values)
        
        if action == best_action:
            return 1.0 - self.epsilon + self.epsilon / self.n_actions
        else:
            return self.epsilon / self.n_actions


class GLIEPolicy:
    """
    GLIE (Greedy in the Limit with Infinite Exploration) 策略
    
    ε 随时间衰减，保证：
    1. 无限探索：所有动作被选择无限次
    2. 贪婪极限：最终收敛到贪婪策略
    """
    
    def __init__(self, n_actions: int, k0: float = 1.0):
        """
        Args:
            n_actions: 动作数量
            k0: 初始衰减常数
        """
        self.n_actions = n_actions
        self.k0 = k0
        self.k = 0  # 当前迭代次数
        self.Q = {}
    
    def set_Q(self, Q: Dict):
        """设置 Q 函数"""
        self.Q = Q
    
    def get_epsilon(self) -> float:
        """获取当前 ε 值"""
        return self.k0 / (self.k0 + self.k)
    
    def get_action(self, state) -> int:
        """
        选择动作
        
        Returns:
            action: 选择的动作
        """
        self.k += 1
        epsilon = self.get_epsilon()
        
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            if state in self.Q:
                q_values = self.Q[state]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                return np.random.choice(best_actions)
            else:
                return np.random.randint(self.n_actions)
