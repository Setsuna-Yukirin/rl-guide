"""
探索策略模块

实现多种探索策略，用于平衡探索与利用
"""

import numpy as np
from typing import Dict, Optional, List
from abc import ABC, abstractmethod


class ExplorationStrategy(ABC):
    """探索策略基类"""
    
    @abstractmethod
    def select_action(self, q_values: np.ndarray, state: int = None) -> int:
        """
        选择动作
        
        Args:
            q_values: Q 值数组
            state: 当前状态（可选）
        
        Returns:
            action: 选择的动作
        """
        pass
    
    @abstractmethod
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        更新策略内部状态
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置策略"""
        pass


class EpsilonGreedy(ExplorationStrategy):
    """
    ε-贪婪策略
    
    以概率 ε 随机探索，以概率 1-ε 选择最优动作
    """
    
    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01
    ):
        """
        Args:
            n_actions: 动作数量
            epsilon: 初始探索概率
            epsilon_decay: 衰减率（每步后 epsilon *= decay）
            epsilon_min: 最小探索概率
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def select_action(self, q_values: np.ndarray, state: int = None) -> int:
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # 处理多个最优动作的情况
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """衰减 ε"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset(self):
        """重置 ε"""
        self.epsilon = max(self.epsilon_min, self.epsilon / self.epsilon_decay)
    
    def set_epsilon(self, epsilon: float):
        """设置 ε 值"""
        self.epsilon = max(self.epsilon_min, epsilon)


class DecayEpsilonGreedy(EpsilonGreedy):
    """
    衰减 ε-贪婪策略
    
    ε 随训练步数线性或指数衰减
    """
    
    def __init__(
        self,
        n_actions: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_steps: int = 10000,
        decay_type: str = 'linear'  # 'linear' or 'exponential'
    ):
        """
        Args:
            n_actions: 动作数量
            epsilon_start: 初始 ε
            epsilon_end: 最终 ε
            decay_steps: 衰减步数
            decay_type: 衰减类型
        """
        super().__init__(n_actions, epsilon_start, 1.0, epsilon_end)
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.steps = 0
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """更新步数并衰减 ε"""
        self.steps += 1
        
        if self.decay_type == 'linear':
            # 线性衰减
            progress = min(1.0, self.steps / self.decay_steps)
            self.epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
        else:
            # 指数衰减
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
                -self.steps / self.decay_steps
            )
        
        self.epsilon = max(self.epsilon_end, self.epsilon)
    
    def reset(self):
        """重置"""
        self.steps = 0
        self.epsilon = self.epsilon_start


class BoltzmannExploration(ExplorationStrategy):
    """
    Boltzmann 探索（Softmax 策略）
    
    按概率选择动作，概率与 Q 值成正比：
    P(a) = exp(Q(s,a)/T) / Σ exp(Q(s,a')/T)
    
    温度 T 控制探索程度：
    - T 高：更随机
    - T 低：更贪婪
    """
    
    def __init__(
        self,
        n_actions: int,
        temperature: float = 1.0,
        temperature_decay: float = 0.995,
        temperature_min: float = 0.1
    ):
        """
        Args:
            n_actions: 动作数量
            temperature: 初始温度
            temperature_decay: 温度衰减率
            temperature_min: 最小温度
        """
        self.n_actions = n_actions
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
    
    def select_action(self, q_values: np.ndarray, state: int = None) -> int:
        """Softmax 动作选择"""
        # 防止数值溢出
        q_values = np.asarray(q_values, dtype=np.float64)
        q_values = q_values - np.max(q_values)
        
        # 计算 softmax 概率
        exp_q = np.exp(q_values / self.temperature)
        probs = exp_q / np.sum(exp_q)
        
        # 按概率采样
        return np.random.choice(self.n_actions, p=probs)
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """衰减温度"""
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
    
    def reset(self):
        """重置温度"""
        self.temperature = max(self.temperature_min, self.temperature / self.temperature_decay)


class UpperConfidenceBound(ExplorationStrategy):
    """
    UCB (Upper Confidence Bound) 探索
    
    选择具有最高不确定性上界的动作：
    UCB(a) = Q(s,a) + c * sqrt(ln(N) / N(a))
    
    其中：
    - N: 总访问次数
    - N(a): 动作 a 的访问次数
    - c: 探索常数
    """
    
    def __init__(
        self,
        n_actions: int,
        c: float = 2.0
    ):
        """
        Args:
            n_actions: 动作数量
            c: 探索常数
        """
        self.n_actions = n_actions
        self.c = c
        
        # 动作访问计数
        self.action_counts = np.zeros(n_actions)
        self.total_count = 0
    
    def select_action(self, q_values: np.ndarray, state: int = None) -> int:
        """UCB 动作选择"""
        self.total_count += 1
        
        # 如果还有未探索的动作，优先探索
        unexplored = np.where(self.action_counts == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)
        
        # 计算 UCB 值
        ucb_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            exploration_bonus = self.c * np.sqrt(np.log(self.total_count) / self.action_counts[a])
            ucb_values[a] = q_values[a] + exploration_bonus
        
        return np.argmax(ucb_values)
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """更新动作计数"""
        self.action_counts[action] += 1
    
    def reset(self):
        """重置计数"""
        self.action_counts = np.zeros(self.n_actions)
        self.total_count = 0


class ThompsonSampling(ExplorationStrategy):
    """
    Thompson Sampling 探索
    
    为每个动作维护一个后验分布，从后验采样并选择最优动作
    
    对于伯努利奖励：使用 Beta 分布
    对于连续奖励：使用正态分布
    """
    
    def __init__(
        self,
        n_actions: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        """
        Args:
            n_actions: 动作数量
            prior_alpha: Beta 分布先验 α
            prior_beta: Beta 分布先验 β
        """
        self.n_actions = n_actions
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Beta 分布参数（用于伯努利奖励）
        self.alpha = np.ones(n_actions) * prior_alpha
        self.beta = np.ones(n_actions) * prior_beta
        
        # 正态分布参数（用于连续奖励）
        self.means = np.zeros(n_actions)
        self.variances = np.ones(n_actions)
    
    def select_action(self, q_values: np.ndarray, state: int = None) -> int:
        """Thompson Sampling 动作选择"""
        # 从后验分布采样
        samples = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # 使用 Beta 分布采样（适用于归一化奖励）
            samples[a] = np.random.beta(self.alpha[a], self.beta[a])
        
        return np.argmax(samples)
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """更新后验分布"""
        # 归一化奖励到 [0, 1]
        normalized_reward = (reward + 10) / 20  # 假设奖励范围 [-10, 10]
        normalized_reward = np.clip(normalized_reward, 0, 1)
        
        # 更新 Beta 分布
        self.alpha[action] += normalized_reward
        self.beta[action] += 1 - normalized_reward
        
        # 更新正态分布（增量式）
        self.means[action] += (reward - self.means[action]) / self.variances[action]
        self.variances[action] += 1
    
    def reset(self):
        """重置后验分布"""
        self.alpha = np.ones(self.n_actions) * self.prior_alpha
        self.beta = np.ones(self.n_actions) * self.prior_beta
        self.means = np.zeros(self.n_actions)
        self.variances = np.ones(self.n_actions)


class NoisyNetwork(ExplorationStrategy):
    """
    噪声网络探索
    
    在 Q 值上添加噪声进行探索：
    Q_noisy(s,a) = Q(s,a) + ε * σ(s,a)
    
    适用于深度强化学习
    """
    
    def __init__(
        self,
        n_actions: int,
        noise_std: float = 0.1,
        noise_decay: float = 0.99
    ):
        """
        Args:
            n_actions: 动作数量
            noise_std: 噪声标准差
            noise_decay: 噪声衰减率
        """
        self.n_actions = n_actions
        self.noise_std = noise_std
        self.noise_decay = noise_decay
    
    def select_action(self, q_values: np.ndarray, state: int = None) -> int:
        """添加噪声后选择动作"""
        noise = np.random.normal(0, self.noise_std, size=q_values.shape)
        q_noisy = q_values + noise
        return np.argmax(q_noisy)
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """衰减噪声"""
        self.noise_std = max(0.01, self.noise_std * self.noise_decay)
    
    def reset(self):
        """重置噪声"""
        self.noise_std = max(0.01, self.noise_std / self.noise_decay)


def create_exploration_strategy(
    name: str,
    n_actions: int,
    **kwargs
) -> ExplorationStrategy:
    """
    工厂函数：创建探索策略
    
    Args:
        name: 策略名称 ('epsilon', 'boltzmann', 'ucb', 'thompson', 'noisy')
        n_actions: 动作数量
        **kwargs: 策略特定参数
    
    Returns:
        ExplorationStrategy: 探索策略实例
    
    Example:
        >>> strategy = create_exploration_strategy('epsilon', 4, epsilon=0.1)
        >>> action = strategy.select_action(q_values)
    """
    strategies = {
        'epsilon': EpsilonGreedy,
        'decay_epsilon': DecayEpsilonGreedy,
        'boltzmann': BoltzmannExploration,
        'ucb': UpperConfidenceBound,
        'thompson': ThompsonSampling,
        'noisy': NoisyNetwork,
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name](n_actions, **kwargs)


def compare_exploration_strategies(
    env,
    agent_class,
    n_episodes: int = 100,
    n_runs: int = 5
) -> Dict:
    """
    比较不同探索策略的性能
    
    Args:
        env: 环境
        agent_class: 智能体类（需要有 exploration_strategy 参数）
        n_episodes: episode 数量
        n_runs: 运行次数
    
    Returns:
        results: {strategy_name: avg_rewards}
    """
    strategies = ['epsilon', 'boltzmann', 'ucb', 'thompson']
    results = {}
    
    for strategy_name in strategies:
        all_rewards = []
        
        for _ in range(n_runs):
            strategy = create_exploration_strategy(
                strategy_name,
                n_actions=env.action_space.n
            )
            
            agent = agent_class(
                n_actions=env.action_space.n,
                exploration_strategy=strategy
            )
            
            rewards = agent.train(env, n_episodes=n_episodes)
            all_rewards.append(rewards)
        
        results[strategy_name] = np.mean(all_rewards, axis=0)
    
    return results
