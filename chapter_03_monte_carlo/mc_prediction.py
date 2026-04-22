"""
蒙特卡洛预测算法

用于估计给定策略的价值函数 V_π 和 Q_π
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def generate_episode(env, policy, max_steps: int = 1000) -> List[Tuple]:
    """
    根据策略生成一个完整的 episode
    
    Args:
        env: 环境（需要有 reset, step 方法）
        policy: 策略函数 π(s) -> a
        max_steps: 最大步数
    
    Returns:
        episode: [(state, action, reward), ...] 列表
    """
    episode = []
    state, _ = env.reset()
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and len(episode) < max_steps:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    
    return episode


def compute_return(episode: List[Tuple], t: int, gamma: float = 1.0) -> float:
    """
    计算从时刻 t 开始的折扣回报 G_t
    
    G_t = R_{t+1} + γ*R_{t+2} + γ²*R_{t+3} + ...
    
    Args:
        episode: episode 列表
        t: 起始时刻
        gamma: 折扣因子
    
    Returns:
        G_t: 折扣回报
    """
    G = 0.0
    for i in range(t, len(episode)):
        _, _, reward = episode[i]
        G += (gamma ** (i - t)) * reward
    return G


def first_visit_mc_prediction(
    env,
    policy,
    n_episodes: int = 10000,
    gamma: float = 1.0
) -> Dict:
    """
    第一访问蒙特卡洛预测
    
    只统计每个状态在 episode 中第一次访问时的回报
    
    Args:
        env: 环境
        policy: 策略函数
        n_episodes: episode 数量
        gamma: 折扣因子
    
    Returns:
        V: 状态价值函数字典 {state: value}
        returns: 每个状态的回报列表 {state: [G1, G2, ...]}
    """
    # 存储每个状态的回报
    returns = defaultdict(list)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy, max_steps=1000)
        
        # 记录本 episode 中已访问的状态
        visited_states = set()
        
        for t in range(len(episode)):
            state, _, _ = episode[t]
            
            # 只处理第一次访问的状态
            if state in visited_states:
                continue
            
            visited_states.add(state)
            
            # 计算回报
            G = compute_return(episode, t, gamma)
            returns[state].append(G)
    
    # 计算平均价值
    V = {s: np.mean(Gs) for s, Gs in returns.items()}
    
    return V, dict(returns)


def every_visit_mc_prediction(
    env,
    policy,
    n_episodes: int = 10000,
    gamma: float = 1.0
) -> Dict:
    """
    每次访问蒙特卡洛预测
    
    统计每个状态在 episode 中每次访问时的回报
    
    Args:
        env: 环境
        policy: 策略函数
        n_episodes: episode 数量
        gamma: 折扣因子
    
    Returns:
        V: 状态价值函数字典 {state: value}
        returns: 每个状态的回报列表 {state: [G1, G2, ...]}
    """
    returns = defaultdict(list)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy, max_steps=1000)
        
        for t in range(len(episode)):
            state, _, _ = episode[t]
            G = compute_return(episode, t, gamma)
            returns[state].append(G)
    
    V = {s: np.mean(Gs) for s, Gs in returns.items()}
    
    return V, dict(returns)


def first_visit_mc_prediction_q(
    env,
    policy,
    n_episodes: int = 10000,
    gamma: float = 1.0
) -> Dict:
    """
    第一访问 MC 预测 - 估计动作价值函数 Q_π
    
    Args:
        env: 环境
        policy: 策略函数
        n_episodes: episode 数量
        gamma: 折扣因子
    
    Returns:
        Q: 动作价值函数字典 {(state, action): value}
        returns: 回报列表
    """
    returns = defaultdict(list)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy, max_steps=1000)
        
        # 记录已访问的 (state, action) 对
        visited = set()
        
        for t in range(len(episode)):
            state, action, _ = episode[t]
            state_action = (state, action)
            
            if state_action in visited:
                continue
            
            visited.add(state_action)
            
            G = compute_return(episode, t, gamma)
            returns[state_action].append(G)
    
    Q = {sa: np.mean(Gs) for sa, Gs in returns.items()}
    
    return Q, dict(returns)


class MCPredictor:
    """
    蒙特卡洛预测器 - 增量式更新版本
    
    支持在线学习，无需存储所有回报
    """
    
    def __init__(self, gamma: float = 1.0):
        """
        Args:
            gamma: 折扣因子
        """
        self.gamma = gamma
        self.V = defaultdict(float)
        self.N = defaultdict(int)  # 访问计数
    
    def update(self, episode: List[Tuple], first_visit: bool = True):
        """
        根据一个 episode 更新价值估计
        
        Args:
            episode: episode 列表
            first_visit: 是否使用第一访问
        """
        visited = set()
        
        for t in range(len(episode)):
            state, _, _ = episode[t]
            
            if first_visit and state in visited:
                continue
            
            visited.add(state)
            G = compute_return(episode, t, self.gamma)
            
            # 增量式更新
            self.N[state] += 1
            n = self.N[state]
            self.V[state] += (1/n) * (G - self.V[state])
    
    def train(self, env, policy, n_episodes: int = 10000, first_visit: bool = True):
        """
        训练预测器
        
        Args:
            env: 环境
            policy: 策略函数
            n_episodes: episode 数量
            first_visit: 是否使用第一访问
        """
        for _ in range(n_episodes):
            episode = generate_episode(env, policy, max_steps=1000)
            self.update(episode, first_visit)
    
    def get_value(self, state) -> float:
        """获取状态价值"""
        return self.V.get(state, 0.0)
