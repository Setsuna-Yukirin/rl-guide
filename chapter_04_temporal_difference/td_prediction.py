"""
时序差分预测算法

TD(0) - 单步时序差分预测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict


def generate_episode(env, policy, max_steps: int = 1000) -> List[Tuple]:
    """
    根据策略生成一个完整的 episode
    
    Args:
        env: 环境
        policy: 策略函数 state -> action
        max_steps: 最大步数
    
    Returns:
        episode: [(state, action, reward, next_state), ...]
    """
    episode = []
    state, _ = env.reset()
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and len(episode) < max_steps:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward, next_state))
        state = next_state
    
    return episode


def td_0_prediction(
    env,
    policy: Callable,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99
) -> Dict:
    """
    TD(0) 预测算法
    
    估计给定策略 π 的价值函数 V_π
    
    Args:
        env: 环境
        policy: 策略函数 state -> action
        n_episodes: episode 数量
        alpha: 学习率
        gamma: 折扣因子
    
    Returns:
        V: 状态价值函数字典 {state: value}
    """
    V = defaultdict(float)
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # TD 误差
            td_error = reward + gamma * V[next_state] - V[state]
            
            # 更新价值
            V[state] += alpha * td_error
            
            state = next_state
    
    return dict(V)


class TDPredictor:
    """
    TD 预测器 - 增量式版本
    
    支持在线学习和多种 TD 变体
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99):
        """
        Args:
            alpha: 学习率
            gamma: 折扣因子
        """
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(float)
        self.N = defaultdict(int)  # 访问计数
    
    def update(self, state: int, reward: float, next_state: int, terminated: bool):
        """
        单步 TD 更新
        
        Args:
            state: 当前状态
            reward: 奖励
            next_state: 下一状态
            terminated: 是否终止
        """
        next_value = 0.0 if terminated else self.V[next_state]
        
        # TD 误差
        td_error = reward + self.gamma * next_value - self.V[state]
        
        # 更新
        self.V[state] += self.alpha * td_error
        self.N[state] += 1
    
    def train(self, env, policy, n_episodes: int = 1000):
        """
        训练预测器
        
        Args:
            env: 环境
            policy: 策略函数
            n_episodes: episode 数量
        """
        for _ in range(n_episodes):
            state, _ = env.reset()
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = policy(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                self.update(state, reward, next_state, terminated)
                state = next_state
    
    def get_value(self, state) -> float:
        """获取状态价值"""
        return self.V.get(state, 0.0)


def td_lambda_prediction(
    env,
    policy: Callable,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    lambda_: float = 0.5
) -> Dict:
    """
    TD(λ) 预测算法 - 使用资格迹
    
    Args:
        env: 环境
        policy: 策略函数
        n_episodes: episode 数量
        alpha: 学习率
        gamma: 折扣因子
        lambda_: 迹衰减参数 (0=TD(0), 1=蒙特卡洛)
    
    Returns:
        V: 状态价值函数
    """
    V = defaultdict(float)
    eligibility = defaultdict(float)  # 资格迹
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        # 重置资格迹
        eligibility.clear()
        
        while not (terminated or truncated):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # TD 误差
            next_value = 0.0 if terminated else V[next_state]
            td_error = reward + gamma * next_value - V[state]
            
            # 更新当前状态的资格迹
            eligibility[state] += 1
            
            # 更新所有状态的价值
            for s in list(eligibility.keys()):
                V[s] += alpha * td_error * eligibility[s]
                eligibility[s] *= gamma * lambda_
            
            state = next_state
    
    return dict(V)
