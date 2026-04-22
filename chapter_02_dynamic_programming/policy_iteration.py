"""
动态规划算法实现

包含：
- 策略评估 (Policy Evaluation)
- 策略迭代 (Policy Iteration)
- 价值迭代 (Value Iteration)
"""

import numpy as np
from typing import Tuple, Optional


def policy_evaluation(
    mdp,
    policy: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000
) -> Tuple[np.ndarray, int]:
    """
    策略评估：计算给定策略的价值函数 V_π
    
    Args:
        mdp: MDP 环境，需要有 nS, nA, get_transition() 方法
        policy: 策略矩阵 π(a|s), shape (n_states, n_actions)
        gamma: 折扣因子
        theta: 收敛阈值
        max_iterations: 最大迭代次数
    
    Returns:
        V: 状态价值函数，shape (n_states,)
        iterations: 实际迭代次数
    """
    n_states = mdp.nS
    n_actions = mdp.nA
    V = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)
        
        for s in range(n_states):
            v = 0.0
            for a in range(n_actions):
                # 获取转移概率和奖励
                transitions = mdp.get_transition(s, a)
                for prob, next_s, reward, _ in transitions:
                    v += policy[s, a] * prob * (reward + gamma * V[next_s])
            V_new[s] = v
        
        # 检查收敛
        delta = np.max(np.abs(V_new - V))
        V = V_new
        
        if delta < theta:
            return V, iteration + 1
    
    return V, max_iterations


def policy_improvement(
    mdp,
    V: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    策略改进：基于价值函数计算贪婪策略
    
    Args:
        mdp: MDP 环境
        V: 状态价值函数
        gamma: 折扣因子
    
    Returns:
        new_policy: 改进后的策略矩阵
    """
    n_states = mdp.nS
    n_actions = mdp.nA
    new_policy = np.zeros((n_states, n_actions))
    
    for s in range(n_states):
        # 计算每个动作的 Q 值
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            transitions = mdp.get_transition(s, a)
            for prob, next_s, reward, _ in transitions:
                q_values[a] += prob * (reward + gamma * V[next_s])
        
        # 贪婪选择（确定性策略）
        best_action = np.argmax(q_values)
        new_policy[s, best_action] = 1.0
    
    return new_policy


def policy_iteration(
    mdp,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_policy_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    策略迭代算法
    
    Args:
        mdp: MDP 环境
        gamma: 折扣因子
        theta: 策略评估收敛阈值
        max_policy_iterations: 最大策略迭代次数
    
    Returns:
        V: 最优价值函数
        policy: 最优策略
        iterations: 策略迭代次数
    """
    n_states = mdp.nS
    n_actions = mdp.nA
    
    # 初始化随机策略
    policy = np.ones((n_states, n_actions)) / n_actions
    
    for iteration in range(max_policy_iterations):
        # 1. 策略评估
        V, _ = policy_evaluation(mdp, policy, gamma, theta)
        
        # 2. 策略改进
        new_policy = policy_improvement(mdp, V, gamma)
        
        # 检查策略是否收敛
        if np.allclose(policy, new_policy):
            return V, new_policy, iteration + 1
        
        policy = new_policy
    
    return policy_evaluation(mdp, policy, gamma, theta)


def value_iteration(
    mdp,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    价值迭代算法
    
    Args:
        mdp: MDP 环境
        gamma: 折扣因子
        theta: 收敛阈值
        max_iterations: 最大迭代次数
    
    Returns:
        V: 最优价值函数
        policy: 最优策略
        iterations: 实际迭代次数
    """
    n_states = mdp.nS
    n_actions = mdp.nA
    V = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)
        
        for s in range(n_states):
            # 计算每个动作的 Q 值
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                transitions = mdp.get_transition(s, a)
                for prob, next_s, reward, _ in transitions:
                    q_values[a] += prob * (reward + gamma * V[next_s])
            V_new[s] = np.max(q_values)
        
        # 检查收敛
        delta = np.max(np.abs(V_new - V))
        V = V_new
        
        if delta < theta:
            # 提取最优策略
            policy = extract_greedy_policy(mdp, V, gamma)
            return V, policy, iteration + 1
    
    policy = extract_greedy_policy(mdp, V, gamma)
    return V, policy, max_iterations


def extract_greedy_policy(
    mdp,
    V: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    从价值函数提取贪婪策略
    
    Args:
        mdp: MDP 环境
        V: 状态价值函数
        gamma: 折扣因子
    
    Returns:
        policy: 贪婪策略矩阵
    """
    n_states = mdp.nS
    n_actions = mdp.nA
    policy = np.zeros((n_states, n_actions))
    
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            transitions = mdp.get_transition(s, a)
            for prob, next_s, reward, _ in transitions:
                q_values[a] += prob * (reward + gamma * V[next_s])
        
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return policy


def compute_action_value(
    mdp,
    V: np.ndarray,
    state: int,
    action: int,
    gamma: float = 0.99
) -> float:
    """
    计算给定状态 - 动作对的 Q 值
    
    Args:
        mdp: MDP 环境
        V: 状态价值函数
        state: 当前状态
        action: 动作
        gamma: 折扣因子
    
    Returns:
        Q(s, a): 动作价值
    """
    q = 0.0
    transitions = mdp.get_transition(state, action)
    for prob, next_s, reward, _ in transitions:
        q += prob * (reward + gamma * V[next_s])
    return q
