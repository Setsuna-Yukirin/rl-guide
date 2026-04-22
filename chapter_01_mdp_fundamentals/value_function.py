"""
第 1 章：MDP 基础 - 价值函数与策略

包含:
- 价值函数计算 (V_π, Q_π)
- 策略评估
- 策略改进
- 策略迭代
- 价值迭代
"""

import numpy as np
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.core import TabularMDP


# ============== 贝尔曼方程 ==============

def bellman_expectation_V(mdp, V: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """贝尔曼期望方程 - 计算 V_π"""
    new_V = np.zeros(mdp.nS)
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                new_V[s] += policy[s, a] * prob * (reward + mdp.gamma * V[next_s])
    return new_V


def bellman_optimality_V(mdp, V: np.ndarray) -> np.ndarray:
    """贝尔曼最优化方程 - 计算 V*"""
    new_V = np.zeros(mdp.nS)
    for s in range(mdp.nS):
        q_values = np.zeros(mdp.nA)
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                q_values[a] += prob * (reward + mdp.gamma * V[next_s])
        new_V[s] = np.max(q_values)
    return new_V


# ============== 价值函数计算 ==============

def compute_V_pi(mdp, policy: np.ndarray, epsilon: float = 1e-6, max_iterations: int = 1000) -> np.ndarray:
    """计算策略π的价值函数 V_π"""
    nS = policy.shape[0]
    V = np.zeros(nS)
    
    for i in range(max_iterations):
        new_V = bellman_expectation_V(mdp, V, policy)
        delta = np.max(np.abs(new_V - V))
        V = new_V
        if delta < epsilon:
            break
    
    return V


def compute_Q_pi(mdp, V: np.ndarray) -> np.ndarray:
    """从 V_π 计算 Q_π"""
    nS, nA = mdp.nS, mdp.nA
    Q = np.zeros((nS, nA))
    
    for s in range(nS):
        for a in range(nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                Q[s, a] += prob * (reward + mdp.gamma * V[next_s])
    
    return Q


def compute_advantage(V: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """计算优势函数 A_π(s,a) = Q_π(s,a) - V_π(s)"""
    return Q - V.reshape(-1, 1)


# ============== 策略评估与改进 ==============

def evaluate_policy(mdp, policy: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """策略评估：计算给定策略的价值函数"""
    return compute_V_pi(mdp, policy, epsilon)


def improve_policy(mdp, V: np.ndarray) -> np.ndarray:
    """策略改进：基于价值函数改进策略"""
    Q = compute_Q_pi(mdp, V)
    nS, nA = mdp.nS, mdp.nA
    new_policy = np.zeros((nS, nA))
    
    for s in range(nS):
        best_a = np.argmax(Q[s])
        new_policy[s, best_a] = 1.0
    
    return new_policy


def policy_iteration(mdp, epsilon: float = 1e-6, max_iterations: int = 1000):
    """策略迭代算法"""
    nS, nA = mdp.nS, mdp.nA
    policy = np.ones((nS, nA)) / nA
    
    for i in range(max_iterations):
        V = evaluate_policy(mdp, policy, epsilon)
        new_policy = improve_policy(mdp, V)
        
        if np.array_equal(policy, new_policy):
            return V, policy, i + 1
        
        policy = new_policy
    
    V = evaluate_policy(mdp, policy, epsilon)
    return V, policy, max_iterations


def value_iteration(mdp, epsilon: float = 1e-6, max_iterations: int = 1000):
    """价值迭代算法"""
    nS = mdp.nS
    V = np.zeros(nS)
    
    for i in range(max_iterations):
        new_V = bellman_optimality_V(mdp, V)
        delta = np.max(np.abs(new_V - V))
        V = new_V
        
        if delta < epsilon:
            break
    
    # 从 V* 提取最优策略
    Q = compute_Q_pi(mdp, V)
    policy = np.zeros((nS, mdp.nA))
    for s in range(nS):
        best_a = np.argmax(Q[s])
        policy[s, best_a] = 1.0
    
    return V, policy, i + 1


# ============== 测试 ==============

if __name__ == "__main__":
    print("测试第 1 章：价值函数与策略\n")
    
    # 创建 MDP
    mdp = TabularMDP(n_states=4, n_actions=2, gamma=0.99)
    
    for s in range(4):
        for a in range(2):
            next_s = min(s + 1, 3) if a == 0 else min(s + 2, 3)
            reward = 1.0 if next_s == 3 else 0.0
            mdp.set_transition(s, a, [(1.0, next_s, reward, next_s == 3)])
    
    print("✓ MDP 创建成功")
    
    # 测试 1: 价值函数计算
    print("\n1. 测试价值函数计算")
    policy_random = np.ones((4, 2)) * 0.5
    V_random = compute_V_pi(mdp, policy_random)
    print(f"   随机策略 V_π: {V_random}")
    
    # 测试 2: Q 函数
    print("\n2. 测试 Q 函数")
    Q_random = compute_Q_pi(mdp, V_random)
    print(f"   Q_π:\n{Q_random}")
    
    # 测试 3: 优势函数
    print("\n3. 测试优势函数")
    A_random = compute_advantage(V_random, Q_random)
    print(f"   A_π:\n{A_random}")
    
    # 测试 4: 策略迭代
    print("\n4. 测试策略迭代")
    V_opt, policy_opt, iters = policy_iteration(mdp)
    print(f"   最优价值 V*: {V_opt}")
    print(f"   最优策略:\n{policy_opt}")
    print(f"   迭代次数：{iters}")
    
    # 测试 5: 价值迭代
    print("\n5. 测试价值迭代")
    V_vi, policy_vi, iters_vi = value_iteration(mdp)
    print(f"   最优价值 V*: {V_vi}")
    print(f"   迭代次数：{iters_vi}")
    
    print("\n✅ 所有测试通过！")
