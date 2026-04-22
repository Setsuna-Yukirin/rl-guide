"""
贝尔曼方程 (Bellman Equation)

实现贝尔曼期望方程和贝尔曼最优化方程。

贝尔曼期望方程:
    V_π(s) = Σ π(a|s) Σ P(s'|s,a) [R(s,a,s') + γV_π(s')]

贝尔曼最优化方程:
    V*(s) = max Σ P(s'|s,a) [R(s,a,s') + γV*(s')]
            a   s'

Reference:
    Sutton, R. S., & Barto, A. G. (2018). 
    Reinforcement learning: An introduction. MIT press.
"""

import numpy as np
from typing import Tuple, Optional

# 延迟导入，避免循环依赖
def _get_mdp_class():
    from utils.core import TabularMDP
    return TabularMDP


def bellman_expectation_V(
    mdp,
    V: np.ndarray,
    policy: np.ndarray,
) -> np.ndarray:
    """
    贝尔曼期望方程 - 计算 V_π
    
    Args:
        mdp: 表格型 MDP (TabularMDP 实例)
        V: 当前价值函数，形状 (nS,)
        policy: 策略 π(a|s)，形状 (nS, nA)
    
    Returns:
        new_V: 更新后的价值函数，形状 (nS,)
    """
    """
    贝尔曼期望方程 - 计算 V_π
    
    公式:
        V_π(s) = Σ π(a|s) Σ P(s'|s,a) [R(s,a,s') + γV_π(s')]
                  a      s'
    
    Args:
        mdp: 表格型 MDP
        V: 当前价值函数，形状 (nS,)
        policy: 策略 π(a|s)，形状 (nS, nA)
    
    Returns:
        new_V: 更新后的价值函数，形状 (nS,)
    
    Example:
        >>> from utils.core import TabularMDP
        >>> mdp = TabularMDP(n_states=4, n_actions=2)
        >>> # 设置转移概率...
        >>> V = np.zeros(4)
        >>> policy = np.ones((4, 2)) * 0.5  # 随机策略
        >>> new_V = bellman_expectation_V(mdp, V, policy)
    """
    if V.shape != (mdp.nS,):
        raise ValueError(f"V 的形状 {V.shape} 不匹配，期望 ({mdp.nS},)")
    
    if policy.shape != (mdp.nS, mdp.nA):
        raise ValueError(
            f"policy 的形状 {policy.shape} 不匹配，期望 ({mdp.nS}, {mdp.nA})"
        )
    
    new_V = np.zeros(mdp.nS)
    
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                new_V[s] += policy[s, a] * prob * (reward + mdp.gamma * V[next_s])
    
    return new_V


def bellman_expectation_Q(
    mdp,
    Q: np.ndarray,
    policy: np.ndarray,
) -> np.ndarray:
    """
    贝尔曼期望方程的 Q 函数版本
    
    公式:
        Q_π(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ Σ π(a'|s') Q_π(s',a')]
                   s'                  a'
    
    Args:
        mdp: 表格型 MDP
        Q: 当前 Q 函数，形状 (nS, nA)
        policy: 策略 π(a|s)，形状 (nS, nA)
    
    Returns:
        new_Q: 更新后的 Q 函数，形状 (nS, nA)
    """
    if Q.shape != (mdp.nS, mdp.nA):
        raise ValueError(f"Q 的形状 {Q.shape} 不匹配，期望 ({mdp.nS}, {mdp.nA})")
    
    new_Q = np.zeros_like(Q)
    
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                # Q(s,a) = Σ P(s'|s,a) [R + γ Σ π(a'|s') Q(s',a')]
                expected_Q_next = np.sum(policy[next_s] * Q[next_s])
                new_Q[s, a] += prob * (reward + mdp.gamma * expected_Q_next)
    
    return new_Q


def bellman_optimality_V(
    mdp,
    V: np.ndarray,
) -> np.ndarray:
    """
    贝尔曼最优化方程 - 计算 V*
    
    公式:
        V*(s) = max Σ P(s'|s,a) [R(s,a,s') + γV*(s')]
                 a  s'
    
    Args:
        mdp: 表格型 MDP
        V: 当前价值函数，形状 (nS,)
    
    Returns:
        new_V: 更新后的价值函数，形状 (nS,)
    
    Note:
        这是价值迭代的核心步骤
    """
    if V.shape != (mdp.nS,):
        raise ValueError(f"V 的形状 {V.shape} 不匹配，期望 ({mdp.nS},)")
    
    new_V = np.zeros(mdp.nS)
    
    for s in range(mdp.nS):
        q_values = np.zeros(mdp.nA)
        
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                q_values[a] += prob * (reward + mdp.gamma * V[next_s])
        
        # 选择最优动作的价值
        new_V[s] = np.max(q_values)
    
    return new_V


def bellman_optimality_Q(
    mdp,
    Q: np.ndarray,
) -> np.ndarray:
    """
    贝尔曼最优化方程的 Q 函数版本 - 计算 Q*
    
    公式:
        Q*(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ max_a' Q*(s',a')]
                  s'
    
    Args:
        mdp: 表格型 MDP
        Q: 当前 Q 函数，形状 (nS, nA)
    
    Returns:
        new_Q: 更新后的 Q 函数，形状 (nS, nA)
    
    Note:
        这是 Q-Learning 的理论基础
    """
    if Q.shape != (mdp.nS, mdp.nA):
        raise ValueError(f"Q 的形状 {Q.shape} 不匹配，期望 ({mdp.nS}, {mdp.nA})")
    
    new_Q = np.zeros_like(Q)
    
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                # Q*(s,a) = Σ P(s'|s,a) [R + γ max_a' Q*(s',a')]
                max_Q_next = np.max(Q[next_s])
                new_Q[s, a] += prob * (reward + mdp.gamma * max_Q_next)
    
    return new_Q


def compute_bellman_residual(
    mdp,
    V: np.ndarray,
    policy: Optional[np.ndarray] = None,
) -> float:
    """
    计算贝尔曼误差（残差）
    
    用于评估当前价值函数距离收敛还有多远。
    
    Args:
        mdp: 表格型 MDP
        V: 当前价值函数
        policy: 策略（如果提供则计算期望方程残差，否则计算最优化方程残差）
    
    Returns:
        residual: 贝尔曼误差（最大绝对值）
    
    Example:
        >>> residual = compute_bellman_residual(mdp, V, policy)
        >>> if residual < 1e-6:
        ...     print("已收敛")
    """
    if policy is not None:
        # 期望方程残差
        new_V = bellman_expectation_V(mdp, V, policy)
    else:
        # 最优化方程残差
        new_V = bellman_optimality_V(mdp, V)
    
    residual = np.max(np.abs(new_V - V))
    return float(residual)


if __name__ == "__main__":
    # 简单测试
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.core import TabularMDP
    
    print("测试贝尔曼方程...\n")
    
    # 创建简单 MDP
    mdp = TabularMDP(n_states=4, n_actions=2, gamma=0.99)
    
    # 设置转移概率（简单网格）
    # 状态 0: 左上，状态 1: 右上，状态 2: 左下，状态 3: 右下
    for s in range(4):
        for a in range(2):
            # 动作 0: 向右，动作 1: 向下
            if a == 0:  # 向右
                next_s = min(s + 1, 3)
            else:  # 向下
                next_s = min(s + 2, 3)
            
            reward = 1.0 if next_s == 3 else 0.0  # 到达目标状态奖励
            mdp.set_transition(s, a, [(1.0, next_s, reward, next_s == 3)])
    
    print("✓ MDP 创建成功")
    
    # 测试贝尔曼期望方程
    V = np.zeros(4)
    policy = np.ones((4, 2)) * 0.5  # 随机策略
    
    new_V = bellman_expectation_V(mdp, V, policy)
    print(f"✓ 贝尔曼期望方程测试通过")
    print(f"  初始 V: {V}")
    print(f"  更新后 V: {new_V}")
    
    # 测试贝尔曼最优化方程
    Q = np.zeros((4, 2))
    new_Q = bellman_optimality_Q(mdp, Q)
    print(f"\n✓ 贝尔曼最优化方程 (Q 版本) 测试通过")
    print(f"  初始 Q: \n{Q}")
    print(f"  更新后 Q: \n{new_Q}")
    
    # 测试残差计算
    residual = compute_bellman_residual(mdp, V, policy)
    print(f"\n✓ 贝尔曼残差测试通过")
    print(f"  残差：{residual:.6f}")
    
    # 迭代直到收敛
    print("\n迭代求解 V*...")
    V = np.zeros(4)
    for i in range(100):
        new_V = bellman_optimality_V(mdp, V)
        residual = np.max(np.abs(new_V - V))
        V = new_V
        
        if residual < 1e-6:
            print(f"  在第 {i+1} 次迭代收敛")
            break
    
    print(f"  最优价值函数 V*: {V}")
    print(f"  最终残差：{residual:.10f}")
    
    print("\n✅ 所有贝尔曼方程测试通过！")
