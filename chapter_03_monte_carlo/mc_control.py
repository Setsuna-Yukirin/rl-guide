"""
蒙特卡洛控制算法

用于寻找最优策略 π*
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict
import math


def generate_episode_with_policy(env, policy, max_steps: int = 1000) -> list:
    """生成 episode"""
    episode = []
    state, _ = env.reset()
    
    for _ in range(max_steps):
        action = policy.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        
        if terminated or truncated:
            break
        
        state = next_state
    
    return episode


def compute_return(episode: list, t: int, gamma: float = 1.0) -> float:
    """计算折扣回报"""
    G = 0.0
    for i in range(t, len(episode)):
        _, _, reward = episode[i]
        G += (gamma ** (i - t)) * reward
    return G


def mc_control_es(
    env,
    n_episodes: int = 10000,
    gamma: float = 1.0,
    epsilon: float = 0.1
) -> Tuple[Dict, Dict]:
    """
    蒙特卡洛控制 - ε-soft 版本
    
    假设所有状态 - 动作对都有非零概率被选择
    
    Args:
        env: 环境
        n_episodes: episode 数量
        gamma: 折扣因子
        epsilon: 探索概率
    
    Returns:
        Q: 最优动作价值函数
        policy: 最优策略
    """
    # 初始化 Q 值和回报
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    
    # 创建 ε-贪婪策略
    from chapter_03_monte_carlo.epsilon_greedy import EpsilonGreedyPolicy
    policy = EpsilonGreedyPolicy(env.action_space.n, epsilon, Q)
    
    for episode_idx in range(n_episodes):
        # 生成 episode
        episode = generate_episode_with_policy(env, policy)
        
        # 第一访问 MC
        visited = set()
        
        for t in range(len(episode)):
            state, action, _ = episode[t]
            state_action = (state, action)
            
            if state_action in visited:
                continue
            
            visited.add(state_action)
            
            # 计算回报
            G = compute_return(episode, t, gamma)
            returns[state_action].append(G)
            
            # 更新 Q 值
            Q[state][action] = np.mean(returns[state_action])
        
        # 可选：衰减 ε
        # policy.decay_epsilon(decay_rate=0.999, min_epsilon=0.01)
    
    # 提取贪婪策略
    greedy_policy = extract_greedy_policy(Q, env.action_space.n)
    
    return dict(Q), greedy_policy


def mc_control_glie(
    env,
    n_episodes: int = 10000,
    gamma: float = 1.0,
    k0: float = 100.0
) -> Tuple[Dict, Dict]:
    """
    蒙特卡洛控制 - GLIE 版本
    
    使用 GLIE (Greedy in the Limit with Infinite Exploration) 策略
    
    Args:
        env: 环境
        n_episodes: episode 数量
        gamma: 折扣因子
        k0: GLIE 衰减常数
    
    Returns:
        Q: 最优动作价值函数
        policy: 最优策略
    """
    from chapter_03_monte_carlo.epsilon_greedy import GLIEPolicy
    
    # 初始化
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))  # 访问计数
    
    # GLIE 策略
    policy = GLIEPolicy(env.action_space.n, k0)
    policy.set_Q(Q)
    
    for episode_idx in range(n_episodes):
        episode = generate_episode_with_policy(env, policy)
        
        visited = set()
        
        for t in range(len(episode)):
            state, action, _ = episode[t]
            state_action = (state, action)
            
            if state_action in visited:
                continue
            
            visited.add(state_action)
            
            G = compute_return(episode, t, gamma)
            
            # 增量式更新 Q 值
            N[state][action] += 1
            n = N[state][action]
            Q[state][action] += (1/n) * (G - Q[state][action])
    
    greedy_policy = extract_greedy_policy(Q, env.action_space.n)
    
    return dict(Q), greedy_policy


def extract_greedy_policy(Q: Dict, n_actions: int) -> Dict:
    """
    从 Q 函数提取贪婪策略
    
    Args:
        Q: 动作价值函数
        n_actions: 动作数量
    
    Returns:
        policy: {state: best_action}
    """
    policy = {}
    
    for state, q_values in Q.items():
        if isinstance(q_values, np.ndarray):
            policy[state] = int(np.argmax(q_values))
        else:
            # 处理字典形式
            policy[state] = max(q_values.keys(), key=lambda a: q_values[a])
    
    return policy


class MCControl:
    """
    蒙特卡洛控制器 - 增量式版本
    """
    
    def __init__(
        self,
        env,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        use_glie: bool = False
    ):
        """
        Args:
            env: 环境
            gamma: 折扣因子
            epsilon: 初始探索概率
            use_glie: 是否使用 GLIE 策略
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_glie = use_glie
        
        # Q 函数和访问计数
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # 策略
        if use_glie:
            from chapter_03_monte_carlo.epsilon_greedy import GLIEPolicy
            self.policy = GLIEPolicy(env.action_space.n)
        else:
            from chapter_03_monte_carlo.epsilon_greedy import EpsilonGreedyPolicy
            self.policy = EpsilonGreedyPolicy(env.action_space.n, epsilon, self.Q)
        
        self.policy.set_Q(self.Q)
    
    def train(self, n_episodes: int = 10000, verbose: bool = False):
        """
        训练控制器
        
        Args:
            n_episodes: episode 数量
            verbose: 是否打印进度
        """
        for episode_idx in range(n_episodes):
            episode = generate_episode_with_policy(self.env, self.policy)
            
            visited = set()
            
            for t in range(len(episode)):
                state, action, _ = episode[t]
                state_action = (state, action)
                
                if state_action in visited:
                    continue
                
                visited.add(state_action)
                
                G = compute_return(episode, t, self.gamma)
                
                # 增量式更新
                self.N[state][action] += 1
                n = self.N[state][action]
                self.Q[state][action] += (1/n) * (G - self.Q[state][action])
            
            if verbose and (episode_idx + 1) % 1000 == 0:
                avg_return = np.mean([r for _, _, r in episode])
                print(f"Episode {episode_idx + 1}/{n_episodes}, Avg Reward: {avg_return:.2f}")
    
    def get_policy(self) -> Dict:
        """获取当前贪婪策略"""
        return extract_greedy_policy(self.Q, self.env.action_space.n)
    
    def get_action(self, state, explore: bool = True) -> int:
        """
        获取动作
        
        Args:
            state: 状态
            explore: 是否探索
        
        Returns:
            action: 动作
        """
        return self.policy.get_action(state, explore)
