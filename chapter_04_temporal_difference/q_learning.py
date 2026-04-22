"""
Q-Learning 算法

离策略时序差分控制算法
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning 智能体
    
    离策略学习：行为策略 (ε-greedy) 和目标策略 (贪婪) 不同
    """
    
    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        """
        Args:
            n_actions: 动作数量
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索概率
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q 函数
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
        # 统计
        self.total_reward = 0.0
        self.episode_count = 0
    
    def get_action(self, state, explore: bool = True) -> int:
        """
        选择动作（ε-greedy）
        
        Args:
            state: 状态
            explore: 是否探索
        
        Returns:
            action: 选择的动作
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool
    ):
        """
        更新 Q 值
        
        Q-Learning 更新规则：
        Q(s,a) ← Q(s,a) + α * [r + γ*max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            terminated: 是否终止
        """
        # 下一状态的最大 Q 值（终止状态为 0）
        next_max_q = 0.0 if terminated else np.max(self.Q[next_state])
        
        # TD 误差
        td_error = reward + self.gamma * next_max_q - self.Q[state][action]
        
        # 更新 Q 值
        self.Q[state][action] += self.alpha * td_error
    
    def train_episode(self, env, max_steps: int = 1000) -> float:
        """
        训练一个 episode
        
        Args:
            env: 环境
            max_steps: 最大步数
        
        Returns:
            episode_reward: 本 episode 的总奖励
        """
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        
        while not (terminated or truncated):
            # 选择动作
            action = self.get_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            # 更新 Q 值
            self.update(state, action, reward, next_state, terminated)
            
            state = next_state
        
        self.total_reward += episode_reward
        self.episode_count += 1
        
        return episode_reward
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = False) -> list:
        """
        训练多个 episode
        
        Args:
            env: 环境
            n_episodes: episode 数量
            verbose: 是否打印进度
        
        Returns:
            rewards: 每个 episode 的奖励
        """
        rewards = []
        
        for i in range(n_episodes):
            episode_reward = self.train_episode(env)
            rewards.append(episode_reward)
            
            if verbose and (i + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {i+1}/{n_episodes}, Avg Reward (last 100): {avg_reward:.2f}")
        
        return rewards
    
    def get_policy(self) -> Dict:
        """
        获取贪婪策略
        
        Returns:
            policy: {state: best_action}
        """
        policy = {}
        for state, q_values in self.Q.items():
            policy[state] = int(np.argmax(q_values))
        return policy
    
    def decay_epsilon(self, decay_rate: float = 0.99, min_epsilon: float = 0.01):
        """
        衰减 ε
        
        Args:
            decay_rate: 衰减率
            min_epsilon: 最小值
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


def q_learning(
    env,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    decay_epsilon: bool = True
) -> Tuple[Dict, Dict, list]:
    """
    Q-Learning 算法
    
    Args:
        env: 环境
        n_episodes: episode 数量
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 初始探索概率
        decay_epsilon: 是否衰减 ε
    
    Returns:
        Q: 动作价值函数
        policy: 贪婪策略
        rewards: 每个 episode 的奖励
    """
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    rewards = agent.train(env, n_episodes)
    
    # 提取策略
    policy = agent.get_policy()
    
    return dict(agent.Q), policy, rewards


class DoubleQLearningAgent(QLearningAgent):
    """
    Double Q-Learning 智能体
    
    减少 Q-Learning 的最大值偏差
    """
    
    def __init__(self, n_actions: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        super().__init__(n_actions, alpha, gamma, epsilon)
        
        # 两个独立的 Q 函数
        self.Q1 = defaultdict(lambda: np.zeros(n_actions))
        self.Q2 = defaultdict(lambda: np.zeros(n_actions))
    
    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool):
        """
        Double Q-Learning 更新
        
        随机选择一个 Q 函数用于选择动作，另一个用于评估
        """
        if np.random.random() < 0.5:
            # 用 Q1 选择，Q2 评估
            next_action = np.argmax(self.Q1[next_state])
            next_q = self.Q2[next_state][next_action]
            q = self.Q1[state][action]
        else:
            # 用 Q2 选择，Q1 评估
            next_action = np.argmax(self.Q2[next_state])
            next_q = self.Q1[next_state][next_action]
            q = self.Q2[state][action]
        
        # TD 误差
        next_q_value = 0.0 if terminated else next_q
        td_error = reward + self.gamma * next_q_value - q
        
        # 更新
        if np.random.random() < 0.5:
            self.Q1[state][action] += self.alpha * td_error
        else:
            self.Q2[state][action] += self.alpha * td_error
    
    def get_q_values(self, state) -> np.ndarray:
        """获取合并的 Q 值"""
        return self.Q1[state] + self.Q2[state]
    
    def get_policy(self) -> Dict:
        """获取贪婪策略（基于两个 Q 的和）"""
        policy = {}
        for state in set(list(self.Q1.keys()) + list(self.Q2.keys())):
            q_sum = self.Q1[state] + self.Q2[state]
            policy[state] = int(np.argmax(q_sum))
        return policy
