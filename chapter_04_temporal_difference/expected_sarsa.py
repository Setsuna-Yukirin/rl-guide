"""
Expected SARSA 算法

使用时望值代替最大值的时序差分控制
"""

import numpy as np
from typing import Dict, Tuple
from collections import defaultdict


class ExpectedSarsaAgent:
    """
    Expected SARSA 智能体
    
    使用期望值代替最大值，减少方差
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
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state]))
    
    def compute_expected_q(self, state: int) -> float:
        """
        计算期望 Q 值
        
        E[Q(s,·)] = Σ_a π(a|s) * Q(s,a)
        
        对于 ε-greedy 策略：
        = (1 - ε + ε/|A|) * max_a Q(s,a) + (ε/|A|) * Σ_{a≠a*} Q(s,a)
        = (1 - ε) * max_a Q(s,a) + (ε/|A|) * Σ_a Q(s,a)
        
        Args:
            state: 状态
        
        Returns:
            expected_q: 期望 Q 值
        """
        q_values = self.Q[state]
        
        # 方法 1：直接按 ε-greedy 概率加权
        max_q = np.max(q_values)
        mean_q = np.mean(q_values)
        
        expected_q = (1 - self.epsilon) * max_q + self.epsilon * mean_q
        
        return expected_q
    
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
        
        Expected SARSA 更新规则：
        Q(s,a) ← Q(s,a) + α * [r + γ*E[Q(s',·)] - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            terminated: 是否终止
        """
        # 计算期望 Q 值
        if terminated:
            expected_q = 0.0
        else:
            expected_q = self.compute_expected_q(next_state)
        
        # TD 误差
        td_error = reward + self.gamma * expected_q - self.Q[state][action]
        
        # 更新 Q 值
        self.Q[state][action] += self.alpha * td_error
    
    def train_episode(self, env, max_steps: int = 1000) -> float:
        """
        训练一个 episode
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
        """获取贪婪策略"""
        policy = {}
        for state, q_values in self.Q.items():
            policy[state] = int(np.argmax(q_values))
        return policy
    
    def decay_epsilon(self, decay_rate: float = 0.99, min_epsilon: float = 0.01):
        """衰减 ε"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


def expected_sarsa(
    env,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1
) -> Tuple[Dict, Dict, list]:
    """
    Expected SARSA 算法
    
    Args:
        env: 环境
        n_episodes: episode 数量
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 探索概率
    
    Returns:
        Q: 动作价值函数
        policy: 贪婪策略
        rewards: 每个 episode 的奖励
    """
    agent = ExpectedSarsaAgent(
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    rewards = agent.train(env, n_episodes)
    policy = agent.get_policy()
    
    return dict(agent.Q), policy, rewards


def compare_td_algorithms(env, n_episodes: int = 500, n_runs: int = 10) -> Dict:
    """
    比较不同 TD 算法的性能
    
    Args:
        env: 环境
        n_episodes: 每个算法的 episode 数量
        n_runs: 运行次数取平均
    
    Returns:
        results: {algorithm: avg_rewards}
    """
    results = {}
    
    # Q-Learning
    all_q_rewards = []
    for _ in range(n_runs):
        _, _, rewards = q_learning(env, n_episodes=n_episodes)
        all_q_rewards.append(rewards)
    results['Q-Learning'] = np.mean(all_q_rewards, axis=0)
    
    # SARSA
    all_sarsa_rewards = []
    for _ in range(n_runs):
        _, _, rewards = sarsa(env, n_episodes=n_episodes)
        all_sarsa_rewards.append(rewards)
    results['SARSA'] = np.mean(all_sarsa_rewards, axis=0)
    
    # Expected SARSA
    all_expected_rewards = []
    for _ in range(n_runs):
        _, _, rewards = expected_sarsa(env, n_episodes=n_episodes)
        all_expected_rewards.append(rewards)
    results['Expected SARSA'] = np.mean(all_expected_rewards, axis=0)
    
    return results


# 导入以便比较
from chapter_04_temporal_difference.q_learning import q_learning
from chapter_04_temporal_difference.sarsa import sarsa
