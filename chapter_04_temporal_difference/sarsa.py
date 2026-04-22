"""
SARSA 算法

在策略时序差分控制算法
State-Action-Reward-State-Action
"""

import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class SarsaAgent:
    """
    SARSA 智能体
    
    在策略学习：学习和执行的是同一个策略
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
        next_action: int,
        terminated: bool
    ):
        """
        更新 Q 值
        
        SARSA 更新规则：
        Q(s,a) ← Q(s,a) + α * [r + γ*Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            next_action: 下一状态的动作
            terminated: 是否终止
        """
        # 下一状态的 Q 值（终止状态为 0）
        next_q = 0.0 if terminated else self.Q[next_state][next_action]
        
        # TD 误差
        td_error = reward + self.gamma * next_q - self.Q[state][action]
        
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
        
        # 选择初始动作
        action = self.get_action(state)
        
        episode_reward = 0.0
        
        while not (terminated or truncated):
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            # 选择下一动作
            next_action = self.get_action(next_state, explore=True)
            
            # 更新 Q 值
            self.update(state, action, reward, next_state, next_action, terminated)
            
            state = next_state
            action = next_action
        
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
        """衰减 ε"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


def sarsa(
    env,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    decay_epsilon: bool = True
) -> Tuple[Dict, Dict, list]:
    """
    SARSA 算法
    
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
    agent = SarsaAgent(
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    rewards = agent.train(env, n_episodes)
    
    # 提取策略
    policy = agent.get_policy()
    
    return dict(agent.Q), policy, rewards


class NStepSarsaAgent:
    """
    n 步 SARSA 智能体
    
    使用 n 步回报进行更新
    """
    
    def __init__(
        self,
        n_actions: int,
        n_steps: int = 1,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        """
        Args:
            n_actions: 动作数量
            n_steps: n 步更新的 n
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索概率
        """
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def get_action(self, state, explore: bool = True) -> int:
        """ε-greedy 动作选择"""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state]))
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = False) -> list:
        """
        使用 n 步 SARSA 训练
        
        Args:
            env: 环境
            n_episodes: episode 数量
            verbose: 是否打印进度
        
        Returns:
            rewards: 每个 episode 的奖励
        """
        rewards = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            terminated = False
            truncated = False
            
            # 存储轨迹
            states = [state]
            actions = [self.get_action(state)]
            rewards_list = [0]  # 占位符，索引从 1 开始
            
            T = float('inf')  # 终止时间
            t = 0
            episode_reward = 0.0
            
            while True:
                if T == float('inf'):
                    # 执行动作
                    next_state, reward, terminated, truncated, _ = env.step(actions[t])
                    episode_reward += reward
                    rewards_list.append(reward)
                    
                    if terminated or truncated:
                        T = t + 1
                    else:
                        # 选择下一动作
                        next_action = self.get_action(next_state)
                        states.append(next_state)
                        actions.append(next_action)
                
                # 更新时间
                tau = t - self.n_steps + 1
                
                if tau >= 0:
                    # 计算 n 步回报
                    end = min(tau + self.n_steps, T)
                    G = sum(
                        self.gamma ** (i - tau - 1) * rewards_list[i]
                        for i in range(tau + 1, end)
                    )
                    
                    if tau + self.n_steps < T:
                        # 加上 bootstrapped 值
                        G += self.gamma ** self.n_steps * self.Q[states[tau + self.n_steps]][actions[tau + self.n_steps]]
                    
                    # 更新 Q 值
                    state_tau = states[tau]
                    action_tau = actions[tau]
                    self.Q[state_tau][action_tau] += self.alpha * (G - self.Q[state_tau][action_tau])
                
                if T != float('inf'):
                    t += 1
                    if t >= T:
                        break
                else:
                    t += 1
            
            rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}")
        
        return rewards
    
    def get_policy(self) -> Dict:
        """获取贪婪策略"""
        policy = {}
        for state, q_values in self.Q.items():
            policy[state] = int(np.argmax(q_values))
        return policy
