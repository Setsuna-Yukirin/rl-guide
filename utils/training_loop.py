"""
训练循环 (Training Loop)

提供通用的强化学习训练流程管理。

功能:
- 单环境/多环境训练
- 训练统计记录
- 检查点保存
- 早停机制
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import time
from pathlib import Path
import json

from utils.core import Policy, ReplayBuffer


class TrainingLoop:
    """
    通用训练循环管理器
    
    支持各种 RL 算法的训练流程。
    
    Attributes:
        agent: RL 智能体（必须有 get_action 和 update 方法）
        env: RL 环境（Gymnasium 风格）
        num_episodes: 训练集数
        max_steps: 每集最大步数
    
    Example:
        >>> from utils import TrainingLoop
        >>> loop = TrainingLoop(agent, env, num_episodes=1000)
        >>> stats = loop.run()
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        num_episodes: int = 1000,
        max_steps: Optional[int] = None,
        buffer: Optional[ReplayBuffer] = None,
        batch_size: int = 64,
        render: bool = False,
        render_interval: int = 100,
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        """
        初始化训练循环
        
        Args:
            agent: RL 智能体（必须有 get_action 和 update 方法）
            env: RL 环境（Gymnasium 风格）
            num_episodes: 训练集数
            max_steps: 每集最大步数（可选）
            buffer: 经验回放缓冲区（可选，用于 off-policy 算法）
            batch_size: 批量大小
            render: 是否渲染
            render_interval: 渲染间隔（每多少集渲染一次）
            verbose: 是否打印训练信息
            seed: 随机种子
        """
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.buffer = buffer
        self.batch_size = batch_size
        self.render = render
        self.render_interval = render_interval
        self.verbose = verbose
        self.seed = seed
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            if hasattr(env, 'reset'):
                try:
                    env.reset(seed=seed)
                except TypeError:
                    env.reset()
        
        # 训练统计
        self.stats: Dict[str, List[float]] = {
            'rewards': [],
            'steps': [],
            'successes': [],
        }
        
        # 检查点目录
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def run(self) -> Dict[str, List[float]]:
        """
        运行训练循环
        
        Returns:
            训练统计字典
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"开始训练：{self.num_episodes} 集")
            print(f"环境：{self.env}")
            print(f"智能体：{self.agent.__class__.__name__}")
            print("-" * 60)
        
        for episode in range(self.num_episodes):
            episode_start = time.time()
            
            # 重置环境
            state, info = self._reset_env()
            episode_reward = 0.0
            episode_steps = 0
            
            # 训练单集
            done = False
            while not done:
                # 选择动作
                action = self.agent.get_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 存储经验（如果有 buffer）
                if self.buffer is not None:
                    self.buffer.add(state, action, reward, next_state, done)
                
                # 更新智能体
                update_info = self._update_agent(state, action, reward, next_state, done)
                
                # 累计奖励
                episode_reward += reward
                episode_steps += 1
                
                # 更新状态
                state = next_state
                
                # 检查最大步数
                if self.max_steps is not None and episode_steps >= self.max_steps:
                    done = True
                
                # 渲染
                if self.render and episode % self.render_interval == 0:
                    self.env.render()
            
            # 记录统计
            self.stats['rewards'].append(episode_reward)
            self.stats['steps'].append(episode_steps)
            self.stats['successes'].append(float(reward > 0) if 'success' in info else 0.0)
            
            # 打印进度
            if self.verbose and (episode + 1) % max(1, self.num_episodes // 10) == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.stats['rewards'][-100:])
                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg(100): {avg_reward:.2f} | "
                      f"Steps: {episode_steps} | "
                      f"Time: {elapsed:.1f}s")
            
            # 保存检查点
            if (episode + 1) % 100 == 0:
                self._save_checkpoint(episode + 1)
        
        # 训练完成
        total_time = time.time() - start_time
        
        if self.verbose:
            print("-" * 60)
            print(f"训练完成！")
            print(f"总时间：{total_time:.1f}s")
            print(f"平均奖励：{np.mean(self.stats['rewards']):.2f}")
            print(f"最后 100 集平均：{np.mean(self.stats['rewards'][-100:]):.2f}")
        
        return self.stats
    
    def _reset_env(self) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            return result, {}
    
    def _update_agent(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """
        更新智能体
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否终止
            
        Returns:
            更新信息（可选）
        """
        if self.buffer is not None and len(self.buffer) >= self.batch_size:
            # Off-policy 算法：从 buffer 采样
            batch = self.buffer.sample(self.batch_size)
            return self.agent.update(batch)
        else:
            # On-policy 算法：直接使用当前经验
            return self.agent.update((state, action, reward, next_state, done))
    
    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.json"
        
        checkpoint_data = {
            'episode': episode,
            'stats': {
                'rewards': self.stats['rewards'],
                'steps': self.stats['steps'],
            },
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        if self.verbose:
            print(f"  检查点已保存：{checkpoint_path}")
    
    def load_checkpoint(self, episode: int):
        """
        加载检查点
        
        Args:
            episode: 集数
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.json"
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        self.stats['rewards'] = checkpoint_data['stats']['rewards']
        self.stats['steps'] = checkpoint_data['stats']['steps']
        
        if self.verbose:
            print(f"检查点已加载：{checkpoint_path}")


class ParallelTrainingLoop:
    """
    并行训练循环
    
    使用多个并行环境加速训练。
    
    Example:
        >>> from gymnasium.vector import AsyncVectorEnv
        >>> envs = AsyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
        >>> loop = ParallelTrainingLoop(agent, envs, num_episodes=1000)
        >>> stats = loop.run()
    """
    
    def __init__(
        self,
        agent: Any,
        envs: Any,
        num_episodes: int = 1000,
        max_steps: Optional[int] = None,
        buffer: Optional[ReplayBuffer] = None,
        batch_size: int = 64,
        update_interval: int = 1,
        verbose: bool = True,
    ):
        """
        初始化并行训练循环
        
        Args:
            agent: RL 智能体
            envs: 并行环境（Gymnasium VectorEnv）
            num_episodes: 训练集数
            max_steps: 每集最大步数
            buffer: 经验回放缓冲区
            batch_size: 批量大小
            update_interval: 更新间隔（每多少步更新一次智能体）
            verbose: 是否打印信息
        """
        self.agent = agent
        self.envs = envs
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.buffer = buffer
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.verbose = verbose
        
        self.n_envs = envs.num_envs if hasattr(envs, 'num_envs') else 1
        
        self.stats: Dict[str, List[float]] = {
            'rewards': [],
            'steps': [],
        }
    
    def run(self) -> Dict[str, List[float]]:
        """运行并行训练"""
        if self.verbose:
            print(f"开始并行训练：{self.num_episodes} 集 × {self.n_envs} 环境")
        
        states, _ = self.envs.reset()
        episode_rewards = np.zeros(self.n_envs)
        episode_steps = np.zeros(self.n_envs)
        
        total_steps = 0
        
        for episode in range(self.num_episodes):
            for step in range(self.max_steps or 1000):
                # 选择动作
                actions = np.array([
                    self.agent.get_action(states[i])
                    for i in range(self.n_envs)
                ])
                
                # 执行动作
                next_states, rewards, terminateds, truncateds, infos = self.envs.step(actions)
                dones = terminateds | truncateds
                
                # 存储经验
                if self.buffer is not None:
                    for i in range(self.n_envs):
                        self.buffer.add(states[i], actions[i], rewards[i], 
                                       next_states[i], dones[i])
                
                # 更新累计
                episode_rewards += rewards
                episode_steps += 1
                total_steps += self.n_envs
                
                # 处理终止的环境
                for i in range(self.n_envs):
                    if dones[i]:
                        self.stats['rewards'].append(episode_rewards[i])
                        self.stats['steps'].append(episode_steps[i])
                        episode_rewards[i] = 0.0
                        episode_steps[i] = 0
                
                # 更新智能体
                if (self.buffer is not None and 
                    len(self.buffer) >= self.batch_size and
                    total_steps % self.update_interval == 0):
                    batch = self.buffer.sample(self.batch_size)
                    self.agent.update(batch)
                
                states = next_states
                
                if all(dones):
                    break
            
            # 打印进度
            if self.verbose and (episode + 1) % max(1, self.num_episodes // 10) == 0:
                avg_reward = np.mean(self.stats['rewards'][-100:]) if len(self.stats['rewards']) >= 100 else np.mean(self.stats['rewards'])
                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Total Steps: {total_steps}")
        
        return self.stats


if __name__ == "__main__":
    # 简单测试
    import gymnasium as gym
    from utils.core import QLearning, TabularQFunction, EpsilonGreedyPolicy, ReplayBuffer
    
    print("测试 TrainingLoop 类...\n")
    
    # 创建环境和智能体
    env = gym.make("CartPole-v1")
    
    # 简单离散化
    def discretize(state, bins=10):
        state = np.array(state)
        state = (state + 2.4) / 4.8  # 归一化到 [0, 1]
        state = np.floor(state * bins).astype(int)
        state = np.clip(state, 0, bins - 1)
        return int(np.sum(state * (bins ** np.arange(len(state)))))
    
    state_dim = 10 ** 4  # 简化
    action_dim = 2
    
    Q = TabularQFunction(state_dim=100, action_dim=2, init_value=0.0)
    policy = EpsilonGreedyPolicy(state_dim=100, action_dim=2, epsilon=0.1)
    policy.set_q_values(Q.q_values)
    
    buffer = ReplayBuffer(capacity=10000)
    
    # 创建训练循环
    loop = TrainingLoop(
        agent=policy,
        env=env,
        num_episodes=10,  # 快速测试
        buffer=buffer,
        batch_size=32,
        verbose=True,
    )
    
    # 运行训练
    stats = loop.run()
    
    print(f"\n✓ TrainingLoop 测试通过")
    print(f"  训练集数：{len(stats['rewards'])}")
    print(f"  平均奖励：{np.mean(stats['rewards']):.2f}")
    
    print("\n✅ TrainingLoop 测试完成！")
