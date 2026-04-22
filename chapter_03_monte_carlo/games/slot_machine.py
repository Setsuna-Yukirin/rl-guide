"""
Slot Machine 老虎机环境

经典的老虎机游戏，用于演示多臂老虎机 (Multi-Armed Bandit) 问题
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Tuple


class SlotMachineEnv(gym.Env):
    """
    老虎机环境
    
    场景描述：
    - 多个老虎机（臂），每个有不同的奖励分布
    - 目标：最大化总奖励
    - 需要平衡探索（尝试新机器）和利用（选择已知最好的）
    
    状态：当前各机器的统计信息
    动作：选择哪个老虎机
    奖励：随机的（根据机器分布）
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        n_machines: int = 10,
        reward_type: str = 'bernoulli',  # 'bernoulli' or 'gaussian'
        seed: Optional[int] = None
    ):
        """
        Args:
            n_machines: 老虎机数量
            reward_type: 奖励类型 ('bernoulli' or 'gaussian')
            seed: 随机种子
        """
        super().__init__()
        
        self.n_machines = n_machines
        self.reward_type = reward_type
        
        self.action_space = spaces.Discrete(n_machines)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(n_machines * 2,),  # (mean, count) for each machine
            dtype=np.float32
        )
        
        # 每个老虎机的真实奖励概率/均值
        if seed is not None:
            np.random.seed(seed)
        
        if reward_type == 'bernoulli':
            # 伯努利分布：0-1 之间的概率
            self.true_probs = np.random.uniform(0.1, 0.9, n_machines)
        else:
            # 高斯分布：均值 0-1，方差 1
            self.true_means = np.random.uniform(0, 1, n_machines)
            self.true_stds = np.ones(n_machines)
        
        # 统计信息
        self.counts = None
        self.rewards = None
        self.total_reward = 0
        self.steps = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.counts = np.zeros(self.n_machines)
        self.rewards = np.zeros(self.n_machines)
        self.total_reward = 0
        self.steps = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取观测（统计信息）"""
        obs = np.zeros(self.n_machines * 2, dtype=np.float32)
        
        for i in range(self.n_machines):
            if self.counts[i] > 0:
                obs[i * 2] = self.rewards[i] / self.counts[i]  # 平均奖励
            else:
                obs[i * 2] = 0.0
            obs[i * 2 + 1] = min(1.0, self.counts[i] / 100)  # 归一化计数
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        拉老虎机
        
        Returns:
            observation: 更新后的统计信息
            reward: 奖励
            terminated: 是否结束
            truncated: 是否超时
            info: 额外信息
        """
        self.steps += 1
        
        # 生成奖励
        if self.reward_type == 'bernoulli':
            reward = 1.0 if np.random.random() < self.true_probs[action] else 0.0
            true_value = self.true_probs[action]
        else:
            reward = np.random.normal(self.true_means[action], self.true_stds[action])
            true_value = self.true_means[action]
        
        # 更新统计
        self.counts[action] += 1
        self.rewards[action] += reward
        self.total_reward += reward
        
        info = {
            'reward': reward,
            'true_value': true_value,
            'total_reward': self.total_reward,
            'optimal': action == np.argmax(self.true_probs if self.reward_type == 'bernoulli' else self.true_means)
        }
        
        return self._get_observation(), reward, False, False, info
    
    def get_optimal_action(self) -> int:
        """获取最优动作"""
        if self.reward_type == 'bernoulli':
            return int(np.argmax(self.true_probs))
        else:
            return int(np.argmax(self.true_means))
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'ansi':
            return self._render_ansi()
    
    def _render_human(self):
        """人类可读渲染"""
        print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """ANSI 渲染"""
        lines = ["===== Slot Machine ====="]
        
        for i in range(self.n_machines):
            if self.reward_type == 'bernoulli':
                true_val = self.true_probs[i]
            else:
                true_val = self.true_means[i]
            
            if self.counts[i] > 0:
                est_val = self.rewards[i] / self.counts[i]
            else:
                est_val = 0.0
            
            bar_len = int(self.counts[i] / max(1, max(self.counts)) * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            
            lines.append(f"Machine {i:2d}: [{bar}] Est={est_val:.3f} True={true_val:.3f}")
        
        lines.append(f"Total reward: {self.total_reward:.2f}")
        lines.append("========================")
        
        return '\n'.join(lines)


class SlotMachineWithBonus(SlotMachineEnv):
    """
    带奖励的老虎机
    
    某些机器有额外奖励或惩罚
    """
    
    def __init__(
        self,
        n_bonus: int = 2,
        bonus_multiplier: float = 2.0,
        **kwargs
    ):
        """
        Args:
            n_bonus: 奖励机器数量
            bonus_multiplier: 奖励倍数
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        self.n_bonus = n_bonus
        self.bonus_multiplier = bonus_multiplier
        
        # 随机选择奖励机器
        self.bonus_machines = set(np.random.choice(self.n_machines, size=n_bonus, replace=False))
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            self.bonus_machines = set(np.random.choice(self.n_machines, size=self.n_bonus, replace=False))
        
        return super().reset(seed=seed)
    
    def step(self, action: int) -> Tuple:
        """执行动作（考虑奖励）"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 应用奖励倍数
        if action in self.bonus_machines:
            reward *= self.bonus_multiplier
            info['bonus'] = True
        
        return obs, reward, terminated, truncated, info
