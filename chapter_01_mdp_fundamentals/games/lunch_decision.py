"""
第 1 章游戏：🍱 午餐选择器

午餐决策 MDP 环境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional


class LunchDecisionEnv(gym.Env):
    """午餐选择环境"""
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.weather_types = 3
        self.budget_types = 2
        self.meal_types = 3
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([2, 1, 2]),
            dtype=np.int32
        )
        
        self.action_space = spaces.Discrete(3)
        
        self.satisfaction = np.array([5.0, 7.0, 3.0])
        self.cost = np.array([3.0, 10.0, 0.0])
        self.variety_bonus = 2.0
        
        self.weather_transition = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])
        
        self.render_mode = render_mode
        self._state = None
        self._step_count = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        weather = self.np_random.integers(0, 3)
        budget = self.np_random.integers(0, 2)
        last_meal = self.np_random.integers(0, 3)
        self._state = np.array([weather, budget, last_meal], dtype=np.int32)
        self._step_count = 0
        return self._state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        weather, budget, last_meal = self._state
        reward = self._compute_reward(action, budget, last_meal)
        
        new_weather = self._sample_weather(weather)
        new_budget = 1 - budget if self.np_random.random() < 0.3 else budget
        new_last_meal = action
        
        self._state = np.array([new_weather, new_budget, new_last_meal], dtype=np.int32)
        self._step_count += 1
        
        terminated = self._step_count >= 100
        truncated = False
        
        info = {'reward': reward, 'meal': action}
        if self.render_mode == "human":
            self.render()
        
        return self._state, reward, terminated, truncated, info
    
    def _compute_reward(self, action: int, budget: int, last_meal: int) -> float:
        reward = self.satisfaction[action]
        if budget == 1:
            reward -= self.cost[action] * 0.5
        if action != last_meal:
            reward += self.variety_bonus
        return reward
    
    def _sample_weather(self, current: int) -> int:
        return self.np_random.choice(3, p=self.weather_transition[current])
    
    def render(self):
        if self.render_mode is None:
            return
        w, b, m = self._state
        weather_str = ["☀️", "🌧️", "❄️"][w]
        budget_str = ["💰", "💸"][b]
        meal_str = ["🍱", "🛵", "🍙"][m]
        print(f"午餐 | {weather_str} | {budget_str} | 选择：{meal_str}")
    
    def to_mdp(self):
        """转换为表格型 MDP"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from utils.core import TabularMDP
        
        nS = 3 * 2 * 3
        nA = 3
        mdp = TabularMDP(n_states=nS, n_actions=nA, gamma=0.99)
        
        for s in range(nS):
            for a in range(nA):
                transitions = []
                for s_next in range(nS):
                    prob = 1.0 / nS
                    reward = self.satisfaction[a]
                    transitions.append((prob, s_next, reward, False))
                mdp.set_transition(s, a, transitions)
        
        return mdp


if __name__ == "__main__":
    print("测试午餐选择器...\n")
    
    env = LunchDecisionEnv(render_mode="human")
    state, _ = env.reset(seed=42)
    print(f"初始状态：{state}")
    
    for t in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {t+1}: reward={reward:.2f}")
    
    print("\n✅ 测试通过！")
