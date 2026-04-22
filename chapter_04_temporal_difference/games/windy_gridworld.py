"""
Windy Gridworld 环境

带有风力的网格世界，测试算法在随机环境中的表现
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, List


class WindyGridworldEnv(gym.Env):
    """
    Windy Gridworld 环境
    
    环境描述：
    - 7 行 10 列的网格
    - 起点：(3, 0)
    - 终点：(3, 7)
    - 某些列有向上的风
    - 风强度：0, 1, 或 2
    
    动作：
    - 0: 上
    - 1: 下
    - 2: 左
    - 3: 右
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        windy: bool = True,
        king_moves: bool = False,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            windy: 是否有风
            king_moves: 是否允许对角移动（8 方向）
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.windy = windy
        self.king_moves = king_moves
        
        # 网格大小
        self.n_rows = 7
        self.n_cols = 10
        
        # 动作空间
        if king_moves:
            self.n_actions = 8  # 8 方向移动
            self._action_to_direction = {
                0: (-1, 0),   # 上
                1: (-1, 1),   # 右上
                2: (0, 1),    # 右
                3: (1, 1),    # 右下
                4: (1, 0),    # 下
                5: (1, -1),   # 左下
                6: (0, -1),   # 左
                7: (-1, -1),  # 左上
            }
        else:
            self.n_actions = 4  # 4 方向移动
            self._action_to_direction = {
                0: (-1, 0),  # 上
                1: (1, 0),   # 下
                2: (0, -1),  # 左
                3: (0, 1)    # 右
            }
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)
        
        # 风力配置（每列的风强度）
        self.wind = np.zeros(self.n_cols, dtype=int)
        if windy:
            # 3-5 列：弱风 (1)
            # 6-8 列：强风 (2)
            self.wind[3:6] = 1
            self.wind[6:9] = 2
        
        # 起点和终点
        self.start_pos = (3, 0)
        self.goal_pos = (3, 7)
        
        self.agent_pos = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agent_pos = self.start_pos
        return self._get_state(), {}
    
    def _get_state(self) -> int:
        """获取状态"""
        r, c = self.agent_pos
        return r * self.n_cols + c
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """位置转状态"""
        r, c = pos
        return r * self.n_cols + c
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        执行动作
        
        Returns:
            state: 新状态
            reward: 奖励
            terminated: 是否终止
            truncated: 是否超时
            info: 额外信息
        """
        # 基本移动
        dr, dc = self._action_to_direction[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
        
        # 应用风力（向上吹）
        if self.windy:
            wind_strength = self.wind[self.agent_pos[1]]
            new_r -= wind_strength
        
        # 边界检查
        new_r = max(0, min(self.n_rows - 1, new_r))
        new_c = max(0, min(self.n_cols - 1, new_c))
        
        self.agent_pos = (new_r, new_c)
        
        # 奖励：每步 -1
        reward = -1.0
        
        # 检查是否到达终点
        terminated = self.agent_pos == self.goal_pos
        
        return self._get_state(), reward, terminated, False, {}
    
    def get_legal_actions(self, state: Optional[int] = None) -> list:
        """获取合法动作"""
        return list(range(self.n_actions))
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'ansi':
            return self._render_ansi()
    
    def _render_human(self):
        """人类可读渲染"""
        print("\n===== Windy Gridworld =====")
        grid = [['.' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        # 标记风力
        for c in range(self.n_cols):
            if self.wind[c] > 0:
                for r in range(self.n_rows):
                    if grid[r][c] == '.':
                        grid[r][c] = f'W{self.wind[c]}'
        
        # 标记终点
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # 标记起点
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        
        # 标记智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        print('\n'.join([' '.join(f'{cell:>3}' for cell in row) for row in grid]))
        print("=============================\n")
    
    def _render_ansi(self) -> str:
        """ANSI 渲染"""
        grid = [['.' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        for c in range(self.n_cols):
            if self.wind[c] > 0:
                for r in range(self.n_rows):
                    if grid[r][c] == '.':
                        grid[r][c] = f'W{self.wind[c]}'
        
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        return '\n'.join([' '.join(f'{cell:>3}' for cell in row) for row in grid])
    
    def draw_policy(self, policy: dict):
        """绘制策略"""
        if self.king_moves:
            arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖']
        else:
            arrows = ['↑', '↓', '←', '→']
        
        grid = []
        
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                state = self._pos_to_state((r, c))
                
                if (r, c) == self.goal_pos:
                    row.append('G')
                elif (r, c) == self.start_pos:
                    row.append('S')
                elif state in policy:
                    action = policy[state]
                    row.append(arrows[action])
                else:
                    row.append('?')
            grid.append(' '.join(row))
        
        print('\n'.join(grid))


class WindyGridworldWithStochastic(WindyGridworldEnv):
    """
    随机 Windy Gridworld
    
    风力有随机扰动
    """
    
    def __init__(self, stochastic_factor: float = 1.0, **kwargs):
        """
        Args:
            stochastic_factor: 风力随机因子 (0=确定，1=完全随机)
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        self.stochastic_factor = stochastic_factor
    
    def step(self, action: int) -> Tuple:
        """执行动作（带随机风力）"""
        # 基本移动
        dr, dc = self._action_to_direction[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
        
        # 应用风力（带随机扰动）
        if self.windy:
            base_wind = self.wind[self.agent_pos[1]]
            # 随机扰动：±1
            if np.random.random() < self.stochastic_factor:
                wind_variation = np.random.choice([-1, 0, 1])
            else:
                wind_variation = 0
            wind_strength = max(0, base_wind + wind_variation)
            new_r -= wind_strength
        
        # 边界检查
        new_r = max(0, min(self.n_rows - 1, new_r))
        new_c = max(0, min(self.n_cols - 1, new_c))
        
        self.agent_pos = (new_r, new_c)
        
        reward = -1.0
        terminated = self.agent_pos == self.goal_pos
        
        return self._get_state(), reward, terminated, False, {}
