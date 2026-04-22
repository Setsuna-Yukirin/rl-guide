"""
Cliff Walking 环境

经典的强化学习测试环境，用于比较在策略和离策略算法
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional


class CliffWalkingEnv(gym.Env):
    """
    Cliff Walking 环境
    
    环境描述：
    - 4 行 12 列的网格
    - 起点：左下角 (3, 0)
    - 终点：右下角 (3, 11)
    - 悬崖：底部一行 (3, 1) 到 (3, 10)
    - 掉下悬崖：奖励 -100，回到起点
    - 每步奖励：-1
    
    动作：
    - 0: 上
    - 1: 下
    - 2: 左
    - 3: 右
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # 网格大小
        self.n_rows = 4
        self.n_cols = 12
        
        # 动作空间
        self.action_space = spaces.Discrete(4)
        
        # 状态空间（展平的位置）
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)
        
        # 动作到方向的映射
        self._action_to_direction = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        # 起点和终点
        self.start_pos = (3, 0)
        self.goal_pos = (3, 11)
        
        # 悬崖位置
        self.cliff = [(3, i) for i in range(1, 11)]
        
        # 当前状态
        self.agent_pos = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agent_pos = self.start_pos
        return self._get_state(), {}
    
    def _get_state(self) -> int:
        """获取状态（展平的位置索引）"""
        r, c = self.agent_pos
        return r * self.n_cols + c
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """位置转状态"""
        r, c = pos
        return r * self.n_cols + c
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """状态转位置"""
        r = state // self.n_cols
        c = state % self.n_cols
        return (r, c)
    
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
        # 计算新位置
        dr, dc = self._action_to_direction[action]
        new_r = max(0, min(self.n_rows - 1, self.agent_pos[0] + dr))
        new_c = max(0, min(self.n_cols - 1, self.agent_pos[1] + dc))
        new_pos = (new_r, new_c)
        
        # 检查是否掉下悬崖
        if new_pos in self.cliff:
            # 掉下悬崖
            reward = -100.0
            terminated = True
            self.agent_pos = self.start_pos  # 回到起点
        elif new_pos == self.goal_pos:
            # 到达终点
            reward = -1.0
            terminated = True
            self.agent_pos = new_pos
        else:
            # 正常移动
            reward = -1.0
            terminated = False
            self.agent_pos = new_pos
        
        return self._get_state(), reward, terminated, False, {}
    
    def get_legal_actions(self, state: Optional[int] = None) -> list:
        """获取合法动作（所有动作都合法）"""
        return [0, 1, 2, 3]
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'ansi':
            return self._render_ansi()
    
    def _render_human(self):
        """人类可读渲染"""
        grid = [['.' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        # 标记悬崖
        for r, c in self.cliff:
            grid[r][c] = 'C'
        
        # 标记终点
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # 标记起点
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        
        # 标记智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        print('\n'.join([' '.join(row) for row in grid]))
        print()
    
    def _render_ansi(self) -> str:
        """ANSI 渲染"""
        grid = [['.' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        for r, c in self.cliff:
            grid[r][c] = 'C'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        return '\n'.join([' '.join(row) for row in grid])
    
    def draw_policy(self, policy: dict):
        """
        绘制策略
        
        Args:
            policy: {state: action}
        """
        arrows = ['↑', '↓', '←', '→']
        grid = []
        
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                state = self._pos_to_state((r, c))
                
                if (r, c) in self.cliff:
                    row.append('C')
                elif (r, c) == self.goal_pos:
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
