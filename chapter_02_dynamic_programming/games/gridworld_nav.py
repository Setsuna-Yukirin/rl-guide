"""
网格寻路环境 (GridWorld Navigation)

环境描述：
- NxM 网格世界
- 起点：左上角 (0, 0)
- 终点：右下角，奖励 +10
- 陷阱：随机位置，奖励 -5
- 每步奖励：-1
- 动作：上、下、左、右 (0, 1, 2, 3)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Optional


class GridWorldNav(gym.Env):
    """网格寻路环境"""
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (5, 5),
        n_traps: int = 3,
        trap_reward: float = -5.0,
        goal_reward: float = 10.0,
        step_reward: float = -1.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            grid_size: 网格大小 (rows, cols)
            n_traps: 陷阱数量
            trap_reward: 陷阱奖励
            goal_reward: 终点奖励
            step_reward: 每步奖励
            render_mode: 渲染模式
            seed: 随机种子
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.n_rows, self.n_cols = grid_size
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4  # 上、下、左、右
        
        self.n_traps = n_traps
        self.trap_reward = trap_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        
        self.render_mode = render_mode
        
        # 动作空间
        self.action_space = spaces.Discrete(self.n_actions)
        
        # 状态空间（展平的网格位置）
        self.observation_space = spaces.Discrete(self.n_states)
        
        # 环境状态
        self.agent_pos = None
        self.traps = None
        self.goal_pos = None
        self._action_to_direction = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        if seed is not None:
            np.random.seed(seed)
        
        self._reset_environment()
    
    def _reset_environment(self):
        """初始化环境配置"""
        # 设置终点
        self.goal_pos = (self.n_rows - 1, self.n_cols - 1)
        
        # 随机放置陷阱（避开起点和终点）
        available_positions = [
            (r, c) for r in range(self.n_rows) for c in range(self.n_cols)
            if (r, c) != (0, 0) and (r, c) != self.goal_pos
        ]
        
        if len(available_positions) >= self.n_traps:
            self.traps = set(np.random.choice(
                len(available_positions),
                size=self.n_traps,
                replace=False
            ))
            self.traps = {available_positions[i] for i in self.traps}
        else:
            self.traps = set(available_positions)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """重置环境"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self._reset_environment()
        self.agent_pos = (0, 0)
        
        return self._get_state(), {}
    
    def _get_state(self) -> int:
        """获取当前状态（展平的位置索引）"""
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
        self.agent_pos = (new_r, new_c)
        
        # 计算奖励
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True
        elif self.agent_pos in self.traps:
            reward = self.trap_reward
            terminated = True
        else:
            reward = self.step_reward
            terminated = False
        
        return self._get_state(), reward, terminated, False, {}
    
    def get_transition(self, state: int, action: int) -> List[Tuple[float, int, float, bool]]:
        """
        获取转移概率（用于动态规划）
        
        Returns:
            [(prob, next_state, reward, terminated), ...]
        """
        pos = self._state_to_pos(state)
        dr, dc = self._action_to_direction[action]
        
        # 确定性转移
        new_r = max(0, min(self.n_rows - 1, pos[0] + dr))
        new_c = max(0, min(self.n_cols - 1, pos[1] + dc))
        new_pos = (new_r, new_c)
        next_state = self._pos_to_state(new_pos)
        
        # 计算奖励和终止状态
        if new_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True
        elif new_pos in self.traps:
            reward = self.trap_reward
            terminated = True
        else:
            reward = self.step_reward
            terminated = False
        
        return [(1.0, next_state, reward, terminated)]
    
    def get_legal_actions(self, state: Optional[int] = None) -> List[int]:
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
        
        # 标记陷阱
        for trap in self.traps:
            grid[trap[0]][trap[1]] = 'X'
        
        # 标记终点
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # 标记智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        print('\n'.join([' '.join(row) for row in grid]))
        print()
    
    def _render_ansi(self) -> str:
        """ANSI 字符串渲染"""
        grid = [['.' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        for trap in self.traps:
            grid[trap[0]][trap[1]] = 'X'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        return '\n'.join([' '.join(row) for row in grid])
    
    def draw_policy(self, policy: np.ndarray):
        """绘制策略"""
        arrows = ['↑', '↓', '←', '→']
        grid = []
        
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                state = self._pos_to_state((r, c))
                if (r, c) == self.goal_pos:
                    row.append('G')
                elif (r, c) in self.traps:
                    row.append('X')
                else:
                    action = np.argmax(policy[state])
                    row.append(arrows[action])
            grid.append(' '.join(row))
        
        print('\n'.join(grid))
