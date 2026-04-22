"""
Maze Treasure 迷宫寻宝环境

在迷宫中找到宝藏，同时避开陷阱
用于测试 TD 算法在复杂环境中的表现
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, List, Set


class MazeTreasureEnv(gym.Env):
    """
    迷宫寻宝环境
    
    环境描述：
    - NxM 网格迷宫
    - 起点：左上角
    - 宝藏：随机位置，奖励 +10
    - 陷阱：随机位置，奖励 -5
    - 墙壁：不可通过
    - 每步奖励：-1
    
    动作：
    - 0: 上
    - 1: 下
    - 2: 左
    - 3: 右
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        maze_size: Tuple[int, int] = (8, 8),
        n_traps: int = 5,
        n_walls: int = 10,
        treasure_reward: float = 10.0,
        trap_reward: float = -5.0,
        step_reward: float = -1.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            maze_size: 迷宫大小 (rows, cols)
            n_traps: 陷阱数量
            n_walls: 墙壁数量
            treasure_reward: 宝藏奖励
            trap_reward: 陷阱奖励
            step_reward: 每步奖励
            render_mode: 渲染模式
            seed: 随机种子
        """
        super().__init__()
        
        self.n_rows, self.n_cols = maze_size
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        
        self.n_traps = n_traps
        self.n_walls = n_walls
        self.treasure_reward = treasure_reward
        self.trap_reward = trap_reward
        self.step_reward = step_reward
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)
        
        self._action_to_direction = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        # 环境状态
        self.agent_pos = None
        self.treasure_pos = None
        self.traps: Set[Tuple[int, int]] = set()
        self.walls: Set[Tuple[int, int]] = set()
        
        if seed is not None:
            np.random.seed(seed)
        
        self._generate_maze()
    
    def _generate_maze(self):
        """生成迷宫（随机放置宝藏、陷阱、墙壁）"""
        # 起点和终点（宝藏）
        self.start_pos = (0, 0)
        self.treasure_pos = (self.n_rows - 1, self.n_cols - 1)
        
        # 可用位置（排除起点和宝藏）
        available = [
            (r, c) for r in range(self.n_rows) for c in range(self.n_cols)
            if (r, c) != self.start_pos and (r, c) != self.treasure_pos
        ]
        
        # 随机放置陷阱
        if len(available) >= self.n_traps:
            trap_indices = np.random.choice(len(available), size=self.n_traps, replace=False)
            self.traps = {available[i] for i in trap_indices}
            available = [p for p in available if p not in self.traps]
        
        # 随机放置墙壁
        if len(available) >= self.n_walls:
            wall_indices = np.random.choice(len(available), size=self.n_walls, replace=False)
            self.walls = {available[i] for i in wall_indices}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agent_pos = self.start_pos
        self._generate_maze()
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
    
    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效（不是墙壁且在边界内）"""
        r, c = pos
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return pos not in self.walls
    
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
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        
        # 检查是否撞墙
        if self._is_valid_pos(new_pos):
            self.agent_pos = new_pos
        
        # 计算奖励
        if self.agent_pos == self.treasure_pos:
            reward = self.treasure_reward
            terminated = True
        elif self.agent_pos in self.traps:
            reward = self.trap_reward
            terminated = True
        else:
            reward = self.step_reward
            terminated = False
        
        return self._get_state(), reward, terminated, False, {}
    
    def get_legal_actions(self, state: Optional[int] = None) -> List[int]:
        """获取合法动作（排除撞墙的动作）"""
        if state is None:
            pos = self.agent_pos
        else:
            pos = self._state_to_pos(state)
        
        legal = []
        for action, (dr, dc) in self._action_to_direction.items():
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_pos(new_pos):
                legal.append(action)
        
        return legal if legal else [0, 1, 2, 3]  # 如果都被堵死，返回全部
    
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
        grid = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                pos = (r, c)
                if pos == self.agent_pos:
                    row.append('A')
                elif pos == self.treasure_pos:
                    row.append('T')
                elif pos in self.traps:
                    row.append('X')
                elif pos in self.walls:
                    row.append('#')
                else:
                    row.append('.')
            row.append('')
            grid.append(' '.join(row))
        return '\n'.join(grid)
    
    def draw_policy(self, policy: dict, title: str = "Maze Policy"):
        """
        绘制策略
        
        Args:
            policy: {state: action}
            title: 标题
        """
        arrows = ['↑', '↓', '←', '→']
        print(f"\n{title}")
        print("=" * (self.n_cols * 2 + 1))
        
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                pos = (r, c)
                if pos in self.walls:
                    row.append('#')
                elif pos == self.treasure_pos:
                    row.append('T')
                elif pos in self.traps:
                    row.append('X')
                else:
                    state = self._pos_to_state(pos)
                    if state in policy:
                        row.append(arrows[policy[state]])
                    else:
                        row.append('?')
            print(' '.join(row))
        print("=" * (self.n_cols * 2 + 1))


class MazeTreasureWithFog(MazeTreasureEnv):
    """
    带战争迷雾的迷宫寻宝
    
    智能体只能看到周围有限范围
    """
    
    def __init__(self, view_range: int = 2, **kwargs):
        """
        Args:
            view_range: 视野范围（能看到周围几格）
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        self.view_range = view_range
        
        # 部分可观测状态空间
        # 每个位置可以是：未知、空、墙、陷阱、宝藏
        self.observation_space = spaces.Box(
            low=0, high=4,
            shape=(2 * view_range + 1, 2 * view_range + 1),
            dtype=np.int32
        )
    
    def _get_local_observation(self) -> np.ndarray:
        """获取局部观测（战争迷雾）"""
        obs_size = 2 * self.view_range + 1
        obs = np.zeros((obs_size, obs_size), dtype=np.int32)
        
        ar, ac = self.agent_pos
        
        for i in range(obs_size):
            for j in range(obs_size):
                r = ar - self.view_range + i
                c = ac - self.view_range + j
                
                if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
                    obs[i, j] = 4  # 边界外
                elif (r, c) in self.walls:
                    obs[i, j] = 3  # 墙
                elif (r, c) in self.traps:
                    obs[i, j] = 2  # 陷阱（只有靠近才能看到）
                elif (r, c) == self.treasure_pos:
                    obs[i, j] = 1  # 宝藏（只有靠近才能看到）
                else:
                    obs[i, j] = 0  # 空地
        
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        super().reset(seed=seed)
        return self._get_local_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行动作"""
        _, reward, terminated, truncated, info = super().step(action)
        obs = self._get_local_observation()
        return obs, reward, terminated, truncated, info
