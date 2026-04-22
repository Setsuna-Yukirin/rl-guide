"""
Warehouse Robot 仓库机器人环境

模拟仓库中机器人收集物品的场景
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, List, Set
from collections import deque


class WarehouseRobotEnv(gym.Env):
    """
    仓库机器人环境
    
    场景描述：
    - 仓库网格地图
    - 机器人需要收集所有物品并送到指定位置
    - 避开障碍物
    - 最小化总移动距离
    
    状态：(机器人位置，已收集物品集合)
    动作：上、下、左、右
    奖励：收集物品 +10，送到目标 +20，每步 -1
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 8),
        n_items: int = 5,
        n_obstacles: int = 8,
        drop_off_pos: Tuple[int, int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            grid_size: 仓库大小
            n_items: 物品数量
            n_obstacles: 障碍物数量
            drop_off_pos: 卸货点位置（默认左上角）
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.n_rows, self.n_cols = grid_size
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        
        self.n_items = n_items
        self.n_obstacles = n_obstacles
        self.drop_off_pos = drop_off_pos if drop_off_pos else (0, 0)
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
        self.robot_pos = None
        self.items: Set[Tuple[int, int]] = set()
        self.collected_items: Set[Tuple[int, int]] = set()
        self.obstacles: Set[Tuple[int, int]] = set()
        self.carrying = False  # 是否携带物品
        
        self.steps = 0
        self.max_steps = grid_size[0] * grid_size[1] * 3
        
        self._generate_warehouse()
    
    def _generate_warehouse(self):
        """生成仓库布局"""
        available = [
            (r, c) for r in range(self.n_rows) for c in range(self.n_cols)
            if (r, c) != self.drop_off_pos
        ]
        
        # 随机放置障碍物
        if len(available) >= self.n_obstacles:
            obs_indices = np.random.choice(len(available), size=self.n_obstacles, replace=False)
            self.obstacles = {available[i] for i in obs_indices}
            available = [p for p in available if p not in self.obstacles]
        
        # 随机放置物品
        if len(available) >= self.n_items:
            item_indices = np.random.choice(len(available), size=self.n_items, replace=False)
            self.items = {available[i] for i in item_indices}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            self._generate_warehouse()
        
        # 机器人从卸货点出发
        self.robot_pos = self.drop_off_pos
        self.collected_items = set()
        self.carrying = False
        self.steps = 0
        
        return self._get_state(), {}
    
    def _get_state(self) -> int:
        """获取状态（机器人位置）"""
        r, c = self.robot_pos
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
        """检查位置是否有效"""
        r, c = pos
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return pos not in self.obstacles
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        执行动作
        
        Returns:
            state: 新状态
            reward: 奖励
            terminated: 是否完成（收集所有物品并送到）
            truncated: 是否超时
            info: 额外信息
        """
        self.steps += 1
        reward = -1  # 每步成本
        
        # 计算新位置
        dr, dc = self._action_to_direction[action]
        new_pos = (self.robot_pos[0] + dr, self.robot_pos[1] + dc)
        
        # 检查是否撞障碍
        if self._is_valid_pos(new_pos):
            self.robot_pos = new_pos
        
        terminated = False
        
        # 检查是否拾取物品
        if not self.carrying and self.robot_pos in self.items:
            self.carrying = True
            reward += 10  # 拾取奖励
            self.items.discard(self.robot_pos)
        
        # 检查是否卸货
        if self.carrying and self.robot_pos == self.drop_off_pos:
            self.carrying = False
            self.collected_items.add(self.robot_pos)
            reward += 20  # 卸货奖励
        
        # 检查是否完成（所有物品收集并送到）
        if len(self.items) == 0 and not self.carrying:
            terminated = True
            reward += 50  # 完成奖励
        
        truncated = self.steps >= self.max_steps
        
        info = {
            'items_remaining': len(self.items),
            'collected': len(self.collected_items),
            'carrying': self.carrying,
            'steps': self.steps
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def get_legal_actions(self, state: Optional[int] = None) -> List[int]:
        """获取合法动作"""
        if state is None:
            pos = self.robot_pos
        else:
            pos = self._state_to_pos(state)
        
        legal = []
        for action, (dr, dc) in self._action_to_direction.items():
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_pos(new_pos):
                legal.append(action)
        
        return legal if legal else [0, 1, 2, 3]
    
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
                if pos == self.robot_pos:
                    row.append('R' if not self.carrying else 'r')
                elif pos == self.drop_off_pos:
                    row.append('D')
                elif pos in self.items:
                    row.append('I')
                elif pos in self.obstacles:
                    row.append('#')
                else:
                    row.append('.')
            grid.append(' '.join(row))
        
        info = [
            f"Items remaining: {len(self.items)}",
            f"Collected: {len(self.collected_items)}",
            f"Carrying: {self.carrying}"
        ]
        
        return '\n'.join(grid) + '\n' + '\n'.join(info)


class WarehouseRobotWithPriority(WarehouseRobotEnv):
    """
    带优先级的仓库机器人
    
    不同物品有不同优先级（价值）
    """
    
    def __init__(
        self,
        priority_levels: int = 3,
        **kwargs
    ):
        """
        Args:
            priority_levels: 优先级数量
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        self.priority_levels = priority_levels
        self.item_priorities = {}  # {(r,c): priority}
    
    def _generate_warehouse(self):
        """生成带优先级的仓库"""
        super()._generate_warehouse()
        
        # 为每个物品分配优先级
        for item in self.items:
            self.item_priorities[item] = np.random.randint(1, self.priority_levels + 1)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        """重置环境"""
        state, info = super().reset(seed=seed)
        self.item_priorities = {}
        for item in self.items:
            self.item_priorities[item] = np.random.randint(1, self.priority_levels + 1)
        return state, info
    
    def step(self, action: int) -> Tuple:
        """执行动作（考虑优先级）"""
        _, reward, terminated, truncated, info = super().step(action)
        
        # 高优先级物品额外奖励
        if self.robot_pos in self.item_priorities:
            priority = self.item_priorities[self.robot_pos]
            reward += priority * 2  # 优先级越高奖励越多
        
        return self._get_state(), reward, terminated, truncated, info
