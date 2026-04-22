"""
Snake Simple 简单贪吃蛇环境

经典贪吃蛇游戏的简化版，用于 RL 学习
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, List
from collections import deque


class SnakeSimpleEnv(gym.Env):
    """
    简单贪吃蛇环境
    
    环境描述：
    - NxM 网格
    - 蛇：初始长度为 3
    - 食物：随机位置，奖励 +10
    - 撞墙或撞自己：游戏结束，奖励 -10
    - 每步奖励：-0.1（鼓励快速吃到食物）
    
    动作：
    - 0: 上
    - 1: 下
    - 2: 左
    - 3: 右
    
    注意：不能直接反向移动（如正在向右不能直接向左）
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        initial_length: int = 3,
        food_reward: float = 10.0,
        death_reward: float = -10.0,
        step_reward: float = -0.1,
        max_steps: int = 200,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            grid_size: 网格大小 (rows, cols)
            initial_length: 初始蛇长度
            food_reward: 食物奖励
            death_reward: 死亡奖励
            step_reward: 每步奖励
            max_steps: 最大步数（防止无限循环）
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.n_rows, self.n_cols = grid_size
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        
        self.initial_length = initial_length
        self.food_reward = food_reward
        self.death_reward = death_reward
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)
        
        self._action_to_direction = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        # 相反的动作
        self._opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
        
        # 环境状态
        self.snake: deque = None  # 蛇身体（双端队列）
        self.food_pos = None
        self.direction = 3  # 初始方向：向右
        self.steps = 0
        self.length = initial_length
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化蛇（从中间开始）
        start_r = self.n_rows // 2
        start_c = self.n_cols // 2
        
        self.snake = deque()
        for i in range(self.initial_length):
            self.snake.append((start_r, start_c - i))
        
        self.direction = 3  # 向右
        self.steps = 0
        self.length = self.initial_length
        
        # 放置食物
        self._place_food()
        
        return self._get_state(), {}
    
    def _place_food(self):
        """在空位置放置食物"""
        snake_set = set(self.snake)
        empty_positions = [
            (r, c) for r in range(self.n_rows) for c in range(self.n_cols)
            if (r, c) not in snake_set
        ]
        
        if empty_positions:
            self.food_pos = empty_positions[np.random.randint(len(empty_positions))]
        else:
            self.food_pos = None  # 没有空位了（胜利）
    
    def _get_state(self) -> int:
        """获取状态（蛇头位置）"""
        head_r, head_c = self.snake[0]
        return head_r * self.n_cols + head_c
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """位置转状态"""
        r, c = pos
        return r * self.n_cols + c
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """状态转位置"""
        r = state // self.n_cols
        c = state % self.n_cols
        return (r, c)
    
    def get_legal_actions(self, state: Optional[int] = None) -> List[int]:
        """获取合法动作（不能直接反向）"""
        opposite = self._opposite_actions[self.direction]
        return [a for a in range(4) if a != opposite]
    
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
        self.steps += 1
        
        # 检查是否是合法动作（不能直接反向）
        opposite = self._opposite_actions[self.direction]
        if action == opposite:
            # 忽略非法动作，保持原方向
            action = self.direction
        
        self.direction = action
        
        # 计算新蛇头位置
        dr, dc = self._action_to_direction[action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)
        
        # 检查是否撞墙
        if (new_head[0] < 0 or new_head[0] >= self.n_rows or
            new_head[1] < 0 or new_head[1] >= self.n_cols):
            return self._get_state(), self.death_reward, True, False, {"reason": "wall"}
        
        # 检查是否撞自己（不包括尾巴，因为尾巴会移动）
        snake_body = set(list(self.snake)[:-1])
        if new_head in snake_body:
            return self._get_state(), self.death_reward, True, False, {"reason": "self"}
        
        # 移动蛇
        self.snake.appendleft(new_head)
        
        # 检查是否吃到食物
        if new_head == self.food_pos:
            # 吃到食物，蛇变长
            self.length += 1
            reward = self.food_reward
            self._place_food()
            
            # 检查是否胜利（填满整个网格）
            if self.food_pos is None:
                return self._get_state(), reward + 100, True, False, {"reason": "win"}
        else:
            # 没吃到食物，移除尾巴
            self.snake.pop()
            reward = self.step_reward
        
        # 检查是否超过最大步数
        if self.steps >= self.max_steps:
            return self._get_state(), self.death_reward, True, True, {"reason": "timeout"}
        
        return self._get_state(), reward, False, False, {}
    
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
        grid = [['.' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        # 画食物
        if self.food_pos:
            grid[self.food_pos[0]][self.food_pos[1]] = 'F'
        
        # 画蛇
        for i, (r, c) in enumerate(self.snake):
            if i == 0:
                grid[r][c] = 'H'  # 蛇头
            else:
                grid[r][c] = 'S'  # 蛇身
        
        lines = [f"Score: {self.length - self.initial_length}, Steps: {self.steps}"]
        lines.append('=' * (self.n_cols * 2 + 1))
        for row in grid:
            lines.append(' '.join(row))
        lines.append('=' * (self.n_cols * 2 + 1))
        
        return '\n'.join(lines)
    
    def get_snake_observation(self) -> np.ndarray:
        """
        获取完整的蛇状态观测（用于可视化或复杂策略）
        
        Returns:
            grid: (n_rows, n_cols, 3) 张量
                  - 通道 0: 蛇身体
                  - 通道 1: 食物
                  - 通道 2: 边界
        """
        obs = np.zeros((self.n_rows, self.n_cols, 3), dtype=np.float32)
        
        # 蛇身体
        for r, c in self.snake:
            obs[r, c, 0] = 1.0
        
        # 食物
        if self.food_pos:
            obs[self.food_pos[0], self.food_pos[1], 1] = 1.0
        
        # 边界
        obs[0, :, 2] = 1.0
        obs[-1, :, 2] = 1.0
        obs[:, 0, 2] = 1.0
        obs[:, -1, 2] = 1.0
        
        return obs


class SnakeWithDirection(SnakeSimpleEnv):
    """
    带方向信息的贪吃蛇
    
    状态包含蛇头位置和当前方向
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 状态空间：位置 + 方向
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.n_states),  # 位置
            spaces.Discrete(4)  # 方向
        ))
    
    def _get_state(self) -> Tuple[int, int]:
        """获取状态（位置，方向）"""
        pos = super()._get_state()
        return (pos, self.direction)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        """重置环境"""
        super().reset(seed=seed)
        return self._get_state(), {}
    
    def step(self, action: int) -> Tuple:
        """执行动作"""
        _, reward, terminated, truncated, info = super().step(action)
        return self._get_state(), reward, terminated, truncated, info
