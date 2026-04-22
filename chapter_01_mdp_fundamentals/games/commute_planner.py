"""
Commute Planner 下班路线规划环境

模拟下班回家路线选择，学习最优通勤策略
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, List, Dict


class CommutePlannerEnv(gym.Env):
    """
    下班路线规划环境
    
    场景描述：
    - 从公司到家的网格地图
    - 不同路段有不同的拥堵程度
    - 目标是找到最快/最舒适的回家路线
    
    状态：当前位置 (row, col)
    动作：上、下、左、右
    奖励：-时间成本（拥堵路段成本更高）
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (5, 5),
        company_pos: Tuple[int, int] = (0, 0),
        home_pos: Tuple[int, int] = None,
        traffic_map: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            grid_size: 地图大小 (rows, cols)
            company_pos: 公司位置
            home_pos: 家位置（默认右下角）
            traffic_map: 拥堵地图（值越大越拥堵）
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.n_rows, self.n_cols = grid_size
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        
        self.company_pos = company_pos
        self.home_pos = home_pos if home_pos else (self.n_rows - 1, self.n_cols - 1)
        self.render_mode = render_mode
        
        # 生成或设置拥堵地图
        if traffic_map is not None:
            self.traffic_map = traffic_map
        else:
            self.traffic_map = self._generate_traffic()
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)
        
        self._action_to_direction = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        self.agent_pos = None
        self.steps = 0
        self.max_steps = self.n_rows * self.n_cols * 2
    
    def _generate_traffic(self) -> np.ndarray:
        """生成随机拥堵地图"""
        # 基础拥堵值 1-3
        traffic = np.random.randint(1, 4, size=(self.n_rows, self.n_cols))
        
        # 设置一些高拥堵区域（如市中心）
        center_r, center_c = self.n_rows // 2, self.n_cols // 2
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = center_r + dr, center_c + dc
                if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                    traffic[r, c] = np.random.randint(4, 6)
        
        # 公司和家不拥堵
        traffic[self.company_pos] = 1
        traffic[self.home_pos] = 1
        
        return traffic
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[int, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            self.traffic_map = self._generate_traffic()
        
        self.agent_pos = self.company_pos
        self.steps = 0
        
        return self._get_state(), {}
    
    def _get_state(self) -> int:
        """获取状态"""
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
        执行动作（选择路段）
        
        Returns:
            state: 新状态
            reward: 奖励（负的时间成本）
            terminated: 是否到达家
            truncated: 是否超时
            info: 额外信息
        """
        self.steps += 1
        
        # 计算新位置
        dr, dc = self._action_to_direction[action]
        new_r = max(0, min(self.n_rows - 1, self.agent_pos[0] + dr))
        new_c = max(0, min(self.n_cols - 1, self.agent_pos[1] + dc))
        self.agent_pos = (new_r, new_c)
        
        # 计算时间成本（拥堵值）
        time_cost = self.traffic_map[self.agent_pos]
        reward = -time_cost  # 负奖励表示成本
        
        # 检查是否到家
        terminated = self.agent_pos == self.home_pos
        
        if terminated:
            reward += 10  # 到达终点的额外奖励
        
        # 检查超时
        truncated = self.steps >= self.max_steps
        
        info = {
            'time_cost': time_cost,
            'total_steps': self.steps,
            'position': self.agent_pos
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def get_legal_actions(self, state: Optional[int] = None) -> List[int]:
        """获取合法动作"""
        if state is None:
            pos = self.agent_pos
        else:
            pos = self._state_to_pos(state)
        
        legal = []
        for action, (dr, dc) in self._action_to_direction.items():
            new_r = pos[0] + dr
            new_c = pos[1] + dc
            if 0 <= new_r < self.n_rows and 0 <= new_c < self.n_cols:
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
        symbols = {
            1: '·',  # 畅通
            2: 'o',  # 缓行
            3: 'O',  # 拥堵
            4: '×',  # 严重拥堵
            5: '※',  # 极度拥堵
        }
        
        grid = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                if (r, c) == self.agent_pos:
                    row.append('A')
                elif (r, c) == self.company_pos:
                    row.append('S')
                elif (r, c) == self.home_pos:
                    row.append('H')
                else:
                    traffic = self.traffic_map[r, c]
                    row.append(symbols.get(traffic, str(traffic)))
            grid.append(' '.join(row))
        
        legend = [
            "\n图例：S=公司 H=家 A=当前位置",
            "路况：·=畅通 o=缓行 O=拥堵 ×=严重拥堵 ※=极度拥堵"
        ]
        
        return '\n'.join(grid) + '\n' + '\n'.join(legend)
    
    def draw_policy(self, policy: Dict[int, int], title: str = "通勤策略"):
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
                if (r, c) == self.company_pos:
                    row.append('S')
                elif (r, c) == self.home_pos:
                    row.append('H')
                else:
                    state = self._pos_to_state((r, c))
                    if state in policy:
                        row.append(arrows[policy[state]])
                    else:
                        row.append('?')
            print(' '.join(row))
        print("=" * (self.n_cols * 2 + 1))


class CommuteWithWeather(CommutePlannerEnv):
    """
    带天气影响的通勤规划
    
    天气会影响拥堵程度
    """
    
    def __init__(
        self,
        weather_effect: Dict[str, float] = None,
        **kwargs
    ):
        """
        Args:
            weather_effect: 天气对拥堵的影响 {'sunny': 1.0, 'rainy': 1.5, 'snowy': 2.0}
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        self.weather_effect = weather_effect or {
            'sunny': 1.0,
            'cloudy': 1.2,
            'rainy': 1.5,
            'snowy': 2.0
        }
        self.weather = 'sunny'
        self.weather_steps = 0
        self.weather_change_interval = 10
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        """重置环境"""
        state, info = super().reset(seed=seed)
        self.weather = 'sunny'
        self.weather_steps = 0
        return state, info
    
    def _get_effective_traffic(self) -> np.ndarray:
        """获取考虑天气后的有效拥堵地图"""
        effect = self.weather_effect.get(self.weather, 1.0)
        return self.traffic_map * effect
    
    def step(self, action: int) -> Tuple:
        """执行动作（考虑天气影响）"""
        # 更新天气
        self.weather_steps += 1
        if self.weather_steps >= self.weather_change_interval:
            self.weather_steps = 0
            self.weather = np.random.choice(list(self.weather_effect.keys()))
        
        # 计算新位置
        dr, dc = self._action_to_direction[action]
        new_r = max(0, min(self.n_rows - 1, self.agent_pos[0] + dr))
        new_c = max(0, min(self.n_cols - 1, self.agent_pos[1] + dc))
        self.agent_pos = (new_r, new_c)
        
        self.steps += 1
        
        # 计算时间成本（考虑天气）
        base_cost = self.traffic_map[self.agent_pos]
        weather_mult = self.weather_effect.get(self.weather, 1.0)
        time_cost = base_cost * weather_mult
        reward = -time_cost
        
        terminated = self.agent_pos == self.home_pos
        if terminated:
            reward += 10
        
        truncated = self.steps >= self.max_steps
        
        info = {
            'time_cost': time_cost,
            'weather': self.weather,
            'total_steps': self.steps,
            'position': self.agent_pos
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def _render_ansi(self) -> str:
        """ANSI 渲染（带天气）"""
        base_render = super()._render_ansi()
        weather_info = f"\n当前天气：{self.weather}"
        return base_render + weather_info
