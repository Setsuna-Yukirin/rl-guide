"""
Flight Chess 飞行棋环境

简化版飞行棋，用于演示多智能体和随机性
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, List, Dict


class FlightChessEnv(gym.Env):
    """
    简化飞行棋环境
    
    规则简化：
    - 单玩家版本
    - 掷骰子前进
    - 特殊格子：后退、暂停、直接胜利
    - 目标：用最少的回合到达终点
    
    状态：当前位置 + 骰子值
    动作：是否使用特殊技能（如果有）
    奖励：-1 每回合，到达终点 +100
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        board_length: int = 50,
        n_players: int = 1,
        special_squares: Optional[Dict[int, int]] = None,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            board_length: 棋盘长度
            n_players: 玩家数量
            special_squares: 特殊格子 {位置：效果}
                           正数=前进，负数=后退，0=暂停一回合
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.board_length = board_length
        self.n_players = n_players
        self.render_mode = render_mode
        
        # 特殊格子（默认设置一些）
        if special_squares is None:
            self.special_squares = {
                5: 10,    # 前进 10 格
                15: -5,   # 后退 5 格
                25: 15,   # 前进 15 格
                35: -10,  # 后退 10 格
                45: 5,    # 前进 5 格
            }
        else:
            self.special_squares = special_squares
        
        self.n_actions = 2  # 0=正常，1=使用技能（如果有）
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0, high=board_length,
            shape=(n_players + 1,),  # 所有玩家位置 + 骰子值
            dtype=np.int8
        )
        
        self.positions = None
        self.dice_value = None
        self.current_player = 0
        self.skip_turn = [False] * n_players
        self.turns = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.positions = [0] * self.n_players
        self.dice_value = 0
        self.current_player = 0
        self.skip_turn = [False] * self.n_players
        self.turns = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        obs = np.array(self.positions + [self.dice_value], dtype=np.int8)
        return obs
    
    def _roll_dice(self) -> int:
        """掷骰子"""
        return np.random.randint(1, 7)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行回合
        
        Returns:
            observation: 观测
            reward: 奖励
            terminated: 是否结束
            truncated: 是否超时
            info: 额外信息
        """
        self.turns += 1
        info = {'player': self.current_player}
        
        # 检查是否被跳过
        if self.skip_turn[self.current_player]:
            self.skip_turn[self.current_player] = False
            info['skipped'] = True
            self.current_player = (self.current_player + 1) % self.n_players
            return self._get_observation(), 0.0, False, False, info
        
        # 掷骰子
        self.dice_value = self._roll_dice()
        info['dice'] = self.dice_value
        
        # 移动
        old_pos = self.positions[self.current_player]
        new_pos = old_pos + self.dice_value
        
        # 检查是否超过终点
        if new_pos >= self.board_length:
            new_pos = self.board_length
            self.positions[self.current_player] = new_pos
            
            # 到达终点
            reward = 100.0 - self.turns * 0.1  # 越快越好
            terminated = True
            
            if self.n_players > 1:
                info['winner'] = self.current_player
            
            return self._get_observation(), reward, terminated, False, info
        
        self.positions[self.current_player] = new_pos
        
        # 检查特殊格子
        if new_pos in self.special_squares:
            effect = self.special_squares[new_pos]
            info['special'] = effect
            
            if effect == 0:
                # 暂停一回合
                self.skip_turn[self.current_player] = True
            else:
                # 前进或后退
                self.positions[self.current_player] = max(0, min(self.board_length, new_pos + effect))
        
        # 奖励：接近终点
        reward = -1.0  # 每回合成本
        
        # 切换玩家
        self.current_player = (self.current_player + 1) % self.n_players
        
        terminated = False
        truncated = self.turns >= self.board_length * self.n_players * 2
        
        return self._get_observation(), reward, terminated, truncated, info
    
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
        lines = [f"===== Flight Chess (Turn {self.turns}) ====="]
        
        # 简化显示：每 5 格标记一次
        for p in range(self.n_players):
            pos = self.positions[p]
            bar = ''
            for i in range(self.board_length):
                if i == pos:
                    bar += f'P{p+1}'
                elif i in self.special_squares:
                    effect = self.special_squares[i]
                    if effect > 0:
                        bar += '↑'
                    elif effect < 0:
                        bar += '↓'
                    else:
                        bar += '⏸'
                else:
                    bar += '·'
            lines.append(f"P{p+1}: {bar}")
        
        lines.append(f"Dice: {self.dice_value}")
        lines.append(f"Current player: P{self.current_player + 1}")
        
        if any(self.skip_turn):
            skip_info = ', '.join(f"P{p+1}:skip" for p, s in enumerate(self.skip_turn) if s)
            lines.append(f"Skipping: {skip_info}")
        
        lines.append("==========================================")
        
        return '\n'.join(lines)


class FlightChessWithItems(FlightChessEnv):
    """
    带道具的飞行棋
    
    玩家可以收集和使用道具
    """
    
    def __init__(
        self,
        n_item_types: int = 3,
        **kwargs
    ):
        """
        Args:
            n_item_types: 道具类型数量
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        self.n_item_types = n_item_types
        self.items = [[] for _ in range(self.n_players)]
        
        # 道具效果
        self.item_effects = {
            0: ('extra_dice', '额外掷骰'),
            1: ('swap', '交换位置'),
            2: ('shield', '免疫一次特殊格'),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        """重置环境"""
        obs, info = super().reset(seed=seed)
        self.items = [[] for _ in range(self.n_players)]
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观测（包含道具）"""
        base_obs = super()._get_observation()
        # 简单处理：返回基础观测
        return base_obs
    
    def step(self, action: int) -> Tuple:
        """执行回合（考虑道具）"""
        # 简化：暂时不实现道具使用逻辑
        return super().step(action)
    
    def collect_item(self, position: int):
        """在特定位置收集道具"""
        if position % 10 == 0 and position > 0:  # 每 10 格有道具
            item_type = np.random.randint(self.n_item_types)
            self.items[self.current_player].append(item_type)
