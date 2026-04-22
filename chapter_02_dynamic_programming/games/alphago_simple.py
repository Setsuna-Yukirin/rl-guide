"""
AlphaGo Simple 简易围棋环境

简化版围棋，用于演示 MCTS 算法
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, List, Set


class AlphaGoSimpleEnv(gym.Env):
    """
    简易围棋环境
    
    规则简化：
    - 5x5 小棋盘
    - 无打劫规则
    - 无复杂死活判断
    - 简单的气计算
    - 黑棋先手
    
    状态：棋盘状态 (0=空，1=黑，2=白) + 当前玩家
    动作：落子位置 (0-24) 或 停手 (25)
    奖励：赢 +1，输 -1，平局 0
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        board_size: int = 5,
        komi: float = 3.5,
        max_moves: int = 100,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            board_size: 棋盘大小
            komi: 贴目（补偿白棋）
            max_moves: 最大手数
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.board_size = board_size
        self.n_states = board_size * board_size
        self.n_actions = board_size * board_size + 1  # +1 for pass
        
        self.komi = komi
        self.max_moves = max_moves
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=(board_size, board_size),
            dtype=np.int8
        )
        
        # 游戏状态
        self.board = None
        self.current_player = 1  # 1=黑，2=白
        self.captures = {1: 0, 2: 0}  # 各方提子数
        self.move_history = []
        self.passes = 0
        self.moves = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.captures = {1: 0, 2: 0}
        self.move_history = []
        self.passes = 0
        self.moves = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取观测（棋盘状态）"""
        return self.board.copy()
    
    def _get_liberties(self, pos: Tuple[int, int]) -> int:
        """
        计算棋子的气
        
        Args:
            pos: 棋子位置
        
        Returns:
            气的数量
        """
        r, c = pos
        color = self.board[r, c]
        if color == 0:
            return 0
        
        # 找到相连的同色棋子组
        group = {pos}
        queue = [pos]
        
        while queue:
            curr = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        group.add((nr, nc))
                        queue.append((nr, nc))
        
        # 计算组的气
        liberties = set()
        for gr, gc in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == 0:
                        liberties.add((nr, nc))
        
        return len(liberties)
    
    def _remove_group(self, pos: Tuple[int, int]) -> int:
        """
        移除棋子组
        
        Returns:
            移除的棋子数量
        """
        r, c = pos
        color = self.board[r, c]
        
        group = {pos}
        queue = [pos]
        
        while queue:
            curr = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        group.add((nr, nc))
                        queue.append((nr, nc))
        
        # 移除
        for gr, gc in group:
            self.board[gr, gc] = 0
        
        return len(group)
    
    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        """检查落子是否合法"""
        r, c = pos
        
        # 检查是否在棋盘内
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False
        
        # 检查是否已有棋子
        if self.board[r, c] != 0:
            return False
        
        # 模拟落子
        self.board[r, c] = self.current_player
        
        # 检查是否有气
        has_liberties = self._get_liberties(pos) > 0
        
        # 检查是否能提子
        can_capture = False
        opponent = 3 - self.current_player
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == opponent:
                    if self._get_liberties((nr, nc)) == 0:
                        can_capture = True
                        break
        
        # 撤销模拟
        self.board[r, c] = 0
        
        # 自杀规则：无气且不能提子则非法
        if not has_liberties and not can_capture:
            return False
        
        return True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作（落子）
        
        Args:
            action: 0-24=落子位置，25=停手
        
        Returns:
            observation: 棋盘状态
            reward: 奖励（游戏结束时）
            terminated: 是否结束
            truncated: 是否超时
            info: 额外信息
        """
        self.moves += 1
        info = {'player': self.current_player}
        
        # 停手
        if action == self.n_actions - 1:
            self.passes += 1
            self.current_player = 3 - self.current_player
            
            # 连续两次停手则结束
            if self.passes >= 2:
                reward = self._compute_final_reward()
                return self._get_observation(), reward, True, False, info
            
            return self._get_observation(), 0.0, False, False, info
        
        self.passes = 0
        
        # 落子位置
        r = action // self.board_size
        c = action % self.board_size
        pos = (r, c)
        
        # 检查合法性
        if not self._is_valid_move(pos):
            # 非法移动，判负
            info['reason'] = 'illegal_move'
            reward = -1 if self.current_player == 1 else 1
            return self._get_observation(), reward, True, False, info
        
        # 执行落子
        self.board[r, c] = self.current_player
        self.move_history.append((r, c))
        
        # 检查提子
        opponent = 3 - self.current_player
        captured = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == opponent:
                    if self._get_liberties((nr, nc)) == 0:
                        captured += self._remove_group((nr, nc))
        
        self.captures[self.current_player] += captured
        
        # 检查游戏结束
        terminated = False
        reward = 0.0
        
        # 一方无子可下
        if self._count_legal_moves() == 0:
            terminated = True
            reward = self._compute_final_reward()
        
        # 超过最大手数
        if self.moves >= self.max_moves:
            terminated = True
            reward = self._compute_final_reward()
        
        # 切换玩家
        self.current_player = 3 - self.current_player
        
        return self._get_observation(), reward, terminated, False, info
    
    def _count_legal_moves(self) -> int:
        """计算合法移动数量"""
        count = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self._is_valid_move((r, c)):
                    count += 1
        return count
    
    def _compute_final_reward(self) -> float:
        """计算最终奖励（基于地盘）"""
        black_territory = self._count_territory(1)
        white_territory = self._count_territory(2)
        
        # 计算得分
        black_score = black_territory + self.captures[1]
        white_score = white_territory + self.captures[2] + self.komi
        
        if self.current_player == 1:
            if black_score > white_score:
                return 1.0
            elif black_score < white_score:
                return -1.0
            else:
                return 0.0
        else:
            if white_score > black_score:
                return 1.0
            elif white_score < black_score:
                return -1.0
            else:
                return 0.0
    
    def _count_territory(self, player: int) -> int:
        """计算玩家的地盘"""
        territory = 0
        visited = set()
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0 and (r, c) not in visited:
                    # BFS 找到相连的空区域
                    region = {(r, c)}
                    queue = [(r, c)]
                    borders_player = True
                    borders_opponent = False
                    
                    while queue:
                        curr = queue.pop(0)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr[0] + dr, curr[1] + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                if self.board[nr, nc] == 0 and (nr, nc) not in region:
                                    region.add((nr, nc))
                                    queue.append((nr, nc))
                                elif self.board[nr, nc] == player:
                                    borders_player = True
                                elif self.board[nr, nc] == 3 - player:
                                    borders_opponent = True
                    
                    visited.update(region)
                    
                    # 只被一方包围才算地盘
                    if borders_player and not borders_opponent:
                        territory += len(region)
        
        return territory
    
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
        symbols = {0: '.', 1: '●', 2: '○'}
        
        lines = []
        lines.append(f"  {' '.join(chr(ord('a') + c) for c in range(self.board_size))}")
        
        for r in range(self.board_size):
            row = [str(self.board_size - r)]
            for c in range(self.board_size):
                row.append(symbols[self.board[r, c]])
            lines.append(' '.join(row))
        
        info = [
            f"Black captures: {self.captures[1]}",
            f"White captures: {self.captures[2]}",
            f"Current player: {'Black' if self.current_player == 1 else 'White'}",
            f"Moves: {self.moves}"
        ]
        
        return '\n'.join(lines) + '\n' + '\n'.join(info)
    
    def get_legal_actions(self, state: Optional[np.ndarray] = None) -> List[int]:
        """获取合法动作"""
        legal = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self._is_valid_move((r, c)):
                    action = r * self.board_size + c
                    legal.append(action)
        
        # 总是可以停手
        legal.append(self.n_actions - 1)
        
        return legal if legal else [self.n_actions - 1]
