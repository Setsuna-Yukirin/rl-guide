"""
Blackjack (21 点) 环境

简化版 Blackjack，用于强化学习教学
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict


class BlackjackEnv(gym.Env):
    """
    Blackjack 环境
    
    状态空间：(玩家点数，庄家明牌，是否有可用 A)
    动作空间：要牌 (0) / 停牌 (1)
    
    规则简化：
    - 只用一副牌
    - A 可计为 1 或 11
    - 庄家明牌可见
    - 黑杰克 (Blackjack) 奖励 1.5 倍
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # 动作空间：0=要牌 (Hit), 1=停牌 (Stand)
        self.action_space = spaces.Discrete(2)
        
        # 状态空间
        # 玩家点数：12-21 (10 个值)
        # 庄家明牌：1-10 (10 个值)
        # 可用 A: 0 或 1 (2 个值)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # 玩家点数 4-35
            spaces.Discrete(11),  # 庄家明牌 1-10
            spaces.Discrete(2)    # 可用 A
        ))
        
        # 牌堆
        self.deck = None
        self.player_hand = None
        self.dealer_hand = None
        
        self._card_values = {
            'A': 11, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
            'J': 10, 'Q': 10, 'K': 10
        }
    
    def _create_deck(self) -> list:
        """创建一副牌"""
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        deck = [(rank, suit) for rank in ranks for suit in suits]
        np.random.shuffle(deck)
        return deck
    
    def _deal_card(self) -> Tuple[str, str]:
        """发一张牌"""
        if not self.deck or len(self.deck) == 0:
            self.deck = self._create_deck()
        return self.deck.pop()
    
    def _calculate_hand(self, hand: list) -> Tuple[int, bool]:
        """
        计算手牌点数
        
        Returns:
            total: 总点数
            usable_ace: 是否有可用的 A (计为 11 而不爆)
        """
        total = 0
        usable_ace = False
        
        for rank, _ in hand:
            value = self._card_values[rank]
            total += value
            if rank == 'A' and value == 11:
                usable_ace = True
        
        # 如果爆牌且有可用 A，将 A 计为 1
        while total > 21 and usable_ace:
            total -= 10
            usable_ace = False
        
        return total, usable_ace
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.deck = self._create_deck()
        
        # 发牌：玩家两张，庄家两张（一张明牌）
        self.player_hand = [self._deal_card(), self._deal_card()]
        self.dealer_hand = [self._deal_card(), self._deal_card()]
        
        # 检查黑杰克
        player_total, _ = self._calculate_hand(self.player_hand)
        dealer_total, _ = self._calculate_hand(self.dealer_hand)
        
        # 返回 (observation, info)
        return self._get_state(), {}
    
    def _get_state(self) -> Tuple[int, int, int]:
        """获取当前状态"""
        player_total, usable_ace = self._calculate_hand(self.player_hand)
        dealer_showing = self._card_values[self.dealer_hand[0][0]]
        
        # 处理 A 的情况
        if self.dealer_hand[0][0] == 'A':
            dealer_showing = 1
        elif dealer_showing > 10:
            dealer_showing = 10
        
        return (player_total, dealer_showing, int(usable_ace))
    
    def step(self, action: int) -> Tuple:
        """
        执行动作
        
        Args:
            action: 0=要牌，1=停牌
        
        Returns:
            state: 新状态
            reward: 奖励
            terminated: 是否终止
            truncated: 是否超时
            info: 额外信息
        """
        if action == 0:  # 要牌
            self.player_hand.append(self._deal_card())
            player_total, _ = self._calculate_hand(self.player_hand)
            
            if player_total > 21:
                # 玩家爆牌，输
                return self._get_state(), -1.0, True, False, {}
            else:
                return self._get_state(), 0.0, False, False, {}
        
        else:  # 停牌
            # 庄家回合
            return self._play_dealer()
    
    def _play_dealer(self) -> Tuple:
        """庄家回合"""
        while True:
            dealer_total, _ = self._calculate_hand(self.dealer_hand)
            if dealer_total < 17:
                self.dealer_hand.append(self._deal_card())
            else:
                break
        
        # 计算结果
        player_total, _ = self._calculate_hand(self.player_hand)
        dealer_total, _ = self._calculate_hand(self.dealer_hand)
        
        if dealer_total > 21:
            # 庄家爆牌，玩家赢
            reward = 1.0
        elif player_total > dealer_total:
            # 玩家点数大
            reward = 1.0
        elif player_total < dealer_total:
            # 庄家点数大
            reward = -1.0
        else:
            # 平局
            reward = 0.0
        
        return self._get_state(), reward, True, False, {}
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'ansi':
            return self._render_ansi()
    
    def _render_human(self):
        """人类可读渲染"""
        print("\n===== Blackjack =====")
        print(f"庄家：{self.dealer_hand[0]} [?]")
        print(f"玩家：{self.player_hand} = {self._calculate_hand(self.player_hand)[0]}")
        print("=====================\n")
    
    def _render_ansi(self) -> str:
        """ANSI 渲染"""
        player_total, _ = self._calculate_hand(self.player_hand)
        lines = [
            "===== Blackjack =====",
            f"庄家：{self.dealer_hand[0]} [?]",
            f"玩家：{self.player_hand} = {player_total}",
            "====================="
        ]
        return '\n'.join(lines)
    
    def get_legal_actions(self, state: Optional[Tuple] = None) -> list:
        """获取合法动作"""
        return [0, 1]  # 总是可以要牌或停牌


# 简化版状态表示（用于表格型方法）
def state_to_index(state: Tuple[int, int, int]) -> int:
    """
    将状态转换为索引（用于表格存储）
    
    Args:
        state: (player_total, dealer_showing, usable_ace)
    
    Returns:
        index: 状态索引
    """
    player_total, dealer_showing, usable_ace = state
    
    # 只考虑 12-21 的玩家点数
    if player_total < 12:
        player_idx = 0
    elif player_total > 21:
        player_idx = 10
    else:
        player_idx = player_total - 12
    
    dealer_idx = min(dealer_showing - 1, 9)
    ace_idx = usable_ace
    
    return player_idx * 20 + dealer_idx * 2 + ace_idx


def index_to_state(index: int) -> Tuple[int, int, int]:
    """索引转状态"""
    player_idx = index // 20
    dealer_idx = (index % 20) // 2
    ace_idx = index % 2
    
    player_total = player_idx + 12 if player_idx < 10 else 22
    dealer_showing = dealer_idx + 1
    usable_ace = ace_idx
    
    return (player_total, dealer_showing, usable_ace)
