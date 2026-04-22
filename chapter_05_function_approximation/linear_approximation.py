"""
线性函数近似

使用线性模型近似价值函数
"""

import numpy as np
from typing import Callable, Optional, Tuple


class LinearValueFunction:
    """
    线性价值函数
    
    V(s, w) = w^T * φ(s)
    """
    
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99
    ):
        """
        Args:
            n_features: 特征数量
            learning_rate: 学习率
            gamma: 折扣因子
        """
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        
        # 权重初始化
        self.weights = np.zeros(n_features)
    
    def predict(self, features: np.ndarray) -> float:
        """
        预测价值
        
        Args:
            features: 状态特征向量
        
        Returns:
            预测的价值
        """
        return np.dot(self.weights, features)
    
    def update(self, features: np.ndarray, target: float):
        """
        更新权重
        
        TD 更新：w ← w + α * (target - V(s)) * φ(s)
        
        Args:
            features: 状态特征
            target: 目标价值
        """
        prediction = self.predict(features)
        error = target - prediction
        self.weights += self.lr * error * features
    
    def td_update(
        self,
        features: np.ndarray,
        reward: float,
        next_features: np.ndarray,
        terminated: bool
    ):
        """
        TD 更新
        
        Args:
            features: 当前状态特征
            reward: 奖励
            next_features: 下一状态特征
            terminated: 是否终止
        """
        # 计算 TD 目标
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * self.predict(next_features)
        
        # 更新
        self.update(features, target)
    
    def get_weights(self) -> np.ndarray:
        """获取权重"""
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray):
        """设置权重"""
        self.weights = weights.copy()


class LinearQFunction:
    """
    线性 Q 函数
    
    Q(s, a, w) = w_a^T * φ(s)
    每个动作有独立的权重向量
    """
    
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99
    ):
        """
        Args:
            n_features: 特征数量
            n_actions: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        
        # 每个动作一个权重向量
        self.weights = np.zeros((n_actions, n_features))
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测所有动作的 Q 值
        
        Args:
            features: 状态特征
        
        Returns:
            Q 值数组
        """
        return np.dot(self.weights, features)
    
    def predict_action(self, features: np.ndarray, action: int) -> float:
        """
        预测特定动作的 Q 值
        
        Args:
            features: 状态特征
            action: 动作索引
        
        Returns:
            Q 值
        """
        return np.dot(self.weights[action], features)
    
    def get_best_action(self, features: np.ndarray) -> int:
        """
        获取最优动作
        
        Args:
            features: 状态特征
        
        Returns:
            最优动作索引
        """
        q_values = self.predict(features)
        return int(np.argmax(q_values))
    
    def update(
        self,
        features: np.ndarray,
        action: int,
        target: float
    ):
        """
        更新权重
        
        Args:
            features: 状态特征
            action: 动作
            target: 目标 Q 值
        """
        prediction = self.predict_action(features, action)
        error = target - prediction
        self.weights[action] += self.lr * error * features
    
    def q_learning_update(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        terminated: bool
    ):
        """
        Q-Learning 更新
        
        Args:
            features: 当前状态特征
            action: 动作
            reward: 奖励
            next_features: 下一状态特征
            terminated: 是否终止
        """
        if terminated:
            target = reward
        else:
            # 使用最大 Q 值
            next_q_max = np.max(self.predict(next_features))
            target = reward + self.gamma * next_q_max
        
        self.update(features, action, target)
    
    def sarsa_update(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        next_action: int,
        terminated: bool
    ):
        """
        SARSA 更新（在策略）
        
        Args:
            features: 当前状态特征
            action: 动作
            reward: 奖励
            next_features: 下一状态特征
            next_action: 下一动作
            terminated: 是否终止
        """
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * self.predict_action(next_features, next_action)
        
        self.update(features, action, target)


def tile_coding(
    state: np.ndarray,
    n_tilings: int = 8,
    tile_size: float = 1.0,
    offsets: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    平铺编码 (Tile Coding)
    
    将连续状态转换为稀疏二值特征
    
    Args:
        state: 连续状态向量
        n_tilings: 平铺数量
        tile_size: 每个 tile 的大小
        offsets: 每个平铺的偏移
    
    Returns:
        二值特征向量
    """
    state = np.asarray(state)
    n_dims = len(state)
    
    if offsets is None:
        # 生成偏移
        offsets = np.zeros((n_tilings, n_dims))
        for i in range(n_tilings):
            offsets[i] = np.random.uniform(0, tile_size, n_dims)
    
    # 计算每个维度的 tile 索引
    indices = np.floor((state + offsets) / tile_size).astype(int)
    
    # 创建稀疏特征
    n_features = n_tilings
    features = np.zeros(n_features)
    
    for i in range(n_tilings):
        # 使用 hash 或简单编码
        features[i] = 1.0
    
    return features


class TileCodedValueFunction:
    """
    使用平铺编码的线性价值函数
    """
    
    def __init__(
        self,
        state_dim: int,
        n_tilings: int = 8,
        tile_size: float = 1.0,
        learning_rate: float = 0.1,
        gamma: float = 0.99
    ):
        """
        Args:
            state_dim: 状态维度
            n_tilings: 平铺数量
            tile_size: tile 大小
            learning_rate: 学习率
            gamma: 折扣因子
        """
        self.state_dim = state_dim
        self.n_tilings = n_tilings
        self.tile_size = tile_size
        self.lr = learning_rate
        self.gamma = gamma
        
        # 权重（每个 tile 一个权重）
        self.weights = {}
        
        # 偏移
        self.offsets = np.zeros((n_tilings, state_dim))
        for i in range(n_tilings):
            self.offsets[i] = np.random.uniform(0, tile_size, state_dim)
    
    def _get_active_tiles(self, state: np.ndarray) -> list:
        """获取激活的 tile 索引"""
        state = np.asarray(state)
        indices = np.floor((state + self.offsets) / self.tile_size).astype(int)
        
        # 创建唯一索引
        tiles = []
        for i in range(self.n_tilings):
            # 使用元组作为字典键
            tile_key = tuple(indices[i]) + (i,)
            tiles.append(tile_key)
        
        return tiles
    
    def predict(self, state: np.ndarray) -> float:
        """预测价值"""
        tiles = self._get_active_tiles(state)
        value = sum(self.weights.get(t, 0.0) for t in tiles)
        return value
    
    def update(self, state: np.ndarray, target: float):
        """更新权重"""
        tiles = self._get_active_tiles(state)
        prediction = self.predict(state)
        error = target - prediction
        
        # 平均更新
        update_amount = self.lr * error / len(tiles)
        
        for tile in tiles:
            self.weights[tile] = self.weights.get(tile, 0.0) + update_amount
    
    def td_update(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool
    ):
        """TD 更新"""
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * self.predict(next_state)
        
        self.update(state, target)
