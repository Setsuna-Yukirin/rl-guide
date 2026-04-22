"""
价值函数 (Value Function)

价值函数评估状态或状态 - 动作对的好坏程度。

- 状态价值函数 V(s): 从状态 s 开始的期望回报
- 动作价值函数 Q(s, a): 从状态 s 采取动作 a 的期望回报

Reference:
    Sutton, R. S., & Barto, A. G. (2018). 
    Reinforcement learning: An introduction. MIT press.
"""

import numpy as np
from typing import Optional, Dict, Union, Tuple
from abc import ABC, abstractmethod


class ValueFunction(ABC):
    """
    价值函数抽象基类
    
    Attributes:
        state_dim: 状态空间维度
    """
    
    def __init__(self, state_dim: int):
        """
        初始化价值函数
        
        Args:
            state_dim: 状态空间维度
        """
        self.state_dim = state_dim
    
    @abstractmethod
    def get_value(self, state: np.ndarray) -> float:
        """
        获取状态价值 V(s)
        
        Args:
            state: 状态
            
        Returns:
            价值值
        """
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, target: float, lr: float = 0.1):
        """
        更新价值函数
        
        Args:
            state: 状态
            target: 目标价值
            lr: 学习率
        """
        pass


class TabularVFunction(ValueFunction):
    """
    表格型状态价值函数
    
    使用数组存储每个状态的价值。
    
    Attributes:
        values: 状态价值数组
    
    Example:
        >>> from utils.core import TabularVFunction
        >>> V = TabularVFunction(state_dim=10)
        >>> V.update(state=0, target=5.0, lr=0.1)
        >>> print(V.get_value(0))
    """
    
    def __init__(self, state_dim: int, init_value: float = 0.0):
        """
        初始化表格型 V 函数
        
        Args:
            state_dim: 状态空间维度
            init_value: 初始价值值
        """
        super().__init__(state_dim)
        self.values = np.full(state_dim, init_value, dtype=np.float32)
    
    def get_value(self, state: np.ndarray) -> float:
        """
        获取状态价值
        
        Args:
            state: 状态（整数索引）
            
        Returns:
            状态价值
        """
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        return float(self.values[state_idx])
    
    def update(self, state: np.ndarray, target: float, lr: float = 0.1):
        """
        使用 TD 误差更新状态价值
        
        V(s) ← V(s) + α * [target - V(s)]
        
        Args:
            state: 状态
            target: 目标价值（如 R + γV(s')）
            lr: 学习率
        """
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        td_error = target - self.values[state_idx]
        self.values[state_idx] += lr * td_error
    
    def set_values(self, values: np.ndarray):
        """
        直接设置价值数组
        
        Args:
            values: 价值数组，形状为 (state_dim,)
        """
        if values.shape != (self.state_dim,):
            raise ValueError(
                f"价值数组形状 {values.shape} 不匹配，"
                f"期望形状 ({self.state_dim},)"
            )
        self.values = values.copy()
    
    def get_values(self) -> np.ndarray:
        """
        获取所有状态价值
        
        Returns:
            价值数组
        """
        return self.values.copy()
    
    def reset(self, value: float = 0.0):
        """
        重置所有价值
        
        Args:
            value: 重置后的价值值
        """
        self.values.fill(value)


class TabularQFunction:
    """
    表格型动作价值函数 (Q 函数)
    
    使用数组存储每个状态 - 动作对的价值。
    
    Attributes:
        q_values: Q 值数组，形状为 (state_dim, action_dim)
    
    Example:
        >>> from utils.core import TabularQFunction
        >>> Q = TabularQFunction(state_dim=10, action_dim=4)
        >>> Q.update(state=0, action=1, target=5.0, lr=0.1)
        >>> print(Q.get_value(0, 1))
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        init_value: float = 0.0,
    ):
        """
        初始化表格型 Q 函数
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            init_value: 初始 Q 值
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_values = np.full(
            (state_dim, action_dim),
            init_value,
            dtype=np.float32,
        )
    
    def get_value(self, state: np.ndarray, action: int) -> float:
        """
        获取 Q 值 Q(s, a)
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            Q 值
        """
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        return float(self.q_values[state_idx, action])
    
    def get_values(self, state: np.ndarray) -> np.ndarray:
        """
        获取某状态的所有动作价值
        
        Args:
            state: 状态
            
        Returns:
            Q 值数组，形状为 (action_dim,)
        """
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        return self.q_values[state_idx].copy()
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        target: float,
        lr: float = 0.1,
    ):
        """
        使用 TD 误差更新 Q 值
        
        Q(s, a) ← Q(s, a) + α * [target - Q(s, a)]
        
        Args:
            state: 状态
            action: 动作
            target: 目标 Q 值（如 R + γmax_a'Q(s', a')）
            lr: 学习率
        """
        state_idx = int(state) if np.isscalar(state) else int(state[0])
        td_error = target - self.q_values[state_idx, action]
        self.q_values[state_idx, action] += lr * td_error
    
    def get_best_action(self, state: np.ndarray) -> int:
        """
        获取最优动作
        
        Args:
            state: 状态
            
        Returns:
            最优动作索引
        """
        q_values = self.get_values(state)
        return int(np.argmax(q_values))
    
    def set_q_values(self, q_values: np.ndarray):
        """
        直接设置 Q 值数组
        
        Args:
            q_values: Q 值数组，形状为 (state_dim, action_dim)
        """
        if q_values.shape != (self.state_dim, self.action_dim):
            raise ValueError(
                f"Q 值数组形状 {q_values.shape} 不匹配，"
                f"期望形状 ({self.state_dim}, {self.action_dim})"
            )
        self.q_values = q_values.copy()
    
    def get_q_values(self) -> np.ndarray:
        """
        获取所有 Q 值
        
        Returns:
            Q 值数组
        """
        return self.q_values.copy()
    
    def reset(self, value: float = 0.0):
        """
        重置所有 Q 值
        
        Args:
            value: 重置后的 Q 值
        """
        self.q_values.fill(value)


class NeuralVFunction:
    """
    神经网络价值函数
    
    使用 PyTorch 神经网络近似价值函数。
    
    Example:
        >>> from utils.core import NeuralVFunction
        >>> V = NeuralVFunction(state_dim=4, hidden_dims=[64, 64])
        >>> loss = V.update(states, targets)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        device: str = "cpu",
    ):
        """
        初始化神经网络 V 函数
        
        Args:
            state_dim: 状态维度
            hidden_dims: 隐藏层维度元组
            activation: 激活函数 ("relu", "tanh", "sigmoid")
            device: 计算设备 ("cpu", "cuda")
        """
        import torch
        import torch.nn as nn
        
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.torch = torch
        self.nn = nn
        
        # 构建网络
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
    
    def get_value(self, state: np.ndarray) -> float:
        """
        获取状态价值
        
        Args:
            state: 状态
            
        Returns:
            价值值
        """
        self.network.eval()
        with self.torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            value = self.network(state_tensor).item()
        return value
    
    def get_values(self, states: np.ndarray) -> np.ndarray:
        """
        批量获取状态价值
        
        Args:
            states: 状态数组，形状为 (batch_size, state_dim)
            
        Returns:
            价值数组
        """
        self.network.eval()
        with self.torch.no_grad():
            states_tensor = self._to_tensor(states)
            values = self.network(states_tensor).cpu().numpy().flatten()
        return values
    
    def update(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        lr: Optional[float] = None,
    ) -> float:
        """
        更新网络参数
        
        Args:
            states: 状态数组，形状为 (batch_size, state_dim)
            targets: 目标价值数组
            lr: 学习率（可选，如果提供则更新优化器）
            
        Returns:
            平均损失
        """
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.network.train()
        
        states_tensor = self._to_tensor(states)
        targets_tensor = self._to_tensor(targets).unsqueeze(1)
        
        # 前向传播
        predictions = self.network(states_tensor)
        loss = self.nn.MSELoss()(predictions, targets_tensor)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())
    
    def _to_tensor(self, x: np.ndarray) -> "torch.Tensor":
        """转换为 PyTorch 张量"""
        return self.torch.FloatTensor(x).to(self.device)
    
    def save(self, path: str):
        """保存模型"""
        self.torch.save(self.network.state_dict(), path)
    
    def load(self, path: str):
        """加载模型"""
        self.network.load_state_dict(
            self.torch.load(path, map_location=self.device)
        )


if __name__ == "__main__":
    # 简单测试
    print("测试 ValueFunction 类...\n")
    
    # 测试表格型 V 函数
    print("1. 测试 TabularVFunction")
    V = TabularVFunction(state_dim=10)
    V.update(state=0, target=5.0, lr=0.1)
    value = V.get_value(0)
    print(f"   状态 0 的价值：{value:.4f}")
    assert abs(value - 0.5) < 0.01, "V(s) 应该接近 0.5"
    print(f"   ✓ TabularVFunction 测试通过\n")
    
    # 测试表格型 Q 函数
    print("2. 测试 TabularQFunction")
    Q = TabularQFunction(state_dim=10, action_dim=4)
    Q.update(state=0, action=1, target=5.0, lr=0.1)
    q_value = Q.get_value(0, 1)
    print(f"   Q(0, 1) = {q_value:.4f}")
    assert abs(q_value - 0.5) < 0.01, "Q(s,a) 应该接近 0.5"
    
    best_action = Q.get_best_action(0)
    print(f"   状态 0 的最优动作：{best_action}")
    print(f"   ✓ TabularQFunction 测试通过\n")
    
    # 测试神经网络 V 函数
    print("3. 测试 NeuralVFunction")
    V_net = NeuralVFunction(state_dim=4, hidden_dims=[32, 32], device="cpu")
    
    # 训练几步
    states = np.random.randn(32, 4)
    targets = np.random.randn(32)
    
    for i in range(10):
        loss = V_net.update(states, targets)
    
    print(f"   训练后损失：{loss:.4f}")
    print(f"   ✓ NeuralVFunction 测试通过\n")
    
    print("✅ 所有 ValueFunction 测试通过！")
