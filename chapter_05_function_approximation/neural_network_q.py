"""
神经网络 Q 函数

使用 PyTorch 实现神经网络来近似 Q 函数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Union


class QNetwork(nn.Module):
    """
    基础 Q 网络
    
    输入：状态
    输出：每个动作的 Q 值
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: nn.Module = nn.ReLU
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            hidden_dims: 隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # 构建网络
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 (batch_size, state_dim)
        
        Returns:
            Q 值 (batch_size, n_actions)
        """
        return self.network(state)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        选择动作
        
        Args:
            state: 状态
            epsilon: 探索率
        
        Returns:
            动作索引
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return int(torch.argmax(q_values).item())


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN 网络
    
    分离状态价值 V(s) 和优势函数 A(s,a)
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (64, 64)
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        
        self.n_actions = n_actions
        
        # 共享特征提取层
        feature_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*feature_layers)
        
        # 价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1)
        )
        
        # 优势流 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.features(state)
        
        # 计算价值和优势
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)
        
        # 组合：Q = V + A - mean(A)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class CNNQNetwork(nn.Module):
    """
    卷积 Q 网络
    
    用于处理图像输入（如 Atari 游戏）
    """
    
    def __init__(self, n_actions: int, input_shape: Tuple[int, int, int] = (4, 84, 84)):
        """
        Args:
            n_actions: 动作数量
            input_shape: 输入形状 (channels, height, width)
        """
        super().__init__()
        
        self.n_actions = n_actions
        self.input_shape = input_shape
        
        # 卷积层（参考 DQN 论文）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积输出大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self.conv_layers(dummy_input)
            self.flat_dim = conv_output.view(1, -1).size(1)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv_layers(state)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class NeuralQFunction:
    """
    神经网络 Q 函数管理器
    
    封装训练和推理逻辑
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: Optional[str] = None,
        use_dueling: bool = False
    ):
        """
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            device: 计算设备
            use_dueling: 是否使用 Dueling 架构
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 创建网络
        if use_dueling:
            self.q_network = DuelingQNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        else:
            self.q_network = QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        
        # 目标网络
        self.target_network = type(self.q_network)(state_dim, n_actions, *hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """选择动作"""
        return self.q_network.get_action(state, epsilon)
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> float:
        """
        单步训练
        
        Args:
            states: 状态 (batch, state_dim)
            actions: 动作 (batch,)
            rewards: 奖励 (batch,)
            next_states: 下一状态 (batch, state_dim)
            dones: 是否终止 (batch,)
        
        Returns:
            损失值
        """
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q 值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
