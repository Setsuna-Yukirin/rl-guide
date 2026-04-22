"""
Offline RL（离线强化学习）算法实现

Offline RL 从固定数据集中学习策略，无需与环境交互。
适用于安全关键应用、大规模预训练等场景。

核心挑战：
- 分布外 (OOD) 动作：策略可能选择数据集中未见的动作
- 外推误差：Q 函数对 OOD 动作的估计不准确

本章实现：
1. Behavior Cloning (BC) - 行为克隆
2. Conservative Q-Learning (CQL) - 保守 Q 学习
3. Implicit Q-Learning (IQL) - 隐式 Q 学习

参考文献：
- BC: Pomerleau (1988) "Alvinn: An autonomous land vehicle"
- CQL: Kumar et al. "Conservative Q-Learning for Offline RL" (2020)
- IQL: Kostrikov et al. "Offline RL with Implicit Q-Learning" (2022)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class OfflineRLConfig:
    """Offline RL 配置"""
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = int(1e6)
    hidden_dim: int = 256
    
    # CQL 特定
    cql_alpha: float = 1.0
    cql_n_actions: int = 10
    cql_importance_sample: bool = True
    cql_lagrange: bool = False
    cql_target_action_gap: float = 5.0
    
    # IQL 特定
    iql_tau: float = 0.7  # 期望回归分位数
    iql_beta: float = 3.0  # 优势加权温度
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class OfflineDataset:
    """
    离线数据集
    
    存储 (state, action, reward, next_state, done) 元组
    支持随机采样和迭代
    """
    
    def __init__(self, capacity: int = int(1e6)):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """随机采样批次"""
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    @classmethod
    def from_d4rl(cls, dataset_name: str) -> 'OfflineDataset':
        """
        从 D4RL 数据集加载
        
        需要安装：pip install d4rl
        """
        try:
            import d4rl
            import gymnasium as gym
            
            env = gym.make(dataset_name)
            dataset = env.get_dataset()
            
            offline_dataset = cls()
            
            n = dataset['rewards'].shape[0]
            for i in range(n - 1):
                offline_dataset.add(
                    state=dataset['observations'][i],
                    action=dataset['actions'][i],
                    reward=dataset['rewards'][i],
                    next_state=dataset['observations'][i + 1],
                    done=dataset['terminals'][i] or dataset['timeouts'][i],
                )
            
            return offline_dataset
        except ImportError:
            raise ImportError("D4RL not installed. Run: pip install d4rl")
    
    def save(self, path: str):
        """保存数据集"""
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    @classmethod
    def load(cls, path: str) -> 'OfflineDataset':
        """加载数据集"""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        dataset = cls()
        for item in data:
            dataset.buffer.append(item)
        
        return dataset


class BehaviorCloningAgent:
    """
    Behavior Cloning (BC) - 行为克隆
    
    最简单的 Offline RL 方法：监督学习模仿数据集中的动作
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        config: Optional[OfflineRLConfig] = None,
    ):
        self.config = config or OfflineRLConfig()
        self.action_bounds = action_bounds
        self.action_dim = action_dim
        
        # 策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, action_dim),
            nn.Tanh(),  # 输出 [-1, 1]
        ).to(self.config.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.lr)
        
        # 训练统计
        self.total_updates = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            action = self.policy_network(state_tensor)
            
            # 缩放到动作范围
            low, high = self.action_bounds
            action = low + (action + 1) * 0.5 * (high - low)
        
        return action.cpu().numpy()[0]
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            states: 状态批次 [B, S]
            actions: 动作批次 [B, A]
        
        Returns:
            stats: 训练统计
        """
        self.policy_network.train()
        
        # 预测动作
        predicted_actions = self.policy_network(states)
        
        # BC 损失：MSE
        loss = F.mse_loss(predicted_actions, actions)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        
        return {
            'bc_loss': loss.item(),
        }
    
    def train(
        self,
        dataset: OfflineDataset,
        n_epochs: int = 10,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        训练 BC 智能体
        
        Args:
            dataset: 离线数据集
            n_epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
        
        Returns:
            history: 训练历史
        """
        batch_size = batch_size or self.config.batch_size
        history = []
        n_batches = max(1, len(dataset) // batch_size)
        
        for epoch in range(n_epochs):
            epoch_stats = []
            
            for _ in range(n_batches):
                batch = dataset.sample(batch_size)
                
                states = torch.FloatTensor(batch['states']).to(self.config.device)
                actions = torch.FloatTensor(batch['actions']).to(self.config.device)
                
                stats = self.train_step(states, actions)
                epoch_stats.append(stats)
            
            avg_stats = {
                key: np.mean([s[key] for s in epoch_stats])
                for key in epoch_stats[0]
            }
            history.append(avg_stats)
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} | BC Loss: {avg_stats['bc_loss']:.4f}")
        
        return history
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'total_updates': self.total_updates,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.total_updates = checkpoint['total_updates']


class CQLAgent:
    """
    Conservative Q-Learning (CQL)
    
    通过惩罚 OOD 动作的 Q 值，使 Q 函数保守，避免外推误差。
    
    CQL 损失：
    L_CQL = L_TD + α * (E[Q(s, a~π)] - E[Q(s, a~data)])
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        config: Optional[OfflineRLConfig] = None,
    ):
        self.config = config or OfflineRLConfig()
        self.action_bounds = action_bounds
        self.action_dim = action_dim
        
        # Q 网络（两个，防止过估计）
        self.q_network1 = self._create_q_network(state_dim, action_dim).to(self.config.device)
        self.q_network2 = self._create_q_network(state_dim, action_dim).to(self.config.device)
        self.target_q_network1 = self._create_q_network(state_dim, action_dim).to(self.config.device)
        self.target_q_network2 = self._create_q_network(state_dim, action_dim).to(self.config.device)
        
        # 复制权重到目标网络
        self._soft_update(1.0)
        
        # 策略网络
        self.policy_network = self._create_policy_network(state_dim, action_dim).to(self.config.device)
        
        # 优化器
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=self.config.lr
        )
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.lr)
        
        # CQL Lagrange 乘子（可选）
        if self.config.cql_lagrange:
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.config.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.config.cql_alpha), device=self.config.device)
        
        # 训练统计
        self.total_updates = 0
    
    def _create_q_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """创建 Q 网络"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
    
    def _create_policy_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """创建策略网络"""
        return nn.Sequential(
            nn.Linear(state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, action_dim),
            nn.Tanh(),
        )
    
    def _soft_update(self, tau: float):
        """软更新目标网络"""
        for target_param, param in zip(self.target_q_network1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
        for target_param, param in zip(self.target_q_network2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            action = self.policy_network(state_tensor)
            low, high = self.action_bounds
            action = low + (action + 1) * 0.5 * (high - low)
        
        return action.cpu().numpy()[0]
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        单步训练
        
        Returns:
            stats: 训练统计
        """
        # 转换为 tensor
        states = torch.FloatTensor(batch['states']).to(self.config.device)
        actions = torch.FloatTensor(batch['actions']).to(self.config.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.config.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.config.device)
        
        # ==================== 更新 Q 网络 ====================
        
        # TD 目标
        with torch.no_grad():
            next_actions = self.policy_network(next_states)
            q1_target = self.target_q_network1(torch.cat([next_states, next_actions], dim=-1))
            q2_target = self.target_q_network2(torch.cat([next_states, next_actions], dim=-1))
            min_q_target = torch.min(q1_target, q2_target)
            q_target = rewards + self.config.gamma * (1 - dones) * min_q_target
        
        # 当前 Q 值
        q1_current = self.q_network1(torch.cat([states, actions], dim=-1))
        q2_current = self.q_network2(torch.cat([states, actions], dim=-1))
        
        # TD 损失
        td_loss1 = F.mse_loss(q1_current, q_target)
        td_loss2 = F.mse_loss(q2_current, q_target)
        td_loss = td_loss1 + td_loss2
        
        # ==================== CQL 正则化 ====================
        
        alpha = self.log_alpha.exp()
        
        # 采样动作计算 CQL 项
        if self.config.cql_importance_sample:
            # 随机采样动作
            low, high = self.action_bounds
            random_actions = torch.FloatTensor(
                np.random.uniform(low, high, (states.size(0) * self.config.cql_n_actions, self.action_dim))
            ).to(self.config.device)
            state_repeat = states.repeat_interleave(self.config.cql_n_actions, dim=0)
            
            q1_random = self.q_network1(torch.cat([state_repeat, random_actions], dim=-1))
            q2_random = self.q_network2(torch.cat([state_repeat, random_actions], dim=-1))
            
            q1_data = self.q_network1(torch.cat([states, actions], dim=-1))
            q2_data = self.q_network2(torch.cat([states, actions], dim=-1))
            
            cql_loss1 = torch.logsumexp(q1_random / self.config.cql_alpha, dim=0).mean() * self.config.cql_alpha - q1_data.mean()
            cql_loss2 = torch.logsumexp(q2_random / self.config.cql_alpha, dim=0).mean() * self.config.cql_alpha - q2_data.mean()
        else:
            # 使用策略采样
            policy_actions = self.policy_network(states)
            q1_policy = self.q_network1(torch.cat([states, policy_actions], dim=-1))
            q2_policy = self.q_network2(torch.cat([states, policy_actions], dim=-1))
            
            cql_loss1 = q1_policy.mean() - q1_current.mean()
            cql_loss2 = q2_policy.mean() - q2_current.mean()
        
        cql_loss = cql_loss1 + cql_loss2
        
        # CQL Lagrange（可选）
        if self.config.cql_lagrange:
            cql_loss = alpha * (cql_loss - self.config.cql_target_action_gap)
        else:
            cql_loss = alpha * cql_loss
        
        # 总 Q 损失
        q_loss = td_loss + cql_loss
        
        # 优化 Q 网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ==================== 更新策略网络 ====================
        
        policy_actions = self.policy_network(states)
        q_policy = self.q_network1(torch.cat([states, policy_actions], dim=-1))
        policy_loss = -q_policy.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ==================== 更新 Lagrange 乘子 ====================
        
        if self.config.cql_lagrange:
            alpha_loss = -self.log_alpha * (cql_loss.detach() - self.config.cql_target_action_gap)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ==================== 软更新 ====================
        
        self._soft_update(self.config.tau)
        
        self.total_updates += 1
        
        return {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': alpha.item(),
            'mean_q': q1_current.mean().item(),
        }
    
    def train(
        self,
        dataset: OfflineDataset,
        n_epochs: int = 10,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """训练 CQL 智能体"""
        batch_size = batch_size or self.config.batch_size
        history = []
        n_batches = max(1, len(dataset) // batch_size)
        
        for epoch in range(n_epochs):
            epoch_stats = []
            
            for _ in range(n_batches):
                batch = dataset.sample(batch_size)
                stats = self.train_step(batch)
                epoch_stats.append(stats)
            
            avg_stats = {
                key: np.mean([s[key] for s in epoch_stats])
                for key in epoch_stats[0]
            }
            history.append(avg_stats)
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} | "
                      f"TD Loss: {avg_stats['td_loss']:.4f} | "
                      f"CQL Loss: {avg_stats['cql_loss']:.4f} | "
                      f"Mean Q: {avg_stats['mean_q']:.4f}")
        
        return history
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q_network1': self.target_q_network1.state_dict(),
            'target_q_network2': self.target_q_network2.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_updates': self.total_updates,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.q_network1.load_state_dict(checkpoint['q_network1'])
        self.q_network2.load_state_dict(checkpoint['q_network2'])
        self.target_q_network1.load_state_dict(checkpoint['target_q_network1'])
        self.target_q_network2.load_state_dict(checkpoint['target_q_network2'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha'].to(self.config.device)
        self.total_updates = checkpoint['total_updates']


class IQLAgent:
    """
    Implicit Q-Learning (IQL)
    
    使用期望回归学习值函数，避免 OOD 动作问题。
    
    核心思想：
    1. 使用期望回归学习 V(s)：只关注高 Q 值动作
    2. 优势加权行为克隆学习策略
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        config: Optional[OfflineRLConfig] = None,
    ):
        self.config = config or OfflineRLConfig()
        self.action_bounds = action_bounds
        self.action_dim = action_dim
        
        # Q 网络
        self.q_network = self._create_q_network(state_dim, action_dim).to(self.config.device)
        
        # V 网络
        self.v_network = self._create_v_network(state_dim).to(self.config.device)
        
        # 策略网络
        self.policy_network = self._create_policy_network(state_dim, action_dim).to(self.config.device)
        
        # 优化器
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.lr)
        self.v_optimizer = torch.optim.Adam(self.v_network.parameters(), lr=self.config.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.lr)
        
        self.total_updates = 0
    
    def _create_q_network(self, state_dim: int, action_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
    
    def _create_v_network(self, state_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
    
    def _create_policy_network(self, state_dim: int, action_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, action_dim),
            nn.Tanh(),
        )
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            action = self.policy_network(state_tensor)
            low, high = self.action_bounds
            action = low + (action + 1) * 0.5 * (high - low)
        
        return action.cpu().numpy()[0]
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """单步训练"""
        states = torch.FloatTensor(batch['states']).to(self.config.device)
        actions = torch.FloatTensor(batch['actions']).to(self.config.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.config.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.config.device)
        
        # ==================== 更新 V 网络（期望回归） ====================
        
        with torch.no_grad():
            q_target = self.q_network(torch.cat([states, actions], dim=-1))
        
        v_pred = self.v_network(states)
        
        # 期望回归损失：只关注高 Q 值
        exp_weights = torch.exp((q_target - v_pred) * self.config.iql_tau)
        v_loss = (exp_weights * (q_target - v_pred).pow(2)).mean()
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        # ==================== 更新 Q 网络 ====================
        
        with torch.no_grad():
            v_next = self.v_network(next_states)
            q_target = rewards + self.config.gamma * (1 - dones) * v_next
        
        q_pred = self.q_network(torch.cat([states, actions], dim=-1))
        q_loss = F.mse_loss(q_pred, q_target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ==================== 更新策略网络（优势加权 BC） ====================
        
        with torch.no_grad():
            q_current = self.q_network(torch.cat([states, actions], dim=-1))
            v_current = self.v_network(states)
            advantages = (q_current - v_current).exp()
        
        policy_actions = self.policy_network(states)
        
        # BC 损失，优势加权
        policy_loss = F.mse_loss(policy_actions, actions)
        policy_loss = (advantages * policy_loss).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.total_updates += 1
        
        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'mean_advantage': advantages.mean().item(),
        }
    
    def train(
        self,
        dataset: OfflineDataset,
        n_epochs: int = 10,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """训练 IQL 智能体"""
        batch_size = batch_size or self.config.batch_size
        history = []
        n_batches = max(1, len(dataset) // batch_size)
        
        for epoch in range(n_epochs):
            epoch_stats = []
            
            for _ in range(n_batches):
                batch = dataset.sample(batch_size)
                stats = self.train_step(batch)
                epoch_stats.append(stats)
            
            avg_stats = {
                key: np.mean([s[key] for s in epoch_stats])
                for key in epoch_stats[0]
            }
            history.append(avg_stats)
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} | "
                      f"V Loss: {avg_stats['v_loss']:.4f} | "
                      f"Q Loss: {avg_stats['q_loss']:.4f}")
        
        return history
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'v_network': self.v_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'total_updates': self.total_updates,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.v_network.load_state_dict(checkpoint['v_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.total_updates = checkpoint['total_updates']
