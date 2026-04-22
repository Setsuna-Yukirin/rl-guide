# 第 6 章：策略梯度与连续控制

> **当动作是连续的时候** —— 直接优化策略函数，从价值方法到策略方法

---

## ⏱️ TL;DR（30 秒速览）

**核心问题**：待填充

**核心思想**：待填充

**关键算法**：待填充

**学完这章你能**：待填充

**常见误区**：待填充

---

---

## 🎯 本章要解决什么问题

在前面的章节中，我们学习的所有算法（Q-Learning、SARSA、DQN）都有一个共同特点：**学习价值函数，然后从价值函数导出策略**。

这种方法叫做**价值方法**（Value-based Methods）。

- Q-Learning：学习 Q 表，策略是 $\pi(s) = \arg\max_a Q(s, a)$
- DQN：学习神经网络 Q 函数，策略同样是 $\arg\max$

价值方法在**离散动作空间**上非常成功：
- Atari 游戏（4-18 个离散动作）
- 棋盘游戏（有限种落子方式）
- 网格世界（上/下/左/右）

但是，当动作空间是**连续的**，价值方法就失效了。

### 连续控制问题

考虑这些场景：

**场景一：自动驾驶**

- 方向盘角度：连续值 [-30°, 30°]
- 油门：连续值 [0%, 100%]
- 刹车：连续值 [0%, 100%]

**问题**：
- 动作空间是连续的，有无限多种可能
- 无法计算 $\arg\max_a Q(s, a)$（需要搜索整个连续空间）
- 离散化会丢失精度（如方向盘只能转 10 个角度）

**场景二：机器人控制**

一个 7 自由度机械臂：
- 每个关节的力矩：连续值
- 总动作空间：$\mathbb{R}^7$（7 维连续空间）

**问题**：
- 高维连续空间
- 需要精确控制
- 动作之间有复杂的耦合关系

**场景三：无人机飞行**

- 四个旋翼的转速：连续值
- 动作空间：$\mathbb{R}^4$
- 需要实时调整

### 策略方法的核心洞察

策略方法（Policy-based Methods）的核心洞察是：

> **与其学习价值函数再导出策略，不如直接学习策略函数。**

用数学表示：

**价值方法**：
$$Q(s, a; \mathbf{w}) \rightarrow \pi(s) = \arg\max_a Q(s, a; \mathbf{w})$$

**策略方法**：
$$\pi(a|s; \theta) \quad \text{（直接参数化策略）}$$

**关键优势**：
1. **天然处理连续动作**：策略可以直接输出连续值
2. **可以学习随机策略**：输出动作的概率分布
3. **更好的收敛性质**：直接优化目标函数

### 从 REINFORCE 到 TD3

策略方法的发展经历了几个关键阶段：

**阶段一：REINFORCE（1992）**
- 蒙特卡洛策略梯度
- 无偏但高方差
- 需要完整 episode

**阶段二：Actor-Critic（2000s）**
- 用 Critic 降低方差
- 可以单步更新
- A2C 等算法

**阶段三：确定性策略梯度（2014-2016）**
- DDPG：结合 DQN 技术和策略梯度
- 处理高维连续控制
- 经验回放 + 目标网络

**阶段四：改进的 Actor-Critic（2018-）**
- TD3：解决 DDPG 的过估计问题
- SAC：最大熵策略梯度
- 更稳定、更高效

学完本章后，你将能够：
- 理解策略梯度定理的数学原理
- 掌握 REINFORCE 算法及其方差问题
- 理解 Actor-Critic 如何降低方差
- 掌握 A2C、DDPG、TD3 等算法
- 能够用 PyTorch 实现连续控制算法
- 为后续学习 PPO、SAC（第 7 章）打下基础

---

## 📖 场景描述：从倒立摆到机械臂

### 场景一：倒立摆（CartPole）的连续版本

想象第 5 章的 CartPole，但这次：
- 动作不是"左推/右推"两个离散选项
- 而是施加的力：连续值 [-10N, 10N]

**状态**：
- 小车位置、速度
- 杆子角度、角速度

**动作**：
- 施加的力 $F \in [-10, 10]$ 牛顿

**挑战**：
- 如何表示策略？$\pi(a|s)$ 是一个分布
- 如何更新策略？梯度是什么？
- 如何平衡探索和利用？

**策略表示**：
用高斯分布表示随机策略：
$$\pi(a|s) = \mathcal{N}(a; \mu(s), \sigma^2(s))$$

其中：
- $\mu(s)$：均值网络（输出最佳动作）
- $\sigma(s)$：标准差（控制探索）

**学习过程**：
1. 从 $\mathcal{N}(\mu(s), \sigma^2(s))$ 采样动作
2. 执行动作，获得回报
3. 如果回报好 → 增加这个动作的概率
4. 如果回报差 → 减少这个动作的概率

### 场景二：机械臂抓取

考虑一个平面 2 关节机械臂：

```
     关节 2
      O
     /
    / 连杆 2
   O
关节 1
```

**状态**：
- 关节 1 角度 $\theta_1$
- 关节 2 角度 $\theta_2$
- 关节角速度 $\dot{\theta}_1, \dot{\theta}_2$
- 目标物体位置 $(x_{target}, y_{target})$

**动作**：
- 关节 1 力矩 $\tau_1 \in [-5, 5]$ Nm
- 关节 2 力矩 $\tau_2 \in [-5, 5]$ Nm

**奖励**：
- 末端执行器接近目标：+1 每步
- 抓住目标：+100
- 能耗惩罚：-0.01 × (τ₁² + τ₂²)

**挑战**：
- 2 维连续动作空间
- 动作之间有耦合（关节 1 的运动会影响关节 2）
- 需要精细控制

**解决方案**：
用 DDPG 或 TD3 学习确定性策略：
$$\mathbf{a} = \mu(\mathbf{s}; \theta)$$

输出直接是动作值，不需要采样。

### 场景三：四足机器人行走

波士顿动力的机器狗：

**状态**：
- 12 个关节的角度和速度
- 身体姿态（IMU 数据）
- 地面接触传感器

**动作**：
- 12 个关节的目标位置（连续值）

**奖励**：
- 向前行走速度
- 保持平衡
- 能耗最小化

**挑战**：
- 12 维连续动作空间
- 高维状态空间
- 复杂的动力学

**解决方案**：
用 SAC（Soft Actor-Critic，第 7 章）或 TD3：
- 最大熵鼓励探索
- 双 Q 网络减少过估计
- 经验回放提高样本效率

---

## 🧠 核心概念详解

### 概念一：为什么需要策略方法

**直觉理解**：

价值方法在连续动作空间失效的原因有两个：

**问题一：argmax 不可行**

价值方法的策略是：
$$\pi(s) = \arg\max_a Q(s, a)$$

对于离散动作，这是简单的查找。

对于连续动作，这需要**优化**：
$$\pi(s) = \arg\max_{a \in \mathbb{R}^d} Q(s, a)$$

每次选择动作都要解一个优化问题！这太慢了。

**问题二：离散化丢失信息**

如果 discretize 动作空间：
- 方向盘：[-30°, -20°, -10°, 0°, 10°, 20°, 30°]
- 只能转这 7 个角度

问题：
- 精细控制不可能（如转 15°）
- 动作数量指数增长（3 个动作各 7 档 → 343 种组合）

**策略方法的优势**：

策略 $\pi(a|s; \theta)$ 直接参数化：
- 对于连续动作：输出高斯分布的参数
- 对于离散动作：输出概率分布

选择动作：
- 连续：从 $\mathcal{N}(\mu, \sigma^2)$ 采样
- 离散：从 Categorical 分布采样

**不需要 argmax！**

### 概念二：策略梯度定理

**直觉理解**：

我们想要最大化期望回报：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中 $\tau = (s_0, a_0, s_1, a_1, ...)$ 是轨迹。

策略梯度定理告诉我们：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right]$$

**直观解释**：
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$：动作的对数概率梯度
- $Q^{\pi_\theta}(s_t, a_t)$：这个动作有多好

如果 Q 值为正（好动作）：
- 梯度方向增加这个动作的概率

如果 Q 值为负（差动作）：
- 梯度方向减少这个动作的概率

**为什么这样设计**：

策略梯度定理的关键是**似然比技巧**（Likelihood Ratio Trick）：

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

这个技巧让我们可以：
- 用采样近似期望
- 不需要知道环境模型
- 无偏估计梯度

### 概念三：REINFORCE 算法

**直觉理解**：

REINFORCE 是最简单的策略梯度算法。

核心思想：
- 用蒙特卡洛回报 $G_t$ 近似 $Q(s_t, a_t)$
- 更新策略，增加高回报动作的概率

**更新规则**：
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

**问题：高方差**

蒙特卡洛回报 $G_t$ 的方差很大：
- 同一状态 - 动作对，不同 episode 的回报可能差异很大
- 导致梯度估计不稳定
- 需要很多样本才能收敛

**例子**：

考虑 CartPole：
- 状态 s：杆子角度 5°
- 动作 a：向右推 5N
- Episode 1：存活 100 步，$G_t = 100$
- Episode 2：存活 50 步，$G_t = 50$
- Episode 3：存活 200 步，$G_t = 200$

方差：$\text{Var}(G_t) \approx 3500$

这么大的方差导致：
- 梯度方向不稳定
- 学习率需要很小
- 收敛慢

### 概念四：Actor-Critic 架构

**直觉理解**：

Actor-Critic 结合了策略梯度和价值方法的优点：

- **Actor**（演员）：策略网络 $\pi(a|s; \theta)$
  - 负责选择动作
  - 用策略梯度更新

- **Critic**（评论家）：价值网络 $V(s; w)$ 或 $Q(s, a; w)$
  - 负责评估状态/动作
  - 用 TD 学习更新

**关键洞察**：

用 Critic 的估计代替蒙特卡洛回报：
$$\nabla_\theta J(\theta) \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)]$$

其中优势函数：
$$A(s, a) = Q(s, a) - V(s)$$

**为什么这样设计**：

**降低方差**：
- 蒙特卡洛回报：方差大（因为随机性累积）
- Critic 估计：方差小（因为单步 TD）

**引入偏差**：
- Critic 的估计是有偏的（因为自举）
- 但偏差通常可以接受

**方差 - 偏差权衡**：
- REINFORCE：无偏，高方差
- Actor-Critic：有偏，低方差
- 实践中 Actor-Critic 更常用

### 概念五：确定性策略梯度（DDPG）

**直觉理解**：

之前的策略梯度都是**随机策略**：
$$\pi(a|s) = \text{概率分布}$$

DDPG 提出用**确定性策略**：
$$a = \mu(s; \theta)$$

输出直接是动作值，不是分布。

**为什么这样设计**：

**优势**：
1. **计算高效**：不需要采样，直接输出动作
2. **适合连续控制**：精确的动作值
3. **可以利用 Off-Policy 数据**：用经验回放

**挑战**：
- 如何探索？（没有随机性）
- 如何保证收敛？

**解决方案**：
1. **探索**：在输出动作上加噪声
   $$a = \mu(s) + \mathcal{N}(0, \sigma^2)$$
2. **稳定性**：用 DQN 的技术
   - 经验回放
   - 目标网络
   - 软更新

**DDPG 的更新规则**：

Critic（Q 网络）：
$$\mathcal{L} = \mathbb{E}[(r + \gamma Q(s', \mu(s'); \theta^-) - Q(s, a; w))^2]$$

Actor（策略网络）：
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_a Q(s, a; w)|_{a=\mu(s)} \cdot \nabla_\theta \mu(s; \theta)]$$

**直观解释**：
- Critic 学习 Q 函数（类似 DQN）
- Actor 学习最大化 Q 值的动作

### 概念六：TD3（Twin Delayed DDPG）

**直觉理解**：

DDPG 有一个严重问题：**过估计**。

原因：
- Q 网络倾向于高估动作价值
- Actor 利用这个高估计
- 导致性能下降

TD3 提出三个改进：

**改进一：双 Q 网络（Clipped Double Q-Learning）**
- 学习两个 Q 网络：$Q_1, Q_2$
- 目标值用较小的那个：
  $$y = r + \gamma \min(Q_1(s', \mu(s')), Q_2(s', \mu(s')))$$
- 减少过估计

**改进二：延迟策略更新（Delayed Policy Updates）**
- Critic 更新多次后才更新 Actor
- 让 Q 函数更准确后再更新策略
- 典型：Critic 更新 2 次，Actor 更新 1 次

**改进三：目标策略平滑（Target Policy Smoothing）**
- 在目标动作上加噪声：
  $$a' = \mu(s') + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
- 防止过拟合到当前策略
- 鼓励探索

**为什么这样设计**：

这三个改进都针对 DDPG 的弱点：
1. 双 Q 网络 → 减少过估计
2. 延迟更新 → 更准确的梯度
3. 策略平滑 → 鲁棒性更好

**结果**：
- TD3 在大多数连续控制任务上优于 DDPG
- 训练更稳定
- 最终性能更好

---

（第一部分完成，待续...）
## 📐 公式推导

### 推导一：策略梯度定理

**问题设定**：

我们想要最大化期望回报：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中 $\tau = (s_0, a_0, ..., s_T)$ 是轨迹，$R(\tau) = \sum_{t=0}^T \gamma^t r_t$ 是回报。

**轨迹概率**：

$$P(\tau|\theta) = P(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)$$

**梯度计算**：

$$\begin{aligned}
\nabla_\theta J(\theta) 
&= \nabla_\theta \sum_\tau P(\tau|\theta) R(\tau) \\
&= \sum_\tau \nabla_\theta P(\tau|\theta) R(\tau) \\
&= \sum_\tau P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) \\
&= \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau|\theta) R(\tau)]
\end{aligned}$$

**关键步骤：似然比技巧**

$$\begin{aligned}
\nabla_\theta \log P(\tau|\theta) 
&= \nabla_\theta \left[\log P(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t|s_t) + \sum_{t=0}^T \log P(s_{t+1}|s_t, a_t)\right] \\
&= \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)
\end{aligned}$$

因为 $P(s_0)$ 和 $P(s_{t+1}|s_t, a_t)$ 不依赖于 $\theta$。

**最终形式**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)\right]$$

**变体：用 Q 函数代替回报**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) Q^{\pi_\theta}(s_t, a_t)\right]$$

### 推导二：优势函数的方差降低

**问题设定**：

我们想要证明，用优势函数 $A(s, a) = Q(s, a) - V(s)$ 代替 $Q(s, a)$ 可以降低方差。

**基线技巧**：

对于任意只依赖于状态的函数 $b(s)$：

$$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s) \cdot b(s)] = 0$$

**证明**：

$$\begin{aligned}
\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a|s) \cdot b(s)] 
&= b(s) \sum_a \nabla_\theta \log \pi(a|s) \pi(a|s) \\
&= b(s) \nabla_\theta \sum_a \pi(a|s) \\
&= b(s) \nabla_\theta 1 \\
&= 0
\end{aligned}$$

**推论**：

$$\mathbb{E}[\nabla_\theta \log \pi(a|s) \cdot (Q(s, a) - V(s))] = \mathbb{E}[\nabla_\theta \log \pi(a|s) \cdot Q(s, a)]$$

所以用 $A(s, a)$ 代替 $Q(s, a)$ 是**无偏**的。

**方差降低**：

直观上，$V(s)$ 是一个基线，减掉它相当于"中心化"。

中心化后的梯度方差更小，因为：
- 去除了状态本身的方差
- 只保留动作选择的方差

### 推导三：DDPG 的确定性策略梯度

**问题设定**：

对于确定性策略 $\mu(s; \theta)$，如何计算梯度？

**链式法则**：

$$\nabla_\theta J(\theta) = \mathbb{E}_s[\nabla_\theta \mu(s; \theta) \cdot \nabla_a Q(s, a; w)|_{a=\mu(s)}]$$

**证明**：

$$\begin{aligned}
J(\theta) &= \mathbb{E}_s[Q(s, \mu(s; \theta); w)] \\
\nabla_\theta J(\theta) &= \mathbb{E}_s[\nabla_\theta Q(s, \mu(s; \theta); w)] \\
&= \mathbb{E}_s[\nabla_a Q(s, a; w)|_{a=\mu(s)} \cdot \nabla_\theta \mu(s; \theta)]
\end{aligned}$$

**直观解释**：

- $\nabla_a Q$：Q 函数对动作的梯度（动作往哪个方向 Q 值增加）
- $\nabla_\theta \mu$：策略对参数的梯度（如何调整参数改变动作）

链式法则：调整参数 → 改变动作 → 增加 Q 值

### 推导四：TD3 的双 Q 学习

**问题设定**：

证明双 Q 学习可以减少过估计。

**单 Q 网络的过估计**：

$$\mathbb{E}[\max_a Q(s, a)] \geq \max_a \mathbb{E}[Q(s, a)]$$

（Jensen 不等式，max 是凸函数）

**双 Q 网络**：

$$y = r + \gamma \min(Q_1(s', \mu(s')), Q_2(s', \mu(s')))$$

**为什么减少过估计**：

假设 $Q_1$ 和 $Q_2$ 独立，且都有正偏差：

$$\mathbb{E}[Q_1] = Q^* + \epsilon_1, \quad \mathbb{E}[Q_2] = Q^* + \epsilon_2$$

其中 $\epsilon_1, \epsilon_2 > 0$。

则：
$$\mathbb{E}[\min(Q_1, Q_2)] \leq \min(\mathbb{E}[Q_1], \mathbb{E}[Q_2]) = Q^* + \min(\epsilon_1, \epsilon_2)$$

过估计减少了！

---

## 💻 算法实现

### 实现一：REINFORCE 算法

```python
"""
REINFORCE 算法

蒙特卡洛策略梯度
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Dict


class PolicyNetwork(nn.Module):
    """
    策略网络
    
    输入：状态
    输出：动作概率分布
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """输出动作 logits"""
        return self.network(state)
    
    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        从策略采样动作
        
        Args:
            state: 当前状态
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self(state_tensor)
        
        # 创建分类分布
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob


class REINFORCE:
    """
    REINFORCE 算法
    
    更新规则：
        θ ← θ + α * ∇log π(a|s) * G
    """
    
    def __init__(self, state_dim: int, n_actions: int, 
                 lr: float = 1e-3, gamma: float = 0.99):
        """
        初始化 REINFORCE
        
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            lr: 学习率
            gamma: 折扣因子
        """
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """选择动作"""
        return self.policy.get_action(state)
    
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        计算折扣回报
        
        G_t = R_t + γ*R_{t+1} + γ²*R_{t+2} + ...
        
        Args:
            rewards: 奖励序列
            
        Returns:
            returns: 折扣回报序列
        """
        returns = []
        G = 0.0
        
        # 反向计算回报
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # 标准化（降低方差）
        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, states: List[np.ndarray], 
               actions: List[int], 
               log_probs: List[torch.Tensor],
               rewards: List[float]) -> Dict[str, float]:
        """
        执行 REINFORCE 更新
        
        Args:
            states: 状态序列
            actions: 动作序列
            log_probs: 对数概率序列
            rewards: 奖励序列
            
        Returns:
            stats: 训练统计
        """
        # 计算回报
        returns = self.compute_returns(rewards)
        
        # 计算策略梯度损失
        # L = -Σ log π(a_t|s_t) * G_t
        policy_loss = 0.0
        for log_prob, G in zip(log_probs, returns):
            policy_loss -= log_prob * G
        
        # 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'mean_return': np.mean(rewards),
        }
    
    def train(self, env, n_episodes: int = 1000, 
              verbose: bool = True) -> List[float]:
        """
        训练 REINFORCE
        
        Args:
            env: Gymnasium 环境
            n_episodes: 训练 episode 数
            verbose: 是否打印进度
            
        Returns:
            episode_rewards: 每个 episode 的总奖励
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            # 生成一个 episode
            states, actions, log_probs, rewards = [], [], [], []
            state, _ = env.reset()
            
            while True:
                # 选择动作
                action, log_prob = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                # 记录
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state
                
                if done:
                    break
            
            # 更新策略
            stats = self.update(states, actions, log_probs, rewards)
            
            episode_reward = sum(rewards)
            episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Loss: {stats['policy_loss']:.4f}")
        
        return episode_rewards
```

### 实现二：Actor-Critic 算法

```python
"""
Actor-Critic 算法

用 Critic 降低策略梯度的方差
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络
    
    共享特征提取层，两个输出头
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor 头：输出动作概率
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Critic 头：输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_logits: 动作 logits
            value: 状态价值
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Returns:
            action: 动作
            log_prob: 对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_logits, value = self(state_tensor)
        
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value


class ActorCritic:
    """
    Actor-Critic 智能体
    
    更新规则：
        Critic: w ← w + α * (G - V(s)) * ∇V(s)
        Actor: θ ← θ + α * ∇log π(a|s) * (G - V(s))
    """
    
    def __init__(self, state_dim: int, n_actions: int,
                 lr: float = 1e-3, gamma: float = 0.99):
        """
        初始化 Actor-Critic
        
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            lr: 学习率
            gamma: 折扣因子
        """
        self.gamma = gamma
        self.network = ActorCriticNetwork(state_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """选择动作"""
        return self.network.get_action(state)
    
    def compute_returns(self, rewards: List[float], 
                       values: List[torch.Tensor]) -> torch.Tensor:
        """
        计算 TD 回报
        
        G_t = R_t + γ*V(s_{t+1})
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            
        Returns:
            returns: TD 回报序列
        """
        returns = []
        G = 0.0
        
        # 反向计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1].item()
            
            G = rewards[t] + self.gamma * next_value
            returns.insert(0, G)
        
        return torch.FloatTensor(returns)
    
    def update(self, states: List[np.ndarray],
               actions: List[int],
               log_probs: List[torch.Tensor],
               values: List[torch.Tensor],
               rewards: List[float]) -> Dict[str, float]:
        """
        执行 Actor-Critic 更新
        
        Returns:
            stats: 训练统计
        """
        # 计算 TD 回报
        returns = self.compute_returns(rewards, values)
        
        # 计算优势函数
        # A(s, a) = G - V(s)
        values_tensor = torch.cat(values)
        advantages = returns - values_tensor.detach()
        
        # Actor 损失：-Σ log π(a|s) * A(s, a)
        actor_loss = 0.0
        for log_prob, adv in zip(log_probs, advantages):
            actor_loss -= log_prob * adv
        actor_loss /= len(log_probs)
        
        # Critic 损失：Σ (G - V(s))²
        critic_loss = F.mse_loss(values_tensor, returns)
        
        # 总损失
        loss = actor_loss + critic_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_advantage': advantages.mean().item(),
        }
    
    def train(self, env, n_episodes: int = 1000,
              verbose: bool = True) -> List[float]:
        """训练 Actor-Critic"""
        episode_rewards = []
        
        for episode in range(n_episodes):
            states, actions, log_probs, values, rewards = [], [], [], [], []
            state, _ = env.reset()
            
            while True:
                action, log_prob, value = self.select_action(state)
                
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                
                state = next_state
                
                if done:
                    break
            
            stats = self.update(states, actions, log_probs, values, rewards)
            
            episode_reward = sum(rewards)
            episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Actor Loss: {stats['actor_loss']:.4f} | "
                      f"Critic Loss: {stats['critic_loss']:.4f}")
        
        return episode_rewards
```

### 实现三：DDPG 算法

```python
"""
DDPG (Deep Deterministic Policy Gradient)

确定性策略梯度 + 经验回放 + 目标网络
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, Tuple


class Actor(nn.Module):
    """
    Actor 网络
    
    输入：状态
    输出：动作（连续值）
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 action_bounds: Tuple[float, float],
                 hidden_dim: int = 256):
        super().__init__()
        self.action_bounds = action_bounds
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出 [-1, 1]
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """输出动作"""
        action = self.network(state)
        
        # 缩放到动作范围
        low, high = self.action_bounds
        action = low + (action + 1) * 0.5 * (high - low)
        
        return action


class Critic(nn.Module):
    """
    Critic 网络
    
    输入：状态 + 动作
    输出：Q 值
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 状态和动作分别处理
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 合并后输出 Q 值
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """计算 Q 值"""
        state_features = self.state_net(state)
        action_features = self.action_net(action)
        combined = torch.cat([state_features, action_features], dim=1)
        return self.q_net(combined)


class DDPGAgent:
    """
    DDPG 智能体
    
    关键技术：
    - 经验回放
    - 目标网络（Actor 和 Critic）
    - 软更新（Polyak 平均）
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0),
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = int(1e6),
                 batch_size: int = 64,
                 noise_scale: float = 0.1):
        """
        初始化 DDPG
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            action_bounds: 动作范围
            lr_actor: Actor 学习率
            lr_critic: Critic 学习率
            gamma: 折扣因子
            tau: 软更新系数
            buffer_size: 回放缓冲区大小
            batch_size: 批次大小
            noise_scale: 探索噪声标准差
        """
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.action_bounds = action_bounds
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor 网络
        self.actor = Actor(state_dim, action_dim, action_bounds).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, action_bounds).to(self.device)
        
        # Critic 网络
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        
        # 初始化目标网络
        self._soft_update(1.0)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 回放缓冲区
        self.buffer = deque(maxlen=buffer_size)
    
    def _soft_update(self, tau: float):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 当前状态
            explore: 是否添加探索噪声
            
        Returns:
            action: 动作
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore:
            # 添加高斯噪声
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action + noise
            
            # 裁剪到动作范围
            low, high = self.action_bounds
            action = np.clip(action, low, high)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """存储转移"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def update(self) -> Dict[str, float]:
        """执行一次 DDPG 更新"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # 采样批次
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ==================== 更新 Critic ====================
        
        with torch.no_grad():
            # 目标策略
            next_actions = self.target_actor(next_states)
            # 目标 Q 值
            next_q = self.target_critic(next_states, next_actions)
            # TD 目标
            targets = rewards + self.gamma * next_q * (1 - dones)
        
        # 当前 Q 值
        current_q = self.critic(states, actions)
        
        # Critic 损失
        critic_loss = F.mse_loss(current_q, targets)
        
        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==================== 更新 Actor ====================
        
        # Actor 动作
        actor_actions = self.actor(states)
        # Q 值
        q_values = self.critic(states, actor_actions)
        # Actor 损失（最大化 Q 值）
        actor_loss = -q_values.mean()
        
        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ==================== 软更新 ====================
        
        self._soft_update(self.tau)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_q': current_q.mean().item(),
        }
    
    def train(self, env, n_episodes: int = 1000,
              verbose: bool = True) -> list:
        """训练 DDPG"""
        episode_rewards = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                # 选择动作（带探索）
                action = self.select_action(state, explore=True)
                
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                # 存储转移
                self.store_transition(state, action, reward, next_state, done)
                
                # 更新
                stats = self.update()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.2f}")
        
        return episode_rewards
```

（第二部分完成，待续...）
## 🔬 算法对比

### 策略梯度 vs 价值方法

| 维度 | 价值方法 (Q-Learning, DQN) | 策略方法 (PG, Actor-Critic) |
|------|---------------------------|----------------------------|
| **优化对象** | 价值函数 Q(s,a) | 策略函数 π(a\|s) |
| **动作选择** | argmax Q(s,a) | 从 π(a\|s) 采样 |
| **连续动作** | 困难（需要离散化或优化） | 天然支持 |
| **随机策略** | 不能直接学习 | 可以直接学习 |
| **收敛性** | 可能不收敛（函数近似时） | 收敛到局部最优 |
| **样本效率** | 较高（Off-Policy） | 较低（On-Policy） |
| **方差** | 低 | 高（REINFORCE）/ 中（AC） |

### 策略梯度算法对比

| 算法 | 策略类型 | On/Off-Policy | 方差 | 适用场景 |
|------|----------|---------------|------|----------|
| **REINFORCE** | 随机 | On | 高 | 简单任务、离散动作 |
| **Actor-Critic** | 随机 | On | 中 | 通用 |
| **A2C** | 随机 | On | 中 | 并行训练 |
| **DDPG** | 确定性 | Off | 低 | 连续控制 |
| **TD3** | 确定性 | Off | 低 | 连续控制（SOTA） |
| **SAC** | 随机 | Off | 低 | 连续控制（最大熵） |

### DDPG vs TD3

| 维度 | DDPG | TD3 |
|------|------|-----|
| **Q 网络数量** | 1 个 | 2 个（取最小值） |
| **过估计** | 有 | 减少 |
| **更新频率** | Actor/Critic 同步 | Critic 更新 2 次，Actor 更新 1 次 |
| **目标动作** | 直接 μ(s') | μ(s') + 噪声 |
| **稳定性** | 一般 | 更好 |
| **最终性能** | 好 | 更好 |

---

## 🧪 动手实验

### 实验一：REINFORCE vs Actor-Critic 收敛对比

**任务描述**：

在 CartPole 上对比 REINFORCE 和 Actor-Critic 的收敛速度。

**实验设置**：

- 环境：CartPole-v1
- 网络架构：相同（2 层全连接，64 隐藏单元）
- 学习率：REINFORCE 1e-3，Actor-Critic 1e-3
- 训练：1000 集

**分析指标**：
- 学习曲线（每 50 集平均奖励）
- 收敛速度（达到 450 平均奖励的集数）
- 方差（多次运行的标准差）

**预期结果**：
- Actor-Critic 收敛更快（方差低）
- REINFORCE 波动更大
- 最终性能相近

**思考题**：
1. 为什么 Actor-Critic 方差更低？
2. 如果增加 REINFORCE 的学习率，会发生什么？
3. 在什么情况下 REINFORCE 可能更好？

### 实验二：DDPG 超参数敏感性分析

**任务描述**：

研究 DDPG 关键超参数对训练的影响。

**参数设置**：

| 参数 | 组 A | 组 B | 组 C |
|------|------|------|------|
| 噪声标准差 | 0.01 | 0.1 | 1.0 |
| 软更新 τ | 0.001 | 0.005 | 0.01 |
| 批次大小 | 32 | 64 | 256 |

**实验步骤**：

1. 在 Pendulum-v1 环境上训练
2. 每组参数运行 5 次独立实验
3. 绘制学习曲线

**预期现象**：
- 噪声太小：探索不足，陷入局部最优
- 噪声太大：训练不稳定
- τ 太小：目标网络更新慢，学习慢
- τ 太大：目标网络不稳定

### 实验三：TD3 vs DDPG 性能对比

**任务描述**：

实现 TD3 并与 DDPG 对比。

**TD3 实现要点**：

```python
# 1. 双 Q 网络
class TD3Critic(nn.Module):
    def __init__(self, ...):
        self.q1 = Critic(...)
        self.q2 = Critic(...)
    
    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

# 2. 延迟 Actor 更新
if self.step_count % 2 == 0:  # 每 2 步更新一次 Actor
    self._update_actor()

# 3. 目标策略平滑
noise = torch.randn_like(next_action) * 0.2
noise = noise.clamp(-0.5, 0.5)
smoothed_action = (next_action + noise).clamp(low, high)
```

**分析指标**：
- Q 值估计的偏差
- 训练稳定性
- 最终性能（Pendulum 的平均奖励）

**预期结果**：
- TD3 的 Q 值估计更准确
- TD3 训练更稳定
- TD3 最终性能更好

### 实验四：连续控制基准测试

**任务描述**：

在多个连续控制环境上 benchmark 不同算法。

**环境**：
- Pendulum-v1（简单）
- HalfCheetah-v4（中等）
- Ant-v4（困难）

**算法**：
- DDPG
- TD3
- SAC（第 7 章）

**分析指标**：
- 样本效率（达到目标奖励的步数）
- 最终性能
- 训练稳定性

---

## ❓ 常见问题

### Q1: 为什么策略梯度需要基线（baseline）？

**A**: 基线用于降低方差。

**数学分析**：

策略梯度：
$$\nabla J(\theta) = \mathbb{E}[\nabla \log \pi(a|s) \cdot Q(s, a)]$$

用基线 $b(s)$：
$$\nabla J(\theta) = \mathbb{E}[\nabla \log \pi(a|s) \cdot (Q(s, a) - b(s))]$$

因为：
$$\mathbb{E}[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \nabla \sum_a \pi(a|s) = b(s) \nabla 1 = 0$$

所以基线不改变期望（无偏），但降低方差。

**最佳基线**：

理论上最优基线是：
$$b^*(s) = \frac{\mathbb{E}[\nabla \log \pi \cdot Q^2]}{2 \mathbb{E}[\nabla \log \pi \cdot Q]}$$

实践中用 $V(s)$ 近似。

### Q2: 确定性策略和随机策略各有什么优劣？

**A**: 取决于任务特点。

**确定性策略优势**：
- 计算高效（不需要采样）
- 适合精确控制
- 可以利用 Off-Policy 数据

**确定性策略劣势**：
- 需要额外探索（加噪声）
- 可能陷入局部最优
- 不能表示最优随机策略

**随机策略优势**：
- 天然探索
- 可以表示最优随机策略（如石头剪刀布）
- 在某些任务中更稳定

**随机策略劣势**：
- 需要采样
- 方差更高
- 训练可能更慢

**选择建议**：
- 连续控制：确定性（DDPG/TD3）
- 对抗任务：随机（PPO/SAC）
- 需要探索：随机

### Q3: 为什么 TD3 要延迟 Actor 更新？

**A**: 让 Critic 更准确后再更新 Actor。

**问题分析**：

在 DDPG 中：
- Critic 和 Actor 同步更新
- 如果 Critic 不准确，Actor 会利用这个不准确
- 导致性能下降

**TD3 的解决方案**：

- Critic 更新 2 次，Actor 更新 1 次
- 让 Critic 有更准确的目标值
- Actor 基于更准确的梯度更新

**类比**：

就像学生（Actor）和老师（Critic）：
- 如果老师自己都没搞懂，教学生会出错
- 老师先多学习几次，再教学生
- 学生学得更好

### Q4: 如何选择探索噪声？

**A**: 取决于任务和算法。

**高斯噪声**（DDPG 使用）：
$$a = \mu(s) + \mathcal{N}(0, \sigma^2)$$

- 简单，容易实现
- 适合连续控制
- 问题：时间不相关

**Ornstein-Uhlenbeck 噪声**：
$$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$$

- 时间相关的噪声
- 适合物理系统（如机器人）
- 问题：超参数多

**参数空间噪声**：
- 在策略参数上加噪声
- 而不是在动作空间
- 更有效的探索

**选择建议**：
- 简单任务：高斯噪声
- 物理系统：OU 噪声
- 高维任务：参数空间噪声

### Q5: 策略梯度方法收敛到全局最优吗？

**A**: 不一定，通常收敛到局部最优。

**理论保证**：

对于一般的非线性策略：
- 策略梯度收敛到**稳定点**（梯度为 0）
- 可能是局部最优，不一定是全局最优

**特殊情况**：

对于表格型策略（离散状态 - 动作）：
- 如果学习率适当衰减
- 收敛到全局最优

**实践建议**：

1. 多次运行，取最好结果
2. 用更好的初始化
3. 用熵正则化鼓励探索
4. 用 Trust Region 方法（如 TRPO、PPO）

---

## 📚 延伸阅读

### 核心论文

1. **Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.**
   - REINFORCE 原始论文
   - https://link.springer.com/article/10.1007/BF00992696

2. **Lillicrap et al. (2016). Continuous control with deep reinforcement learning.**
   - DDPG 论文
   - https://arxiv.org/abs/1509.02971

3. **Fujimoto et al. (2018). Addressing function approximation error in actor-critic methods.**
   - TD3 论文
   - https://arxiv.org/abs/1802.09477

### 教程

1. **Spinning Up in Deep RL - Policy Gradient**
   - https://spinningup.openai.com
   - 包含 PG、DDPG、TD3 的实现

2. **Deep RL Bootcamp**
   - https://sites.google.com/view/deep-rl-bootcamp
   - Berkeley 的课程

### 代码资源

| 文件 | 内容 | 行数 |
|------|------|------|
| `reinforce.py` | REINFORCE 算法 | ~150 |
| `actor_critic.py` | Actor-Critic | ~180 |
| `a2c.py` | A2C 算法 | ~150 |
| `ddpg.py` | DDPG 算法 | ~250 |
| `td3.py` | TD3 算法 | ~280 |

---

## ✅ 本章检查清单

学完本章后，你应该能够：

**概念理解**：
- [ ] 解释为什么需要策略方法
- [ ] 区分随机和确定性策略
- [ ] 解释 Actor-Critic 的方差降低
- [ ] 解释 TD3 的三个改进

**数学推导**：
- [ ] 推导策略梯度定理
- [ ] 推导基线技巧的无偏性
- [ ] 推导确定性策略梯度
- [ ] 推导双 Q 学习的去偏

**代码实现**：
- [ ] 实现 REINFORCE 算法
- [ ] 实现 Actor-Critic 算法
- [ ] 实现 DDPG 智能体
- [ ] 实现 TD3 智能体

**实验分析**：
- [ ] 对比 REINFORCE 和 Actor-Critic
- [ ] 分析 DDPG 超参数影响
- [ ] 实现 TD3 并对比 DDPG
- [ ] 在连续控制环境上 benchmark

**应用判断**：
- [ ] 根据任务选择算法
- [ ] 合理设置超参数
- [ ] 诊断训练问题

---

## 📝 课后测验（10 分钟）

### 基础题（必答）

**1. 策略方法 vs 价值方法的核心区别是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

| 维度 | 价值方法 | 策略方法 |
|------|----------|----------|
| 优化对象 | Q(s,a) | π(a\|s) |
| 动作选择 | argmax Q | 从 π 采样 |
| 连续动作 | 困难 | 天然支持 |
| 随机策略 | 不能 | 可以 |
</details>

---

**2. REINFORCE 为什么方差高？**

<details>
<summary>点击查看答案</summary>

**答案**：

REINFORCE 用蒙特卡洛回报 G_t 近似 Q(s,a)。

G_t 是完整轨迹的累积奖励，方差大因为：
- 同一状态 - 动作对，不同 episode 回报差异大
- 随机性累积

解决：用 Actor-Critic，用 Critic 估计代替 G_t。
</details>

---

**3. DDPG 的四个关键技术是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

1. **确定性策略**：a = μ(s)
2. **经验回放**：打破相关性
3. **目标网络**：稳定训练
4. **软更新**：τ 很小（0.005）
</details>

---

### 进阶题（选答）

**4. TD3 的三个改进分别解决什么问题？**

<details>
<summary>点击查看答案</summary>

**答案**：

1. **双 Q 网络**：减少过估计
2. **延迟 Actor 更新**：让 Critic 更准确
3. **目标策略平滑**：防止过拟合
</details>

---

**5. 为什么确定性策略需要额外探索？**

<details>
<summary>点击查看答案</summary>

**答案**：

确定性策略 a = μ(s) 没有内在随机性。

需要加噪声：
- 高斯噪声：a = μ(s) + N(0,σ²)
- OU 噪声：时间相关
- 参数空间噪声：在θ上加噪声
</details>

---

### 编程题（实践）

**6. 实现 TD3 并在 Pendulum 上测试**

<details>
<summary>查看提示</summary>

**提示**：
1. 双 Critic 网络
2. Critic 更新 2 次，Actor 更新 1 次
3. 目标动作加噪声
4. 对比 DDPG 的性能
</details>

---

## 🚀 下一章预告

**第 7 章：高级策略优化**

本章将介绍更先进的策略优化算法：
- PPO（Proximal Policy Optimization）
- SAC（Soft Actor-Critic）
- DPO（Direct Preference Optimization）
- GRPO（Group Relative Policy Optimization）

**预告实验**：用 PPO 训练 LLM 对齐！🤖

---

*最后更新：2026-04-22*  
*作者：Hermes <neko_yukirin@qq.com>*
