# 第 5 章：函数近似与深度 Q 网络

> **当状态空间太大时** —— 用神经网络近似价值函数，从表格 RL 迈向深度 RL

---

## ⏱️ TL;DR（30 秒速览）

**核心问题**：状态空间太大时，表格方法失效，怎么办？

**核心思想**：
> 用函数近似代替表格，从状态预测价值。

**两种方法**：
- **线性近似**：V(s) = wᵀφ(s)，简单可解释
- **神经网络**：DQN，自动学习特征

**DQN 关键技术**：
- 经验回放：打破数据相关性
- 目标网络：固定目标值，稳定训练

**变体**：
- Double DQN：减少过估计
- Dueling DQN：分离 V 和 A

**关键洞察**：
- 泛化：相似状态有相似价值
- 连续空间：直接处理
- 样本效率：高（共享知识）

**学完这章你能**：
- ✅ 理解为什么需要函数近似
- ✅ 实现 DQN 算法
- ✅ 理解经验回放和目标网络
- ✅ 训练 Atari 游戏 AI

**常见误区**：
- ❌ 函数近似一定更好 → ✅ 小状态空间表格法足够
- ❌ DQN 不需要目标网络 → ✅ 会发散
- ❌ 经验回放只是提高样本效率 → ✅ 关键是打破相关性

---

---

## 🎯 本章要解决什么问题

在前面的章节中，我们学习的所有算法（动态规划、蒙特卡洛、时序差分）都有一个共同特点：**用表格存储价值函数**。

- Q-Learning：用一个 Q 表，Q[s, a] 存储每个状态 - 动作对的价值
- SARSA：同样用 Q 表
- 策略迭代：用 V 表存储状态价值

这种方法叫做**表格型方法**（Tabular Methods）。它在状态空间较小时非常有效。

### 表格型方法的局限

但是，当状态空间很大时，表格型方法就失效了。考虑这些场景：

**场景一：Atari 游戏**

Atari 游戏的输入是 210×160×3 的 RGB 图像。即使我们把图像下采样到 84×84 灰度图，可能的状态数也是：

$$256^{84 \times 84} \approx 10^{147733}$$

这个数字比宇宙中的原子数（$10^{80}$）还要大得多！

**问题**：
- 无法用表格存储所有状态
- 即使能存储，也永远无法访问所有状态
- 无法泛化到未见过的状态

**场景二：机器人控制**

一个机器人的状态可能包括：
- 关节角度（连续值）
- 关节速度（连续值）
- 摄像头图像（高维）
- 力传感器读数（连续值）

**问题**：
- 状态是连续的，有无限多种可能
- 表格无法表示连续空间

**场景三：围棋**

围棋的状态空间约为 $10^{170}$。虽然理论上可以用表格存储，但：
- 需要海量内存
- 需要访问每个状态无数次才能准确估计价值
- 无法利用状态之间的相似性（如对称性）

### 函数近似的核心洞察

函数近似（Function Approximation）的核心洞察是：

> **我们不需要存储每个状态的价值，只需要学习一个函数，能从状态预测价值。**

用数学表示：

**表格型**：
$$V(s) = \text{表格中的第 s 个元素}$$

**函数近似**：
$$V(s) \approx \hat{V}(s, \mathbf{w})$$

其中 $\hat{V}$ 是一个参数化函数（如神经网络），$\mathbf{w}$ 是参数。

**关键优势**：
1. **泛化能力**：相似的状态会有相似的输出
2. **内存效率**：只需要存储参数，不需要存储所有状态
3. **连续空间**：可以直接处理连续状态

### 从表格 RL 到深度 RL

函数近似不是新概念，但直到 2013-2015 年 DeepMind 的 DQN 论文发表，大家才发现：

> **用深度神经网络 + RL，可以在 Atari 游戏上达到人类水平！**

这就是**深度强化学习**的诞生。

DQN 的成功基于三个关键创新：
1. **卷积神经网络**：从像素中提取特征
2. **经验回放**：打破数据相关性，提高稳定性
3. **目标网络**：固定目标值，避免发散

学完本章后，你将能够：
- 理解为什么需要函数近似
- 掌握线性函数近似的原理
- 理解神经网络如何近似 Q 函数
- 掌握 DQN 的核心技术和数学原理
- 理解 Double DQN、Dueling DQN 等变体
- 能够用 PyTorch 实现 DQN 并训练 Atari 游戏
- 为后续学习策略梯度（第 6 章）和 PPO/DPO（第 7 章）打下基础

---

## 📖 场景描述：从 CartPole 到 Atari

### 场景一：倒立摆平衡（CartPole）

想象一个小车在轨道上，车上竖立着一根杆子：

```
    |
    |  ← 杆子
  =====  ← 小车
━━━━━━━━━━━━━━  ← 轨道
```

**目标**：通过左右推动小车，保持杆子不倒。

**状态空间**：
- 小车位置：连续值 [-2.4, 2.4]
- 小车速度：连续值
- 杆子角度：连续值 [-24°, 24°]
- 杆子角速度：连续值

**动作空间**：
- 向左推
- 向右推

**奖励**：
- 每存活一步 +1 分
- 杆子倒了或小车出轨 → episode 结束

**挑战**：
- 状态是连续的，有无限多种可能
- 表格型 Q-Learning 无法直接应用（无法 discretize 所有状态）
- 需要用函数近似

**解决方案**：
用神经网络近似 Q 函数：
$$Q(\text{[位置，速度，角度，角速度]}, \text{动作}; \theta) \rightarrow \text{Q 值}$$

### 场景二：Atari Breakout（打砖块）

这是 DeepMind DQN 论文的经典实验：

```
╔════════════════════════╗
║ 🧱🧱🧱🧱🧱🧱🧱🧱🧱🧱 ║  ← 砖块
║ 🧱🧱🧱🧱🧱🧱🧱🧱🧱🧱 ║
║                        ║
║                        ║
║        🏐             ║  ← 球
║                        ║
║   [====]              ║  ← 挡板
╚════════════════════════╝
```

**状态**：游戏画面（210×160×3 RGB 图像）
**动作**：左移、右移、发射
**奖励**：
- 打破砖块：+1 到 +3 分
- 球掉落：失去一条命
- 游戏结束：episode 终止

**挑战**：
- 状态空间巨大（$10^{147733}$ 种可能画面）
- 需要从像素中提取有用特征（球的位置、挡板位置、砖块布局）
- 需要理解游戏物理（球的运动轨迹）

**DQN 的解决方案**：
1. 卷积神经网络提取图像特征
2. 经验回放存储历史画面
3. 目标网络稳定训练

**结果**：
DQN 在 Breakout 上达到了超越人类的水平！

### 场景三：自动驾驶（连续控制）

想象训练一个自动驾驶汽车：

**状态**：
- 摄像头图像（前视、后视、侧视）
- 雷达数据（距离、速度）
- 车辆状态（速度、加速度、转向角）
- GPS 位置

**动作**：
- 方向盘角度（连续值）
- 油门（连续值）
- 刹车（连续值）

**奖励**：
- 安全行驶：+1 每步
- 到达目的地：+100
- 碰撞：-100
- 违反交通规则：-10

**挑战**：
- 高维连续状态空间
- 连续动作空间（DQN 只能处理离散动作）
- 安全关键（不能容忍随机探索）

**解决方案**：
- 用神经网络处理多模态输入
- 用策略梯度方法处理连续动作（第 6 章内容）
- 用安全约束限制探索

---

## 🧠 核心概念详解

### 概念一：为什么表格型方法失效

**直觉理解**：

表格型方法的核心假设是：**每个状态都是独立的**。

在 Q-Learning 中：
- Q[s1, a] 的更新不影响 Q[s2, a]
- 即使 s1 和 s2 非常相似，它们的 Q 值也完全独立

这在状态空间小时没问题，但在状态空间大时有两个致命问题：

**问题一：无法访问所有状态**

假设状态空间有 $10^{10}$ 个状态（这在 RL 中算小的）。

即使每秒访问 1000 个状态，需要：
$$\frac{10^{10}}{1000 \times 3600 \times 24 \times 365} \approx 317 \text{年}$$

才能访问所有状态一次！

**问题二：无法泛化**

假设你在状态 s1 学会了"这里很危险"，但从未访问过相似的状态 s2。

表格型方法：s2 的价值仍然是初始值（如 0），不知道危险。
函数近似：s2 的价值会接近 s1（因为相似），知道危险。

**形式化分析**：

表格型方法的样本复杂度是 $O(|S| \times |A|)$。

当 $|S|$ 很大时，这是不可行的。

函数近似的样本复杂度是 $O(d)$，其中 $d$ 是参数数量。

关键：$d \ll |S| \times |A|$。

### 概念二：线性函数近似

**直觉理解**：

线性函数近似是最简单的函数近似方法。

核心思想：**用特征的线性组合来近似价值函数**。

$$V(s) \approx w_1 \phi_1(s) + w_2 \phi_2(s) + ... + w_d \phi_d(s)$$

其中：
- $\phi_i(s)$ 是状态 s 的第 i 个特征
- $w_i$ 是第 i 个特征的权重

**例子**：CartPole

特征设计：
- $\phi_1(s)$ = 小车位置
- $\phi_2(s)$ = 小车速度
- $\phi_3(s)$ = 杆子角度
- $\phi_4(s)$ = 杆子角速度
- $\phi_5(s)$ = 角度²（捕捉非线性）
- $\phi_6(s)$ = 位置 × 角度（捕捉交互）

价值估计：
$$V(s) = w_1 \times \text{位置} + w_2 \times \text{速度} + ... + w_6 \times \text{位置} \times \text{角度}$$

**学习过程**：
- 初始化权重 $w = [0, 0, ..., 0]$
- 根据 TD 误差调整权重
- 权重大的特征对价值影响大

**为什么这样设计**：

线性近似的优势：
1. **简单**：实现容易，计算高效
2. **可解释**：可以分析每个特征的贡献
3. **凸优化**：保证收敛到全局最优（对于预测问题）

劣势：
1. **表达能力有限**：只能表示线性函数
2. **需要手工设计特征**：特征工程耗时

### 概念三：神经网络 Q 函数

**直觉理解**：

神经网络是线性近似的推广。

线性近似：
$$V(s) = \mathbf{w}^T \phi(s)$$

神经网络：
$$V(s) = f_L(f_{L-1}(...f_1(s; \mathbf{w}_1)...; \mathbf{w}_{L-1}); \mathbf{w}_L)$$

其中 $f_i$ 是非线性激活函数（如 ReLU）。

关键区别：
- 线性近似：特征 $\phi(s)$ 是手工设计的
- 神经网络：特征是从数据中**自动学习**的

**Q 函数近似**：

对于 Q 函数，有两种常见架构：

**架构一：单动作输出**
```
输入：状态 s + 动作 a
输出：Q(s, a)
```

每次前向传播只计算一个动作的 Q 值。

**架构二：多动作输出（DQN 使用）**
```
输入：状态 s
输出：[Q(s, a₁), Q(s, a₂), ..., Q(s, aₙ)]
```

一次前向传播计算所有动作的 Q 值。

**为什么架构二更好**：
- 计算效率高（一次前向传播 vs n 次）
- 更容易找到最大值（直接 argmax）

### 概念四：DQN 的核心创新

**问题设定**：

直接用神经网络近似 Q 函数会遇到什么问题？

**问题一：数据相关性**

RL 的数据是序列相关的：
$$(s_t, a_t, r_t, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}), ...$$

连续的数据点高度相关。

神经网络的 SGD 假设数据是独立同分布的。违反这个假设会导致：
- 训练不稳定
- 可能发散

**问题二：目标不固定**

在监督学习中，目标值是固定的（如图像标签）。

在 RL 中，目标值是：
$$y = r + \gamma \max_{a'} Q(s', a'; \theta)$$

注意：目标值也依赖于当前参数 $\theta$！

这导致：
- 目标值随着网络更新而变化
- 如同"追着移动的目标射击"
- 训练可能振荡或发散

**DQN 的解决方案**：

**创新一：经验回放（Experience Replay）**

- 存储所有转移到一个缓冲区
- 训练时随机采样批次
- 打破数据相关性

**创新二：目标网络（Target Network）**

- 用独立的网络计算目标值
- 目标网络参数 $\theta^-$ 定期更新
- 目标值在短期内固定

更新规则：
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$
$$\theta \leftarrow \theta + \alpha [y - Q(s, a; \theta)] \nabla Q(s, a; \theta)$$

### 概念五：DQN 的变体

**Double DQN**

**问题**：DQN 倾向于高估 Q 值。

原因：max 操作会选择噪声大的估计。

$$\max_a Q(s, a) \geq \mathbb{E}[Q(s, a)]$$

**解决方案**：用两个网络解耦动作选择和价值评估。

$$y = r + \gamma Q(s', \arg\max_{a} Q(s', a; \theta); \theta^-)$$

- 用当前网络 $\theta$ 选择动作
- 用目标网络 $\theta^-$ 评估价值

**Dueling DQN**

**洞察**：有些状态下，所有动作的价值都差不多（如远离危险时）。

**架构**：分离状态价值 V(s) 和优势函数 A(s, a)。

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')$$

其中：
- $V(s)$：状态本身的价值
- $A(s, a)$：动作 a 相对于平均的优势

**优势**：
- 更容易学习"哪些状态好"
- 不需要学习每个动作的绝对价值

**Prioritized Experience Replay**

**洞察**：不是所有转移都同样重要。

TD 误差大的转移包含更多信息。

**解决方案**：按 TD 误差优先级采样。

$$P(\text{采样转移 i}) \propto |\delta_i|^\alpha$$

**优势**：
- 更快学习重要转移
- 提高样本效率

---

（第一部分完成，待续...）
## 📐 公式推导

### 推导一：线性函数近似的梯度下降

**问题设定**：

我们想要学习权重 $\mathbf{w}$，使得 $\hat{V}(s, \mathbf{w})$ 接近真实价值 $V_\pi(s)$。

**损失函数**：

$$J(\mathbf{w}) = \mathbb{E}_\pi[(V_\pi(s) - \hat{V}(s, \mathbf{w}))^2]$$

**梯度下降**：

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{1}{2} \alpha \nabla_\mathbf{w} J(\mathbf{w})$$

$$\nabla_\mathbf{w} J(\mathbf{w}) = -2 \mathbb{E}_\pi[(V_\pi(s) - \hat{V}(s, \mathbf{w})) \nabla_\mathbf{w} \hat{V}(s, \mathbf{w})]$$

**问题**：真实价值 $V_\pi(s)$ 未知！

**解决方案**：用 MC 回报或 TD 目标近似。

**MC 更新**：
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha [G_t - \hat{V}(s_t, \mathbf{w})] \nabla_\mathbf{w} \hat{V}(s_t, \mathbf{w})$$

**TD 更新**：
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha [r + \gamma \hat{V}(s_{t+1}, \mathbf{w}) - \hat{V}(s_t, \mathbf{w})] \nabla_\mathbf{w} \hat{V}(s_t, \mathbf{w})$$

**线性情况的简化**：

对于线性近似 $\hat{V}(s, \mathbf{w}) = \mathbf{w}^T \phi(s)$：

$$\nabla_\mathbf{w} \hat{V}(s, \mathbf{w}) = \phi(s)$$

TD 更新简化为：
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \phi(s)$$

其中 $\delta = r + \gamma \mathbf{w}^T \phi(s') - \mathbf{w}^T \phi(s)$。

### 推导二：DQN 的梯度

**问题设定**：

DQN 用神经网络近似 Q 函数：$Q(s, a; \theta)$。

**损失函数**：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}[(y - Q(s, a; \theta))^2]$$

其中目标值：
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**梯度**：

$$\nabla_\theta L(\theta) = -2 \mathbb{E}[(y - Q(s, a; \theta)) \nabla_\theta Q(s, a; \theta)]$$

**SGD 更新**：

$$\theta \leftarrow \theta + \alpha (y - Q(s, a; \theta)) \nabla_\theta Q(s, a; \theta)$$

**关键理解**：

注意目标值 $y$ 不依赖于 $\theta$（因为用了目标网络 $\theta^-$）。

这使得梯度下降稳定。

### 推导三：Double DQN 的无偏性

**问题**：标准 DQN 高估 Q 值。

**证明**：

$$\mathbb{E}[\max_a Q(s, a)] \geq \max_a \mathbb{E}[Q(s, a)]$$

这是因为 max 是凸函数，由 Jensen 不等式。

**Double DQN 的目标**：

$$y = r + \gamma Q(s', \arg\max_a Q(s', a; \theta); \theta^-)$$

**关键**：
- 动作选择：$\arg\max_a Q(s', a; \theta)$ 用当前网络
- 价值评估：$Q(s', \cdot; \theta^-)$ 用目标网络

这减少了高估，因为：
- 当前网络选择它认为最好的动作
- 目标网络独立评估这个动作的价值
- 两个网络的噪声不太可能同时偏向同一个动作

### 推导四：Dueling DQN 的可识别性

**问题**：从 Q 值分解出 V 和 A 不是唯一的。

如果 $Q(s, a) = V(s) + A(s, a)$，那么对于任意函数 $f(s)$：
$$Q(s, a) = (V(s) + f(s)) + (A(s, a) - f(s))$$

也是有效的分解。

**解决方案**：添加约束。

**约束一**（Dueling DQN 使用）：
$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')$$

这强制 $\sum_a A(s, a) = 0$，保证唯一分解。

**约束二**（替代方案）：
$$Q(s, a) = V(s) + A(s, a) - A(s, \arg\max_a A(s, a))$$

强制最优动作的优势为 0。

---

## 💻 算法实现

### 实现一：线性函数近似

```python
"""
线性函数近似

用特征的线性组合近似价值函数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class LinearValueFunction:
    """
    线性价值函数
    
    V(s, w) = w^T · φ(s)
    
    Attributes:
        n_features (int): 特征维度
        weights (np.ndarray): 权重向量
        lr (float): 学习率
    """
    
    def __init__(self, n_features: int, lr: float = 0.01):
        """
        初始化线性价值函数
        
        Args:
            n_features: 特征维度
            lr: 学习率
        """
        self.n_features = n_features
        self.lr = lr
        
        # 初始化权重为小的随机值
        self.weights = np.random.randn(n_features) * 0.01
    
    def predict(self, features: np.ndarray) -> float:
        """
        预测状态价值
        
        Args:
            features: 状态特征向量 φ(s)
            
        Returns:
            V(s): 价值估计
        """
        return np.dot(self.weights, features)
    
    def update(self, features: np.ndarray, target: float) -> float:
        """
        用 TD 误差更新权重
        
        更新规则：
            w ← w + α * (target - V(s)) * φ(s)
        
        Args:
            features: 状态特征向量
            target: TD 目标（r + γ·V(s')）
            
        Returns:
            td_error: TD 误差
        """
        # 当前预测
        prediction = self.predict(features)
        
        # TD 误差
        td_error = target - prediction
        
        # 梯度下降更新
        self.weights += self.lr * td_error * features
        
        return td_error
    
    def batch_update(self, features_batch: np.ndarray, targets_batch: np.ndarray) -> float:
        """
        批量更新权重
        
        Args:
            features_batch: 特征批次 [batch_size, n_features]
            targets_batch: 目标批次 [batch_size]
            
        Returns:
            mse: 均方误差
        """
        # 预测
        predictions = features_batch @ self.weights
        
        # 误差
        errors = targets_batch - predictions
        
        # 梯度（平均）
        gradient = -features_batch.T @ errors / len(errors)
        
        # 更新
        self.weights -= self.lr * gradient
        
        # 返回 MSE
        mse = np.mean(errors ** 2)
        return mse


class LinearQFunction:
    """
    线性 Q 函数
    
    Q(s, a, w) = w_a^T · φ(s)
    
    每个动作有独立的权重向量
    """
    
    def __init__(self, n_features: int, n_actions: int, lr: float = 0.01):
        """
        初始化线性 Q 函数
        
        Args:
            n_features: 特征维度
            n_actions: 动作数量
            lr: 学习率
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        
        # 每个动作一个权重向量
        self.weights = np.random.randn(n_actions, n_features) * 0.01
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测所有动作的 Q 值
        
        Args:
            features: 状态特征向量
            
        Returns:
            Q_values: [Q(s, a_0), Q(s, a_1), ...]
        """
        return self.weights @ features
    
    def get_action(self, features: np.ndarray, epsilon: float = 0.1) -> int:
        """
        ε-greedy 动作选择
        
        Args:
            features: 状态特征
            epsilon: 探索率
            
        Returns:
            action: 选择的动作
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.predict(features)
            return np.argmax(q_values)
    
    def update(self, features: np.ndarray, action: int, target: float) -> float:
        """
        更新指定动作的 Q 值
        
        更新规则：
            w_a ← w_a + α * (target - Q(s, a)) * φ(s)
        
        Args:
            features: 状态特征
            action: 动作索引
            target: TD 目标
            
        Returns:
            td_error: TD 误差
        """
        # 当前预测
        prediction = self.weights[action] @ features
        
        # TD 误差
        td_error = target - prediction
        
        # 更新（只更新选中动作的权重）
        self.weights[action] += self.lr * td_error * features
        
        return td_error
```

### 实现二：DQN 核心算法

```python
"""
DQN (Deep Q-Network) 算法

用深度神经网络近似 Q 函数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
from typing import Dict, List, Tuple, Optional


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储转移 (s, a, r, s', done)，支持随机采样
    """
    
    def __init__(self, capacity: int = int(1e6)):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        存储转移
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否终止
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        随机采样批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            states, actions, rewards, next_states, dones
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """
    DQN 网络
    
    输入：状态
    输出：每个动作的 Q 值
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        """
        初始化 DQN
        
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)


class DQNAgent:
    """
    DQN 智能体
    
    关键特性：
    - 经验回放
    - 目标网络
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_size: int = int(1e6),
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        """
        初始化 DQN 智能体
        
        Args:
            state_dim: 状态维度
            n_actions: 动作数量
            lr: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            buffer_size: 回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.q_network = DQN(state_dim, n_actions).to(self.device)
        self.target_network = DQN(state_dim, n_actions).to(self.device)
        
        # 初始化目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 回放缓冲区
        self.buffer = ReplayBuffer(buffer_size)
        
        # 训练计数
        self.step_count = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy 动作选择
        
        Args:
            state: 当前状态
            
        Returns:
            action: 选择的动作
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """存储转移"""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        执行一次 DQN 更新
        
        Returns:
            stats: 训练统计
        """
        # 检查缓冲区是否有足够数据
        if len(self.buffer) < self.batch_size:
            return {}
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 转换为 tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算目标值
        with torch.no_grad():
            # 目标网络计算下一状态的 Q 值
            next_q_values = self.target_network(next_states)
            # 选择最大 Q 值
            max_next_q = next_q_values.max(dim=1)[0]
            # TD 目标
            targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 计算当前 Q 值
        q_values = self.q_network(states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算损失
        loss = F.mse_loss(q_values_for_actions, targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {
            'loss': loss.item(),
            'q_value_mean': q_values_for_actions.mean().item(),
        }
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = True) -> List[float]:
        """
        训练 DQN
        
        Args:
            env: Gymnasium 环境
            n_episodes: 训练 episode 数
            verbose: 是否打印进度
            
        Returns:
            episode_rewards: 每个 episode 的总奖励
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                # 选择动作
                action = self.select_action(state)
                
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
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward (last 10): {avg_reward:.2f}")
        
        return episode_rewards
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
```

（第二部分完成，待续...）
## 🔬 算法对比

### 表格型 vs 函数近似

| 维度 | 表格型 | 函数近似 |
|------|--------|----------|
| **状态表示** | 离散索引 | 连续特征向量 |
| **内存需求** | O(|S|×|A|) | O(参数数量) |
| **泛化能力** | 无 | 有 |
| **样本效率** | 低（每个状态独立学） | 高（相似状态共享知识） |
| **理论保证** | 收敛到最优 | 可能收敛到局部最优 |
| **适用场景** | 小状态空间 | 大状态空间/连续空间 |

### DQN 变体对比

| 算法 | 核心改进 | 优势 | 劣势 |
|------|----------|------|------|
| **DQN** | 基础版本 | 简单 | 高估 Q 值 |
| **Double DQN** | 解耦动作选择和价值评估 | 减少高估 | 实现稍复杂 |
| **Dueling DQN** | 分离 V 和 A | 更快学习状态价值 | 需要额外网络头 |
| **Prioritized Replay** | 按 TD 误差采样 | 更高样本效率 | 需要维护优先级 |

### 线性 vs 神经网络近似

| 维度 | 线性 | 神经网络 |
|------|------|----------|
| **表达能力** | 低（只能线性） | 高（万能近似） |
| **特征工程** | 需要手工设计 | 自动学习 |
| **训练速度** | 快 | 慢 |
| **可解释性** | 高 | 低 |
| **收敛保证** | 凸优化 | 非凸，可能局部最优 |

---

## 🧪 动手实验

### 实验一：线性近似 vs 表格型 Q-Learning

**任务描述**：

在 CartPole 上对比线性近似和表格型 Q-Learning。

**实验设置**：

**表格型**：
- 将连续状态离散化（如位置分 10 档，角度分 10 档...）
- 总状态数：10×10×10×10 = 10000
- 用 Q-Learning 训练

**线性近似**：
- 特征：原始状态 + 平方项 + 交叉项
- 特征维度：约 20
- 用线性 Q 函数训练

**分析指标**：
- 学习曲线（每 10 集平均奖励）
- 最终性能
- 训练稳定性

**预期结果**：
- 线性近似学习更快（泛化）
- 表格型可能需要更多 episode
- 线性近似更稳定

### 实验二：DQN 超参数敏感性分析

**任务描述**：

研究 DQN 关键超参数对训练的影响。

**参数设置**：

| 参数 | 组 A | 组 B | 组 C |
|------|------|------|------|
| 学习率 | 0.001 | 0.01 | 0.1 |
| 回放缓冲区 | 1000 | 10000 | 100000 |
| 目标网络更新 | 10 | 100 | 1000 |

**实验步骤**：

1. 对每组参数，训练 DQN 在 CartPole 上
2. 记录学习曲线
3. 分析收敛速度和最终性能

**预期现象**：
- 学习率太大：训练不稳定
- 学习率太小：学习慢
- 缓冲区太小：数据相关性高
- 缓冲区太大：旧数据过多
- 目标网络更新太频繁：不稳定
- 更新太少：目标过时

### 实验三：Double DQN 实现与对比

**任务描述**：

实现 Double DQN 并与标准 DQN 对比。

**实现提示**：

```python
# 标准 DQN
with torch.no_grad():
    next_q = self.target_network(next_states)
    max_next_q = next_q.max(dim=1)[0]
    targets = rewards + gamma * max_next_q

# Double DQN
with torch.no_grad():
    # 当前网络选择动作
    action = self.q_network(next_states).argmax(dim=1)
    # 目标网络评估
    next_q = self.target_network(next_states)
    selected_q = next_q.gather(1, action.unsqueeze(1)).squeeze(1)
    targets = rewards + gamma * selected_q
```

**分析指标**：
- Q 值估计的偏差（与真实值比较）
- 训练稳定性
- 最终性能

### 实验四：Atari 游戏训练

**任务描述**：

用 DQN 训练 Atari Breakout。

**实现步骤**：

1. 安装 Atari 环境：`pip install gymnasium[atari]`
2. 实现图像预处理（灰度化、下采样、帧堆叠）
3. 实现 CNN 架构的 DQN
4. 训练并可视化结果

**CNN 架构**：
```python
class AtariDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),  # 4 帧堆叠
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
```

**预期结果**：
- 训练 100 万帧后达到人类水平
- 学会有用的策略（如挖隧道）

---

## ❓ 常见问题

### Q1: 为什么 DQN 用目标网络？不用会怎样？

**A**: 目标网络解决"移动目标"问题。

**不用目标网络**：
$$y = r + \gamma \max_{a'} Q(s', a'; \theta)$$

注意 $y$ 依赖于 $\theta$，而 $\theta$ 正在被更新。

这导致：
- 目标值随网络变化
- 如同追移动的目标
- 可能发散

**数学分析**：

考虑简单情况 $Q(s, a; \theta) = \theta$（常数函数）。

更新规则：
$$\theta \leftarrow \theta + \alpha (r + \gamma \theta - \theta) = \theta + \alpha (r - (1-\gamma)\theta)$$

如果 $\gamma$ 接近 1，这会导致振荡。

**目标网络**固定 $\theta^-$ 一段时间，使目标稳定。

### Q2: 经验回放为什么能提高稳定性？

**A**: 打破数据相关性。

**问题**：RL 数据是序列相关的。

连续转移：
$$(s_t, a_t, r_t, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}), ...$$

高度相关。

**SGD 假设**：数据独立同分布（i.i.d.）。

违反假设导致：
- 梯度估计有偏
- 训练不稳定

**经验回放**：
- 随机采样批次
- 打破时间相关性
- 近似 i.i.d.

### Q3: 为什么 DQN 会高估 Q 值？

**A**: max 操作的统计偏差。

**数学分析**：

假设 $Q(s, a)$ 有无噪声估计：
$$\hat{Q}(s, a) = Q^*(s, a) + \epsilon_a$$

其中 $\epsilon_a$ 是零均值噪声。

max 操作的期望：
$$\mathbb{E}[\max_a \hat{Q}(s, a)] \geq \max_a \mathbb{E}[\hat{Q}(s, a)] = \max_a Q^*(s, a)$$

这是因为 max 是凸函数（Jensen 不等式）。

**直观理解**：
- 噪声大的动作更可能被选中
- 选中的动作倾向于正噪声
- 导致高估

**Double DQN 解决方案**：
- 用一个网络选择动作
- 用另一个网络评估
- 减少噪声相关性

### Q4: 如何选择函数近似的架构？

**A**: 取决于任务特点。

**低维状态（< 100 维）**：
- 全连接网络
- 2-3 层，64-256 隐藏单元

**图像输入**：
- CNN
- 参考 DQN 架构

**序列输入**：
- RNN/LSTM
- Transformer

**多模态输入**：
- 多分支网络
- 融合层

**经验法则**：
1. 从简单架构开始
2. 如果欠拟合，增加容量
3. 如果过拟合，加正则化
4. 监控训练/测试差距

### Q5: 函数近似一定比表格型好吗？

**A**: 不一定。

**函数近似优势场景**：
- 状态空间大
- 连续状态
- 需要泛化

**表格型优势场景**：
- 状态空间小（< 10000）
- 需要精确最优解
- 理论分析

**混合方法**：
- 粗粒度表格 + 细粒度近似
- 局部加权学习
- 决策树 + 神经网络

---

## 📚 延伸阅读

### 核心论文

1. **Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.**
   - DQN 原始论文
   - https://www.nature.com/articles/nature14236

2. **Van Hasselt et al. (2016). Deep reinforcement learning with double Q-learning.**
   - Double DQN
   - https://arxiv.org/abs/1509.06461

3. **Wang et al. (2016). Dueling network architectures for deep reinforcement learning.**
   - Dueling DQN
   - https://arxiv.org/abs/1511.06581

### 教程

1. **Spinning Up in Deep RL**
   - https://spinningup.openai.com
   - DQN 实现教程

2. **Deep RL Bootcamp**
   - https://sites.google.com/view/deep-rl-bootcamp
   - Berkeley 的深度学习 RL 课程

### 代码资源

| 文件 | 内容 | 行数 |
|------|------|------|
| `linear_approximation.py` | 线性近似 | ~150 |
| `neural_network_q.py` | 神经网络 Q 函数 | ~100 |
| `dqn.py` | DQN 算法 | ~250 |
| `games/cartpole_balance.py` | CartPole 环境 | ~80 |

---

## ✅ 本章检查清单

学完本章后，你应该能够：

**概念理解**：
- [ ] 解释为什么需要函数近似
- [ ] 区分线性和神经网络近似
- [ ] 解释经验回放的作用
- [ ] 解释目标网络的作用

**数学推导**：
- [ ] 推导线性近似的梯度下降
- [ ] 推导 DQN 的梯度
- [ ] 解释 Double DQN 的无偏性
- [ ] 解释 Dueling DQN 的可识别性

**代码实现**：
- [ ] 实现线性 Q 函数
- [ ] 实现 DQN 智能体
- [ ] 实现经验回放
- [ ] 实现目标网络

**实验分析**：
- [ ] 对比表格型和函数近似
- [ ] 分析 DQN 超参数影响
- [ ] 实现 Double DQN
- [ ] 训练 Atari 游戏

**应用判断**：
- [ ] 根据任务选择近似架构
- [ ] 合理设置超参数
- [ ] 诊断训练问题

---

## 📝 课后测验（10 分钟）

### 基础题（必答）

**1. 为什么需要函数近似？表格方法有什么局限？**

<details>
<summary>点击查看答案</summary>

**答案**：

表格方法局限：
1. 状态空间太大（如 Atari：10^147733 种画面）
2. 无法泛化到未见状态
3. 无法处理连续空间

函数近似优势：
1. 泛化：相似状态有相似价值
2. 内存效率：只存参数，不存所有状态
3. 连续空间：直接处理
</details>

---

**2. DQN 的两个关键技术是什么？作用是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

1. **经验回放**：
   - 存储转移到缓冲区
   - 随机采样打破相关性
   - 提高数据效率

2. **目标网络**：
   - 独立网络计算目标值
   - 定期更新参数
   - 避免"移动目标"问题
</details>

---

**3. Double DQN 解决了什么问题？**

<details>
<summary>点击查看答案</summary>

**答案**：

**问题**：DQN 倾向于高估 Q 值（max 操作的统计偏差）

**解决方案**：
- 用当前网络选择动作
- 用目标网络评估价值
- 减少过估计
</details>

---

### 进阶题（选答）

**4. Dueling DQN 为什么要分离 V(s) 和 A(s,a)？**

<details>
<summary>点击查看答案</summary>

**答案**：

洞察：有些状态下，所有动作价值差不多。

优势：
- 更容易学习"哪些状态好"
- 不需要学习每个动作的绝对价值
- 更快收敛
</details>

---

**5. 经验回放为什么能提高稳定性？**

<details>
<summary>点击查看答案</summary>

**答案**：

RL 数据是序列相关的，违反 SGD 的 i.i.d.假设。

经验回放：
- 随机采样批次
- 打破时间相关性
- 近似 i.i.d.
- 梯度估计更准确
</details>

---

### 编程题（实践）

**6. 实现 Double DQN，并与标准 DQN 对比**

<details>
<summary>查看提示</summary>

**提示**：
1. Double DQN 的 target：
   y = r + γ * Q_target(s', argmax_a Q_current(s',a))
2. 在 CartPole 上对比
3. 分析 Q 值估计的差异
</details>

---

## 🚀 下一章预告

**第 6 章：策略梯度方法**

当动作空间是连续的，或者需要随机策略时，Q 学习方法不再适用。本章将介绍：
- REINFORCE 算法（蒙特卡洛策略梯度）
- Actor-Critic 方法
- A2C、DDPG、TD3 等算法
- 连续控制应用

**预告实验**：训练机械臂抓取物体！🎯

---

*最后更新：2026-04-22*  
*作者：Hermes <neko_yukirin@qq.com>*
