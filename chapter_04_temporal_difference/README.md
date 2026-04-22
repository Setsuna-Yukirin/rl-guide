# 第 4 章：时序差分学习 (Temporal Difference Learning) ⭐

> **强化学习的核心突破** —— 结合动态规划与蒙特卡洛的优点，实现高效的在线学习

---

## ⏱️ TL;DR（30 秒速览）

**核心问题**：如何**结合 DP 和 MC 的优点**？

**核心思想**：
> 像 DP 一样每步更新（自举），像 MC 一样无需模型（采样）。

**三种算法**：
- **TD(0)**：单步时序差分预测
- **SARSA**：在策略 TD 控制（怂，考虑探索）
- **Q-Learning**：离策略 TD 控制（头铁，假设最优）

**核心公式**：
- TD 误差：δ = R + γV(s') - V(s)
- SARSA：Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
- Q-Learning：Q(s,a) ← Q(s,a) + α[R + γmax_a'Q(s',a') - Q(s,a)]

**关键洞察**：
- TD 每步更新，无需等 episode 结束
- SARSA 学当前策略，Q-Learning 学最优策略
- 悬崖行走：SARSA 绕远路，Q-Learning 贴悬崖

**学完这章你能**：
- ✅ 理解 TD 误差的含义
- ✅ 区分 SARSA 和 Q-Learning
- ✅ 实现 TD 控制算法
- ✅ 解释探索 - 利用权衡

**代码实现**：TD 预测、SARSA、Q-Learning、Expected SARSA

**常见误区**：
- ❌ Q-Learning 一定比 SARSA 好 → ✅ 取决于场景
- ❌ TD 是有偏的所以不好 → ✅ 偏差换方差，通常更好
- ❌ SARSA 和 Q-Learning 只是实现不同 → ✅ 本质是在/离策略区别

---

## 🎯 本章要解决什么问题

时序差分学习（Temporal Difference Learning，简称 TD 学习）是整个强化学习理论体系中**最重要的突破之一**。要理解它的重要性，我们需要先回顾一下在它之前，强化学习面临的困境。

在 TD 学习出现之前，强化学习主要有两种方法：**动态规划（Dynamic Programming, DP）**和**蒙特卡洛方法（Monte Carlo, MC）**。这两种方法各有优劣，但都存在致命缺陷。

**动态规划**需要完整的环境模型——也就是说，你必须知道从任何状态采取任何动作后，会以什么概率转移到什么状态、获得什么奖励。这个要求在实际应用中几乎不可能满足。想象一下，如果你要训练一个机器人走路，你需要精确知道电机施加多大扭矩会让机器人以什么概率走到什么位置——这几乎是不可能的。动态规划的另一个问题是计算量大，每次更新都需要遍历所有状态。

**蒙特卡洛方法**不需要环境模型，它通过实际采样来学习。这听起来很完美，但它有一个致命问题：**必须等到一个完整的 episode 结束才能更新**。在某些应用场景中，一个 episode 可能非常长（比如一局围棋可能有 200 步），这意味着你要等很久才能得到一次学习信号。更糟糕的是，有些任务是没有明确终点的（比如持续运行的推荐系统），这时蒙特卡洛方法完全无法使用。

**时序差分学习**的突破在于：它**结合了两种方法的优点**——像蒙特卡洛一样不需要环境模型，像动态规划一样可以单步更新。这个看似简单的想法，却是强化学习从理论走向实践的关键转折点。

学完本章后，你将能够：
- 深刻理解 TD 学习的核心思想和数学原理
- 掌握 SARSA 和 Q-Learning 两大经典算法
- 理解在策略（on-policy）和离策略（off-policy）学习的本质区别
- 能够根据实际应用场景选择合适的算法
- 为后续学习深度强化学习（DQN、PPO 等）打下坚实基础

---

## 📖 场景描述

### 场景一：悬崖行走

想象你站在一个 4 行 12 列的网格左下角。你的目标是走到右下角的终点。但是，网格的最下面一行（除了起点和终点）全是悬崖——只要你踩上去，就会掉下去，回到起点，并且受到 -100 分的严厉惩罚。每走一步，你还会受到 -1 分的小惩罚（鼓励你尽快到达终点）。

你可以选择四个方向：上、下、左、右。但是，如果你的移动会让你走出网格边界，你会留在原地，但仍然受到 -1 分的惩罚。

现在有两个智能体要学习走这个迷宫：

**智能体 A（保守型）**：它在学习过程中非常谨慎。每当它靠近悬崖时，它会想："如果我下一步不小心走错了，掉下悬崖怎么办？"所以它选择了一条远离悬崖的安全路线，即使这条路要多走几步。

**智能体 B（激进型）**：它在学习过程中非常自信。它总是假设自己下一步会走出最优的动作，所以它学会了贴着悬崖边缘走最短路径。

在训练过程中，智能体 A 很少掉下悬崖，但走的路线较长；智能体 B 偶尔会掉下悬崖（特别是在探索阶段），但它找到了理论上的最短路径。

这个场景中，智能体 A 使用的是**SARSA**算法（在策略学习），智能体 B 使用的是**Q-Learning**算法（离策略学习）。它们的核心区别在于：**学习时如何考虑未来的动作**。

### 场景二：估计通勤时间

让我们用一个生活化的例子来理解 TD 学习的核心思想。

假设你每天开车从家到公司，你想估计在不同路段上还需要多少时间才能到达公司。

**蒙特卡洛方法**的做法是：每天早上出发，晚上到家后，根据今天实际花费的总时间来更新你的估计。比如，你今天从家到公司实际用了 35 分钟，而你之前的估计是 30 分钟，那你就会把估计调高一些。但问题是，你要等一整天才能得到一次更新信号。

**动态规划**的做法是：你需要知道每条路的精确路况模型——比如从 A 路口到 B 路口在早高峰需要多少分钟、平峰需要多少分钟。然后你在家就可以用这些模型计算出精确的时间估计。但问题是，路况是动态变化的，你不可能有精确的模型。

**TD 学习**的做法是：你开车到半路时，看导航软件说"从这里到公司还要 15 分钟"。你回想一下，出发时你以为从这里到公司要 20 分钟。那你现在就知道："哦，我高估了 5 分钟"。于是你立刻更新你的估计，不需要等到达公司。

**这就是 TD 学习的核心：边走边更新，用后续的估计来修正当前的估计。** 这种方法叫做"自举"（Bootstrapping）——你自己拉着自己的鞋带把自己提起来。听起来很荒谬，但在强化学习中，它确实有效。

---

## 🧠 核心概念详解

### 概念一：自举（Bootstrapping）

**直觉理解**：

自举这个词来源于英文谚语"Pull yourself up by your bootstraps"（拉着自己的鞋带把自己提起来），意思是靠自己的力量完成看似不可能的任务。在强化学习中，自举指的是**用后续的估计值来更新当前的估计值**。

这听起来像是循环论证，但实际上非常有效。想象一下，你要估计从北京到上海的距离。你不知道确切数字，但你知道：
- 从北京到南京大约 1000 公里
- 从南京到上海大约 300 公里

那你可以估计北京到上海大约 1300 公里。你用了对南京到上海的估计来更新北京到上海的估计——这就是自举。

**形式化定义**：

在强化学习中，自举体现在价值函数的更新公式中。对于状态价值函数 V(s)，TD 学习的更新公式是：

$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

注意公式中的 $V(s_{t+1})$ —— 这是**对下一个状态价值的估计**，而不是真实值。我们用这个估计值来更新当前的估计值，这就是自举。

**为什么这样设计**：

自举的设计动机来自对动态规划和蒙特卡洛方法的深入分析。

动态规划使用自举（它用后继状态的价值来更新当前状态的价值），所以它可以单步更新，但它需要完整的环境模型。蒙特卡洛不使用自举（它用实际的回报来更新），所以它不需要环境模型，但要等 episode 结束。

TD 学习的洞察是：**自举本身是有价值的，它让单步更新成为可能；问题不在于自举，而在于需要环境模型**。如果我们能去掉对环境模型的依赖，同时保留自举的优点，那就能结合两种方法的优点。

TD 学习做到了这一点：它用**采样**代替环境模型（像蒙特卡洛一样），用**自举**实现单步更新（像动态规划一样）。

### 概念二：TD 误差（TD Error）

**直觉理解**：

TD 误差是 TD 学习的核心驱动力。它衡量的是**你的预期和实际体验之间的差距**。

继续用通勤时间的例子：
- 你以为从当前位置到公司要 20 分钟（这是你的预期 V(s_t)）
- 你开了 5 分钟到了下一个位置（这是实际得到的奖励 r_{t+1}）
- 导航说从这里到公司还要 12 分钟（这是对未来的新估计 V(s_{t+1})）

那你的 TD 误差就是：5 + 12 - 20 = -3 分钟。负值说明你高估了，下次要调低估计。

**形式化定义**：

TD 误差 δ_t 定义为：

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

其中：
- $r_{t+1}$ 是 t+1 时刻获得的即时奖励
- $\gamma$ 是折扣因子（通常取 0.9 到 0.99）
- $V(s_{t+1})$ 是下一状态的价值估计
- $V(s_t)$ 是当前状态的价值估计

**为什么这样设计**：

TD 误差的设计背后有深刻的数学原理。在强化学习中，我们想要学习的是价值函数 V(s)，它满足贝尔曼方程：

$$V(s) = \mathbb{E}[r + \gamma V(s') | s]$$

这个方程说：一个状态的价值等于从该状态出发的期望回报。但问题是，我们不知道期望值（因为不知道环境模型）。

TD 误差提供了一个巧妙的解决方案：**用采样来近似期望**。每次我们采样一个转移 (s, r, s')，我们就可以计算一个 TD 误差，然后用它来更新价值估计。从随机近似的理论可以证明，如果学习率适当衰减，这个过程会收敛到正确的价值函数。

### 概念三：在策略 vs 离策略学习

**直觉理解**：

这是强化学习中最重要的概念区分之一，也是 SARSA 和 Q-Learning 的本质区别。

**在策略（On-Policy）**：你学习的是**当前正在执行的策略**的价值。也就是说，你学习时会考虑你的探索行为。比如 SARSA，它在更新时会用下一步实际采取的动作（可能是探索动作）来计算目标值。

**离策略（Off-Policy）**：你学习的是**某个目标策略**的价值，而你用来学习的数据可以来自另一个不同的策略。比如 Q-Learning，它在更新时总是假设下一步会采取最优动作，不管实际采取的是什么动作。

**形式化定义**：

在策略学习：行为策略 μ = 目标策略 π

离策略学习：行为策略 μ ≠ 目标策略 π

其中行为策略是实际用来选择动作的策略，目标策略是我们想要学习的策略。

**为什么这样设计**：

这个区分的重要性在于**探索与利用的权衡**。

在策略学习（如 SARSA）的优点是：它学习到的策略考虑了探索的影响，所以更安全。在悬崖行走的例子中，SARSA 知道"我有时候会随机探索"，所以它学会了远离悬崖。

离策略学习（如 Q-Learning）的优点是：它可以直接学习最优策略，不受探索策略的影响。所以它能找到理论上的最短路径。但缺点是：它假设自己总是能执行最优动作，这在实际中可能不成立（比如有探索噪声时）。

选择哪种方式取决于应用场景：
- 如果探索成本高（真实机器人、自动驾驶）→ 在策略
- 如果可以在仿真中充分探索（游戏 AI）→ 离策略

---

## 📐 公式推导

### 推导一：TD(0) 预测算法

**问题设定**：

给定一个固定的策略 π，我们想要估计它的状态价值函数 V_π(s)。V_π(s) 的定义是：从状态 s 出发，遵循策略 π，期望获得的累积折扣奖励。

**贝尔曼期望方程**：

价值函数满足以下方程：

$$V_\pi(s) = \mathbb{E}_\pi[r_{t+1} + \gamma V_\pi(s_{t+1}) | s_t = s]$$

这个方程的直观含义是：当前状态的价值等于即时奖励的期望加上折扣后下一状态价值的期望。

**推导过程**：

我们想要从上述方程推导出一个可以实际使用的更新规则。

第一步，注意到期望值可以用采样来近似。如果我们从策略 π 采样一个转移 (s_t, r_{t+1}, s_{t+1})，那么：

$$r_{t+1} + \gamma V_\pi(s_{t+1})$$

是 $V_\pi(s_t)$ 的一个无偏估计。

第二步，定义 TD 误差为估计值与当前估计的差：

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

第三步，使用随机梯度下降来最小化 TD 误差的平方。目标函数是：

$$J(V) = \mathbb{E}[\frac{1}{2}\delta_t^2]$$

对 V(s_t) 求梯度：

$$\nabla_{V(s_t)} J(V) = -\delta_t \cdot \nabla_{V(s_t)} V(s_t) = -\delta_t$$

第四步，得到更新规则：

$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t = V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**收敛性说明**：

从随机近似理论可以证明，如果学习率满足以下条件：
- $\sum_t \alpha_t = \infty$（学习率之和发散，保证能学到任何东西）
- $\sum_t \alpha_t^2 < \infty$（学习率平方和收敛，保证最终稳定）

那么 TD(0) 算法会收敛到 V_π。

### 推导二：Q-Learning 的离策略学习

**问题设定**：

我们想要学习最优动作价值函数 Q*(s, a)，它表示从状态 s 采取动作 a 开始，之后都采取最优动作，期望获得的累积折扣奖励。

**贝尔曼最优方程**：

最优动作价值函数满足：

$$Q^*(s, a) = \mathbb{E}[r_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') | s_t = s, a_t = a]$$

**推导过程**：

第一步，定义 Q-Learning 的 TD 误差：

$$\delta_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$

注意这里的关键：**max 操作**。这意味着我们假设在下一状态会采取最优动作。

第二步，更新规则：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t$$

第三步，分析离策略性质。假设我们的行为策略是 ε-greedy（以 1-ε 的概率选择最优动作，以 ε 的概率随机探索），但更新时用的是 max 操作（假设采取最优动作）。这意味着：

- 行为策略：μ(a|s) = (1-ε) 如果 a = argmax Q(s,a)，否则 ε/|A|
- 目标策略：π(a|s) = 1 如果 a = argmax Q(s,a)，否则 0

显然 μ ≠ π，所以这是离策略学习。

**收敛性定理**：

Q-Learning 的收敛性由 Watkins 和 Dayan (1992) 证明。关键条件是：
1. 所有状态 - 动作对被无限次访问（GLIE 条件）
2. 学习率满足随机近似条件
3. 奖励有界

在这些条件下，Q-Learning 几乎必然收敛到 Q*。

### 推导三：SARSA 的在策略学习

**问题设定**：

与 Q-Learning 不同，SARSA 想要学习的是**当前行为策略**的动作价值函数 Q_π(s, a)。

**推导过程**：

第一步，定义 SARSA 的 TD 误差：

$$\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$

注意与 Q-Learning 的区别：这里用的是 $Q(s_{t+1}, a_{t+1})$，其中 $a_{t+1}$ 是**实际采取的下一个动作**，而不是最优动作。

第二步，更新规则与 Q-Learning 相同：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t$$

第三步，分析在策略性质。SARSA 的行为策略和目标策略都是 ε-greedy：

- 行为策略：μ(a|s) = (1-ε) 如果 a = argmax Q(s,a)，否则 ε/|A|
- 目标策略：π(a|s) = (1-ε) 如果 a = argmax Q(s,a)，否则 ε/|A|

所以 μ = π，这是在策略学习。

**收敛性**：

SARSA 的收敛性由 Singh 等人 (2000) 证明。关键条件是：
1. 策略逐渐变为贪婪（ε → 0）
2. 学习率满足随机近似条件

在这些条件下，SARSA 收敛到 Q_π，其中 π 是最优策略。

---

## 💻 算法实现

### 算法一：TD(0) 预测

```python
"""
TD(0) 预测算法

用于估计给定策略 π 的状态价值函数 V_π

核心公式：
    V(s) ← V(s) + α * [r + γ * V(s') - V(s)]

算法复杂度：
    - 时间：O(每个 episode 的步数)
    - 空间：O(|S|)，需要存储所有状态的价值
"""

import numpy as np
from typing import Tuple, Optional


class TDPredictor:
    """
    TD(0) 预测器
    
    估计给定策略的状态价值函数
    
    Attributes:
        alpha (float): 学习率，控制更新幅度
        gamma (float): 折扣因子，平衡短期和长期回报
        V (np.ndarray): 状态价值函数表
    """
    
    def __init__(self, n_states: int, alpha: float = 0.1, gamma: float = 0.99):
        """
        初始化 TD 预测器
        
        Args:
            n_states: 状态空间大小
            alpha: 学习率
                   - 太大 (0.5): 更新幅度过大，导致震荡
                   - 太小 (0.01): 学习速度慢
                   - 推荐值：0.1（平衡收敛速度和稳定性）
            gamma: 折扣因子
                   - 0.9: 更重视短期回报
                   - 0.99: 几乎考虑全部未来回报
                   - 1.0: 不考虑折扣（仅适用于有终止状态的任务）
        """
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros(n_states)  # 初始化所有状态价值为 0
        
    def update(self, state: int, reward: float, next_state: int, done: bool) -> float:
        """
        执行单步 TD 更新
        
        这是 TD(0) 算法的核心实现，对应公式：
        V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
        
        Args:
            state: 当前状态 s_t 的索引
            reward: 即时奖励 r_{t+1}
            next_state: 下一状态 s_{t+1} 的索引
            done: 是否到达终止状态
            
        Returns:
            td_error: 计算得到的 TD 误差值（可用于分析学习过程）
            
        Note:
            当 done=True 时，下一状态的价值 V(s') = 0
            这是因为终止状态没有未来回报
        """
        # 获取当前状态的价值估计
        v_current = self.V[state]
        
        # 获取下一状态的价值估计
        # 如果是终止状态，未来价值为 0
        v_next = 0.0 if done else self.V[next_state]
        
        # 计算 TD 误差 ⭐ 核心公式
        # TD 误差 = 实际得到的 + 对未来估计 - 之前的估计
        td_error = reward + self.gamma * v_next - v_current
        
        # 更新当前状态的价值估计
        # 朝着 TD 误差的方向调整，调整幅度由学习率控制
        self.V[state] = v_current + self.alpha * td_error
        
        return td_error
    
    def predict(self, state: int) -> float:
        """
        预测给定状态的价值
        
        Args:
            state: 状态索引
            
        Returns:
            该状态的价值估计
        """
        return self.V[state]
    
    def get_value_function(self) -> np.ndarray:
        """
        获取完整的价值函数表
        
        Returns:
            所有状态的价值估计数组
        """
        return self.V.copy()
```

### 算法二：Q-Learning

```python
"""
Q-Learning 算法

离策略时序差分控制算法，学习最优动作价值函数 Q*

核心公式：
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

算法特点：
    - 离策略学习：行为策略 ≠ 目标策略
    - 收敛到最优策略（满足 GLIE 条件）
    - 可能过度乐观（假设总是执行最优动作）
"""

import numpy as np
from typing import Tuple, Optional


class QLearning:
    """
    Q-Learning 智能体
    
    使用 ε-greedy 策略进行探索，学习最优动作价值函数
    
    Attributes:
        Q (np.ndarray): 动作价值函数表 Q[s, a]
        epsilon (float): 探索率
        alpha (float): 学习率
        gamma (float): 折扣因子
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        """
        初始化 Q-Learning 智能体
        
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            alpha: 学习率（建议 0.1）
            gamma: 折扣因子（建议 0.99）
            epsilon: 探索率
                     - 0.1: 10% 的概率随机探索
                     - 0.01: 1% 的探索（更适合训练后期）
                     - 1.0: 完全随机探索（仅用于测试）
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 初始化 Q 表为 0
        # Q[s, a] 表示在状态 s 采取动作 a 的期望累积回报
        self.Q = np.zeros((n_states, n_actions))
        
    def get_action(self, state: int, training: bool = True) -> int:
        """
        根据 ε-greedy 策略选择动作
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
                     - True: 使用 ε-greedy（有探索）
                     - False: 使用贪婪（纯利用）
                     
        Returns:
            选择的动作索引
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择 Q 值最大的动作
            # 如果有多个最大值，随机选择一个
            q_values = self.Q[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        执行 Q-Learning 更新
        
        核心公式：
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        关键理解：
        - 使用 max_a' Q(s',a') 而不是 Q(s',a')
        - 这表示假设下一步会采取最优动作
        - 这是离策略学习的核心特征
        
        Args:
            state: 当前状态 s_t
            action: 当前动作 a_t
            reward: 即时奖励 r_{t+1}
            next_state: 下一状态 s_{t+1}
            done: 是否 episode 结束
            
        Returns:
            td_error: TD 误差值
        """
        # 获取当前 Q 值
        q_current = self.Q[state, action]
        
        # 计算目标 Q 值
        if done:
            # 终止状态：没有未来回报
            q_target = reward
        else:
            # 非终止状态：使用 max 操作 ⭐ 离策略的关键
            q_next_max = np.max(self.Q[next_state])
            q_target = reward + self.gamma * q_next_max
        
        # 计算 TD 误差
        td_error = q_target - q_current
        
        # 更新 Q 值
        self.Q[state, action] = q_current + self.alpha * td_error
        
        return td_error
    
    def get_q_values(self, state: int) -> np.ndarray:
        """
        获取给定状态的所有动作价值
        
        Args:
            state: 状态索引
            
        Returns:
            该状态下所有动作的 Q 值数组
        """
        return self.Q[state].copy()
    
    def get_policy(self, state: int) -> int:
        """
        获取给定状态的最优动作（贪婪策略）
        
        Args:
            state: 状态索引
            
        Returns:
            最优动作索引
        """
        return np.argmax(self.Q[state])
```

### 算法三：SARSA

```python
"""
SARSA 算法

在策略时序差分控制算法

名称来源：State-Action-Reward-State-Action
每次更新需要五个元素：(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})

核心公式：
    Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]

与 Q-Learning 的唯一区别：
    - 使用 Q(s', a') 而不是 max_a' Q(s', a')
    - a' 是实际采取的下一个动作，不是最优动作
"""

import numpy as np


class SARSA:
    """
    SARSA 智能体
    
    在策略学习，考虑探索行为的影响
    
    Attributes:
        Q (np.ndarray): 动作价值函数表
        epsilon (float): 探索率
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def get_action(self, state: int, training: bool = True) -> int:
        """ε-greedy 动作选择"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,  # ⭐ 与 Q-Learning 的关键区别
        done: bool,
    ) -> float:
        """
        执行 SARSA 更新
        
        核心公式：
        Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        
        关键理解：
        - 使用 Q(s', a')，其中 a' 是实际采取的下一个动作
        - 这表示学习的是当前策略的价值，考虑了探索
        - 这是在策略学习的核心特征
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一状态实际采取的动作 ⭐
            done: 是否 episode 结束
        """
        q_current = self.Q[state, action]
        
        if done:
            q_target = reward
        else:
            # ⭐ 使用实际的下一个动作，不是 max
            q_target = reward + self.gamma * self.Q[next_state, next_action]
        
        td_error = q_target - q_current
        self.Q[state, action] = q_current + self.alpha * td_error
        
        return td_error
```

---

## 🔬 算法对比

### SARSA vs Q-Learning：深入分析

| 维度 | SARSA | Q-Learning |
|------|-------|------------|
| **策略类型** | 在策略 (on-policy) | 离策略 (off-policy) |
| **更新目标** | Q(s', a') 实际动作 | max_a Q(s', a) 最优动作 |
| **探索影响** | 学习时考虑探索 | 学习时忽略探索 |
| **安全性** | 保守，避开风险 | 激进，追求最优 |
| **收敛速度** | 较慢 | 较快 |
| **最终策略** | 考虑探索的最优 | 理论全局最优 |
| **适用场景** | 安全关键任务 | 仿真/游戏环境 |

### 数学对比

**SARSA 更新**：
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Q-Learning 更新**：
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**唯一区别**：$Q(S_{t+1}, A_{t+1})$ vs $\max_a Q(S_{t+1}, a)$

这个看似微小的区别导致了完全不同的学习行为：

1. **SARSA 学习的是当前策略的价值**
   - 如果当前策略是 ε-greedy，SARSA 知道"我有时候会随机探索"
   - 所以它学会了避开高风险区域（如悬崖边缘）
   - 收敛到的策略考虑了探索噪声

2. **Q-Learning 学习的是最优策略的价值**
   - 它假设"我下一步总是走最优动作"
   - 所以它找到了理论上的最短路径
   - 但如果实际执行时有探索噪声，可能会掉下悬崖

### 代码对比

```python
# ============== SARSA ==============
# 1. 选择当前动作
action = agent.get_action(state)

# 2. 执行动作，观察结果
next_state, reward, done = env.step(action)

# 3. 选择下一个动作 ⭐ 关键
next_action = agent.get_action(next_state)

# 4. 更新 Q 值
agent.update(state, action, reward, next_state, next_action, done)


# ============== Q-Learning ==============
# 1. 选择当前动作
action = agent.get_action(state)

# 2. 执行动作，观察结果
next_state, reward, done = env.step(action)

# 3. 不需要选择下一个动作 ⭐ 关键区别
# 更新时直接用 max

# 4. 更新 Q 值
agent.update(state, action, reward, next_state, done)
```

---

## 🧪 动手实验

### 实验一：悬崖行走对比

**任务描述**：

在 Cliff Walking 环境中对比 SARSA 和 Q-Learning 的行为差异。

**环境参数**：
- 网格大小：4 行 × 12 列
- 起点：(3, 0) 左下角
- 终点：(3, 11) 右下角
- 悬崖：(3, 1) 到 (3, 10)
- 掉悬崖奖励：-100
- 每步奖励：-1

**实验步骤**：

1. 分别用 SARSA 和 Q-Learning 训练智能体（各 500 个 episode）
2. 记录每个 episode 的总奖励
3. 绘制学习曲线
4. 分析最终策略

**预期结果**：

| 指标 | SARSA | Q-Learning |
|------|-------|------------|
| 平均总奖励 | ~-15 | ~-13 |
| 平均步数 | ~15 | ~13 |
| 掉悬崖次数 | ~0 | ~5 |
| 路径特点 | 绕远路（安全） | 贴悬崖（最优） |

**思考题**：
1. 为什么 SARSA 的总奖励更低但更安全？
2. 如果 ε=0（不探索），两者行为会趋同吗？
3. 在真实机器人上，你会选择哪个算法？为什么？

### 实验二：学习率敏感性分析

**任务描述**：

研究学习率 α 对 Q-Learning 收敛的影响。

**参数设置**：

| 组别 | α | 预期现象 |
|------|---|----------|
| A | 0.01 | 收敛慢，曲线平滑 |
| B | 0.1 | 收敛速度适中 |
| C | 0.5 | 震荡明显，可能不收敛 |

**实验步骤**：

1. 对每组参数，运行 10 次独立实验（每次 500 episode）
2. 计算每 10 个 episode 的平均奖励（平滑曲线）
3. 绘制三条学习曲线（带标准差阴影）
4. 分析收敛速度和稳定性

**分析指标**：
- 收敛速度：达到 -15 平均奖励所需的 episode 数
- 稳定性：收敛后奖励的标准差
- 最终性能：最后 100 个 episode 的平均奖励

### 实验三：实现 Expected SARSA

**任务描述**：

Expected SARSA 是介于 SARSA 和 Q-Learning 之间的算法，它使用期望值而不是最大值。

**公式**：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \mathbb{E}[Q(S_{t+1}, A_{t+1})] - Q(S_t, A_t)]$$

其中期望值对于 ε-greedy 策略：

$$\mathbb{E}[Q(S_{t+1}, A_{t+1})] = (1-\varepsilon) \max_a Q(S_{t+1}, a) + \frac{\varepsilon}{|A|} \sum_a Q(S_{t+1}, a)$$

**实现提示**：

```python
def expected_sarsa_update(self, state, action, reward, next_state, done):
    q_current = self.Q[state, action]
    
    if done:
        q_target = reward
    else:
        # 计算期望值
        q_next = self.Q[next_state]
        max_q = np.max(q_next)
        mean_q = np.mean(q_next)
        
        # Expected SARSA 的期望计算
        expected_q = (1 - self.epsilon) * max_q + self.epsilon * mean_q
        q_target = reward + self.gamma * expected_q
    
    td_error = q_target - q_current
    self.Q[state, action] = q_current + self.alpha * td_error
```

**实验任务**：
1. 实现 Expected SARSA
2. 在 Cliff Walking 上与 SARSA、Q-Learning 对比
3. 分析性能差异

### 实验四：折扣因子影响分析

**任务描述**：

研究折扣因子 γ 对智能体行为的影响。

**参数设置**：
- γ = 0.5：短视，只关心眼前利益
- γ = 0.9：平衡短期和长期
- γ = 0.99：远视，重视长期回报

**思考题**：
1. γ 太小会导致什么问题？
2. γ 太大会有什么问题？
3. 对于不同长度的任务，如何选择合适的 γ？

---

## ❓ 常见问题

### Q1: 为什么 TD 比蒙特卡洛收敛更快？

**A**: 从方差 - 偏差权衡的角度可以严格分析这个问题。

蒙特卡洛使用完整回报 G_t 作为目标：
- 优点：无偏估计（$\mathbb{E}[G_t] = V(s_t)$）
- 缺点：高方差（因为 G_t 依赖于整个轨迹的随机性）

TD 使用单步回报 $r_{t+1} + \gamma V(s_{t+1})$ 作为目标：
- 优点：低方差（只依赖一步的随机性）
- 缺点：有偏估计（因为 V(s_{t+1}) 本身是估计值）

从学习理论可以证明，在有限样本下，**有偏但低方差的估计往往比无偏但高方差的估计有更小的均方误差**。这就是 TD 通常比 MC 收敛更快的原因。

数学上，均方误差可以分解为：
$$MSE = Bias^2 + Variance$$

TD 通过引入一些偏差，大幅降低了方差，总体 MSE 更小。

### Q2: SARSA 和 Q-Learning 哪个更好？

**A**: 没有绝对答案，取决于应用场景。

**选择 SARSA 的场景**：
- 探索成本高（真实机器人、自动驾驶）
- 安全是关键考虑因素
- 策略执行时有噪声
- 在线学习（边学边用）

**选择 Q-Learning 的场景**：
- 可以在仿真中充分探索（游戏 AI）
- 追求理论最优性能
- 离线学习（学完再部署）
- 环境相对安全

**理论对比**：
- Q-Learning 收敛到最优策略 π*
- SARSA 收敛到考虑探索的最优策略 π_ε

在实际应用中，一个常见的做法是：**训练时用 Q-Learning（收敛快），部署时用 SARSA 风格的策略（更安全）**。

### Q3: 学习率 α 怎么调？

**A**: 从理论和实践两个角度回答。

**理论保证**：
随机近似理论要求学习率满足：
- $\sum_t \alpha_t = \infty$（学习率之和发散）
- $\sum_t \alpha_t^2 < \infty$（学习率平方和收敛）

满足这两个条件的典型选择是：$\alpha_t = \frac{1}{t}$

**实践建议**：
- 固定学习率：α = 0.1 是不错的起点
- 学习率衰减：$\alpha_t = \frac{\alpha_0}{1 + kt}$
- Adam 等自适应优化器（深度 RL 中常用）

**调试技巧**：
1. 从 α = 0.1 开始
2. 如果学习曲线震荡 → 减小 α
3. 如果学习太慢 → 增大 α
4. 观察 TD 误差的变化趋势（应该逐渐减小）

### Q4: ε 应该怎么设置？

**A**: ε 控制探索与利用的权衡。

**常见策略**：

1. **固定 ε**：
   - ε = 0.1：10% 探索（常用）
   - ε = 0.01：1% 探索（训练后期）

2. **ε 衰减**：
   $$\varepsilon_t = \max(\varepsilon_{min}, \varepsilon_0 \cdot decay^t)$$
   - 初期高探索（ε = 1.0）
   - 后期低探索（ε = 0.01）
   - decay 通常取 0.995

3. **GLIE 策略**（Greedy in the Limit with Infinite Exploration）：
   - $\lim_{t \to \infty} \varepsilon_t = 0$
   - $\sum_t \varepsilon_t = \infty$
   - 理论收敛保证

**实践建议**：
- 简单任务：固定 ε = 0.1
- 复杂任务：ε 从 1.0 衰减到 0.01
- 安全关键：始终保持一定探索（ε ≥ 0.01）

### Q5: 什么时候用表格方法，什么时候用函数近似？

**A**: 这取决于状态空间的大小。

**表格方法（本章内容）**：
- 适用：状态数 < 10^6
- 优点：简单，收敛有保证
- 缺点：无法处理大状态空间

**函数近似（第 5 章内容）**：
- 适用：状态数很大或连续
- 方法：用神经网络等近似 Q 函数
- 优点：可以处理高维输入（如图像）
- 缺点：训练不稳定，收敛性难保证

**经验法则**：
- 网格世界、简单游戏 → 表格方法
- Atari 游戏、机器人控制 → 函数近似（DQN 等）
- LLM 后训练 → 函数近似（PPO、DPO 等）

---

## 📚 延伸阅读

### 理论推导

1. **Sutton & Barto, Chapter 6**: Temporal-Difference Learning
   - 最权威的 TD 学习教材
   - 包含完整收敛性证明

2. **Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3), 279-292.**
   - Q-Learning 的原始论文
   - 包含收敛性证明

3. **Singh, S. P., et al. (2000). Convergence results for single-step on-policy reinforcement-learning algorithms. Machine learning, 38(3), 287-308.**
   - SARSA 的收敛性分析

### 代码文件

| 文件 | 内容 | 行数 |
|------|------|------|
| `td_prediction.py` | TD(0) 预测算法 | ~100 |
| `q_learning.py` | Q-Learning 算法 | ~150 |
| `sarsa.py` | SARSA 算法 | ~150 |
| `expected_sarsa.py` | Expected SARSA | ~150 |
| `games/cliff_walking.py` | 悬崖行走环境 | ~200 |

### 进阶主题

- **n 步 TD 方法**：结合多步回报和自举
- **TD(λ)**：指数加权平均所有 n 步回报
- **Double Q-Learning**：解决 Q-Learning 的过估计问题
- **Dueling DQN**：分离状态价值和动作优势

---

## ✅ 本章检查清单

学完本章后，你应该能够：

**概念理解**：
- [ ] 解释 TD 学习的核心思想（自举 + 采样）
- [ ] 区分在策略和离策略学习
- [ ] 理解 SARSA 和 Q-Learning 的本质区别
- [ ] 解释为什么 TD 通常比 MC 收敛更快

**数学推导**：
- [ ] 推导 TD(0) 的更新公式
- [ ] 推导 Q-Learning 的贝尔曼最优方程
- [ ] 解释 TD 误差的数学含义

**代码实现**：
- [ ] 独立实现 TD(0) 预测
- [ ] 独立实现 Q-Learning
- [ ] 独立实现 SARSA
- [ ] 实现 Expected SARSA

**实验分析**：
- [ ] 运行 Cliff Walking 对比实验
- [ ] 分析学习率对收敛的影响
- [ ] 比较不同算法的性能差异

**应用判断**：
- [ ] 根据场景选择合适的算法
- [ ] 合理设置超参数（α, γ, ε）
- [ ] 诊断训练中的问题（震荡、不收敛等）

---

## 📝 课后测验（10 分钟）

### 基础题（必答）

**1. TD 误差的公式是什么？直观含义是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

公式：δ = R + γV(s') - V(s)

直观含义：**预期和实际的差距**
- V(s)：之前以为的价值
- R + γV(s')：实际得到 + 新估计
- δ > 0：比预期好，增加 V(s)
- δ < 0：比预期差，减少 V(s)
</details>

---

**2. SARSA 和 Q-Learning 的核心区别是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

| 维度 | SARSA | Q-Learning |
|------|-------|------------|
| 策略类型 | 在策略 | 离策略 |
| 更新目标 | Q(s', a') 实际动作 | max_a Q(s', a) 最优动作 |
| 安全性 | 保守（考虑探索） | 激进（假设最优） |
| 悬崖行走 | 绕远路 | 贴悬崖 |
</details>

---

**3. 为什么 TD 通常比 MC 收敛更快？**

<details>
<summary>点击查看答案</summary>

**答案**：

方差 - 偏差权衡：
- MC：无偏，高方差（完整轨迹的随机性）
- TD：有偏，低方差（只依赖一步随机性）

有限样本下，有偏但低方差通常 MSE 更小。
</details>

---

### 进阶题（选答）

**4. 解释 Expected SARSA 的优势**

<details>
<summary>点击查看答案</summary>

**答案**：

Expected SARSA 用期望值代替最大值：
- 方差比 Q-Learning 小
- 性能通常优于 SARSA 和 Q-Learning
- 计算成本稍高（需要求和）
</details>

---

**5. 在什么场景下应该选择 SARSA 而不是 Q-Learning？**

<details>
<summary>点击查看答案</summary>

**答案**：

安全关键场景：
- 真实机器人控制
- 自动驾驶
- 任何探索成本高的场景

Q-Learning 假设总是最优，但实际执行时有探索噪声，可能导致危险。
</details>

---

### 编程题（实践）

**6. 实现 Expected SARSA，并与 SARSA、Q-Learning 对比**

<details>
<summary>查看提示</summary>

**提示**：
1. Expected SARSA 的 target：
   E[Q] = (1-ε)*max Q + (ε/|A|)*Σ Q
2. 在 Cliff Walking 上对比
3. 分析性能差异
</details>

---

## 🚀 下一章预告

**第 5 章：函数近似与深度 Q 网络**

当状态空间很大时，表格方法不再适用。本章将介绍：
- 用神经网络近似 Q 函数
- DQN 的核心创新（经验回放、目标网络）
- 从表格 Q-Learning 到深度 RL 的演进

**预告实验**：训练一个能玩 Atari 打砖块游戏的 AI！🕹️

---

*最后更新：2026-04-22*  
*作者：Hermes <neko_yukirin@qq.com>*
