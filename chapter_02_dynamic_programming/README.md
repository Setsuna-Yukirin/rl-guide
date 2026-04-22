# 第 2 章：动态规划与蒙特卡洛树搜索

> **当 MDP 已知时** —— 用动态规划求解最优策略，用 MCTS 处理大规模状态空间

---

## ⏱️ TL;DR（30 秒速览）

**核心问题**：MDP 已知时，如何**高效求解最优策略**？

**两种算法**：
- **策略迭代**：评估策略 → 改进策略 → 重复（收敛快，每步慢）
- **价值迭代**：直接迭代贝尔曼最优方程（收敛慢，每步快）

**核心公式**：
- 策略评估：$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V_k(s')]$
- 价值迭代：$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V_k(s')]$

**MCTS（蒙特卡洛树搜索）**：
- 选择 → 扩展 → 模拟 → 回溯
- UCT 公式：平衡探索和利用
- AlphaGo 的核心技术

**关键洞察**：
- DP 需要完整模型（P 和 R 已知）
- 策略迭代通常收敛更快（迭代次数少）
- MCTS 只搜索有希望的状态（聚焦）

**学完这章你能**：
- ✅ 实现策略迭代和价值迭代
- ✅ 理解 MCTS 的四步流程
- ✅ 用 DP 求解网格世界问题
- ✅ 理解 AlphaGo 的规划机制

**代码实现**：策略迭代、价值迭代、MCTS

**常见误区**：
- ❌ DP 是强化学习 → ✅ DP 是规划方法，RL 是学习方法
- ❌ 价值迭代不如策略迭代 → ✅ 取决于问题规模
- ❌ MCTS 是随机搜索 → ✅ MCTS 是智能聚焦搜索

---

## 🎯 本章要解决什么问题

在第 1 章中，我们学习了 MDP 的基本概念和贝尔曼方程。但有一个关键问题没有回答：**知道了贝尔曼方程，怎么实际求解最优策略？**

这就是动态规划（Dynamic Programming, DP）要解决的问题。

### 动态规划的核心假设

动态规划有一个**强假设**：**MDP 完全已知**。也就是说，你知道：
- 所有状态转移概率 $P(s'|s, a)$
- 所有奖励函数 $R(s, a, s')$

这个假设在实际强化学习中往往不成立（否则就不需要"学习"了）。那为什么还要学动态规划？

**原因一：理论价值**。动态规划是理解后续所有 RL 算法的基础。蒙特卡洛方法、时序差分学习、甚至 PPO/DPO，都可以看作是在 MDP 未知时对动态规划的近似。

**原因二：规划问题**。在某些场景中，MDP 确实是已知的。比如：
- 游戏 AI（游戏规则完全已知）
- 路径规划（地图和转移确定）
- 资源调度（系统模型已知）

这时动态规划可以直接使用。

**原因三：内循环规划**。在现代 RL（如 AlphaGo、AlphaZero）中，动态规划用作"内循环"——智能体有一个 learned model，然后在这个模型上用 DP 进行规划。

### 动态规划 vs 强化学习

理解这个区别至关重要：

| 维度 | 动态规划 | 强化学习 |
|------|----------|----------|
| MDP 已知 | 是 | 否 |
| 需要交互 | 否 | 是 |
| 计算方式 | 解析/迭代 | 采样学习 |
| 典型算法 | 策略迭代、价值迭代 | Q-Learning、PPO、DPO |
| 适用场景 | 规划问题 | 学习问题 |

**关键洞察**：强化学习 = 动态规划 + 采样近似

学完本章后，你将能够：
- 理解策略迭代和价值迭代的原理与区别
- 掌握蒙特卡洛树搜索（MCTS）的核心思想
- 能够用动态规划求解中等规模的 MDP
- 理解 AlphaGo 等系统的规划机制
- 为后续学习蒙特卡洛方法（第 3 章）和时序差分（第 4 章）打下基础

---

## 📖 场景描述：从网格寻路到 AlphaGo

### 场景一：网格世界寻路

想象一个机器人在 5×5 的网格中移动：

```
S . . . .
. X . . .
. . X . .
. . . X .
. . . . G
```

- **S**：起点 (0, 0)
- **G**：终点 (4, 4)，奖励 +10
- **X**：陷阱，奖励 -10
- **每步**：奖励 -1（鼓励尽快到达）
- **动作**：上/下/左/右
- **转移**：90% 按预期移动，10% 随机滑向相邻方向

**问题**：机器人应该如何选择移动方向，让**期望总奖励最大**？

这是一个典型的**规划问题**——MDP 完全已知，只需要计算最优策略。

用动态规划可以这样求解：
1. 初始化所有状态价值为 0
2. 用贝尔曼最优化方程迭代更新
3. 收敛后，从价值函数提取最优策略

**直觉理解**：价值迭代就像"水波扩散"——终点的 +10 奖励像水波一样向外传播，每一步传播都打个折（折扣因子 γ）。最终，每个状态的价值表示"从这里到终点还能获得多少期望奖励"。

### 场景二：仓库机器人调度

一个更复杂的场景：Amazon 仓库中有多个机器人需要：
- 从货架取货
- 送到打包站
- 避开其他机器人和障碍
- 最小化总配送时间

**状态空间**：所有机器人的位置 × 货物位置 × 打包站队列
**动作空间**：每个机器人的移动方向
**奖励**：-1 每步（时间成本），+100 成功配送

这个问题用动态规划求解会遇到**维数灾难**——状态空间是指数级的。

**解决方案**：
1. 分解问题（每个机器人独立规划）
2. 用启发式剪枝
3. 或者用 MCTS（只搜索有希望的路径）

### 场景三：AlphaGo 的决策机制

2016 年，AlphaGo 击败人类围棋冠军。它的核心创新之一就是**用 MCTS 进行规划**。

**围棋的状态空间**：$10^{170}$ 种可能局面
- 比宇宙原子数还多
- 无法用表格表示所有状态
- 传统动态规划完全不可行

**AlphaGo 的解决方案**：
1. 用神经网络评估局面（价值网络）
2. 用神经网络选择候选着法（策略网络）
3. 用 MCTS 在候选着法中搜索最优解

**MCTS 的工作方式**：
- 从当前局面出发，模拟大量对局
- 用神经网络指导搜索方向（而不是随机）
- 根据模拟结果更新着法价值
- 选择胜率最高的着法

关键洞察：**MCTS 是一种"聚焦"的动态规划**——它不计算所有状态的价值，只计算搜索树中状态的价值。这使得它能处理超大规模的状态空间。

---

## 🧠 核心概念详解

### 概念一：动态规划的本质

**直觉理解**：

动态规划这个名字听起来很复杂，但核心思想非常简单：

> **把复杂问题分解成子问题，记住子问题的答案，避免重复计算。**

"动态"指的是问题有多阶段决策，"规划"指的是预先计算最优策略。

用网格寻路举例：
- 复杂问题：从起点到终点的最优路径
- 子问题：从每个格子到终点的最优路径
- 记住答案：每个格子的价值函数
- 避免重复：用迭代法，每次更新都用上次的结果

**形式化定义**：

动态规划适用于满足两个性质的问题：

**1. 最优子结构**：最优解包含子问题的最优解。

在 MDP 中，这体现为贝尔曼方程：最优价值函数 $V^*$ 满足递归关系。

**2. 重叠子问题**：子问题会被重复求解。

在 MDP 中，从不同状态出发可能到达同一个状态，所以价值函数可以复用。

**为什么这样设计**：

动态规划的设计基于一个深刻的洞察：**多阶段决策问题可以逆向求解**。

考虑从终点倒推：
- 终点的价值是确定的（+10）
- 终点前一步的价值 = 即时奖励 + γ × 终点价值
- 终点前两步的价值 = 即时奖励 + γ × 前一步价值
- ...

这个"倒推"思想就是贝尔曼方程的本质。

### 概念二：策略评估

**直觉理解**：

策略评估回答的问题是：**如果我一直遵循某个策略，长期来看能获得多少回报？**

想象你有一个固定的决策规则（比如"总是选择看起来最近的路"），你想知道这个规则长期来看有多好。策略评估就是计算这个"长期期望回报"。

**形式化定义**：

给定策略 $\pi$，策略评估计算价值函数 $V_\pi$：

$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_\pi(s')]$$

这是一个线性方程组，可以用迭代法求解：

$$V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_k(s')]$$

**为什么这样设计**：

策略评估的迭代法基于**压缩映射定理**。

定义贝尔曼期望算子 $T^\pi$：

$$(T^\pi V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]$$

可以证明，当 $\gamma < 1$ 时，$T^\pi$ 是压缩映射：

$$||T^\pi V_1 - T^\pi V_2||_\infty \leq \gamma ||V_1 - V_2||_\infty$$

由巴拿赫不动点定理，迭代 $V_{k+1} = T^\pi V_k$ 必然收敛到唯一不动点 $V_\pi$。

**收敛速度**：

收敛速度取决于 $\gamma$：
- $\gamma$ 接近 0：快速收敛（未来不重要）
- $\gamma$ 接近 1：慢速收敛（未来很重要）

误差上界：

$$||V_k - V_\pi||_\infty \leq \frac{\gamma^k}{1-\gamma} ||V_1 - V_0||_\infty$$

### 概念三：策略改进

**直觉理解**：

策略改进回答的问题是：**给定当前价值函数，如何改进策略？**

答案很直观：**在每个状态下，选择能带来最大期望回报的动作**。

这就是"贪婪"策略——只看眼前最优。

**形式化定义**：

给定价值函数 $V$，贪婪策略定义为：

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]$$

用动作价值函数表示更简洁：

$$\pi'(s) = \arg\max_a Q(s, a)$$

其中：

$$Q(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]$$

**为什么这样设计**：

策略改进的正确性由**策略改进定理**保证：

**定理**：如果 $\pi'$ 是相对于 $V_\pi$ 的贪婪策略，则 $V_{\pi'} \geq V_\pi$。

**证明思路**：

1. 由贪婪策略定义：$Q_\pi(s, \pi'(s)) \geq V_\pi(s)$
2. 展开 $V_{\pi'}$ 的贝尔曼方程
3. 用归纳法证明 $V_{\pi'} \geq V_\pi$

关键洞察：**局部改进导致全局改进**。这是动态规划能收敛到最优解的根本原因。

### 概念四：策略迭代

**直觉理解**：

策略迭代把策略评估和策略改进交替进行：

```
评估当前策略 → 改进策略 → 评估新策略 → 再改进 → ... → 收敛到最优
```

就像爬山：
- 策略评估：计算当前位置的高度
- 策略改进：往高处走一步
- 重复直到到达山顶

**形式化定义**：

策略迭代算法：

```
初始化任意策略 π_0
重复 k = 0, 1, 2, ...:
    1. 策略评估：计算 V_{π_k}
    2. 策略改进：π_{k+1} = greedy(V_{π_k})
    3. 如果 π_{k+1} = π_k，返回 π_k
```

**为什么这样设计**：

策略迭代的收敛性由以下定理保证：

**定理**：对于有限 MDP，策略迭代在有限步内收敛到最优策略。

**证明思路**：
1. 每次策略改进都严格改进（或保持不变）
2. 有限 MDP 的策略数量有限（$|A|^{|S|}$ 个）
3. 价值函数有上界
4. 因此必然在有限步内收敛

**收敛速度**：

策略迭代的收敛速度是**线性**的，但通常很快：
- 实践中通常只需几次到几十次迭代
- 每次迭代需要完整的策略评估（可能很慢）

### 概念五：价值迭代

**直觉理解**：

价值迭代是策略迭代的简化版本。它观察到：**策略评估不需要完全收敛**，只要更新一步就可以改进策略。

就像爬山时，不需要精确知道当前位置的高度，只要知道往哪个方向走是上坡就行。

**形式化定义**：

价值迭代算法：

```
初始化 V_0(s) = 0
重复 k = 0, 1, 2, ...:
    对于每个状态 s:
        V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_k(s')]
    如果 max_s |V_{k+1}(s) - V_k(s)| < ε，返回
```

**为什么这样设计**：

价值迭代的核心观察：

策略评估的贝尔曼方程：
$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R + \gamma V_k(s')]$$

价值迭代的更新：
$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s, a) [R + \gamma V_k(s')]$$

区别只是把"对动作求平均"改成了"对动作取 max"。

这个改动有两个效果：
1. 不需要显式维护策略
2. 每次更新都同时改进价值和策略

**收敛性**：

价值迭代也收敛到 $V^*$，收敛速度是**线性**的：

$$||V_{k+1} - V^*||_\infty \leq \gamma ||V_k - V^*||_\infty$$

### 概念六：蒙特卡洛树搜索（MCTS）

**直觉理解**：

MCTS 是一种**基于采样的规划算法**。它的核心思想是：

> **与其计算所有状态的价值，不如只计算有希望的状态。**

想象你在下围棋：
- 传统动态规划：计算所有可能局面的价值（不可能，$10^{170}$ 种）
- MCTS：只计算当前局面下"看起来合理"的着法

**MCTS 的四步流程**：

1. **选择（Selection）**：从根节点出发，用 UCT 公式选择子节点
2. **扩展（Expansion）**：添加一个新的子节点
3. **模拟（Simulation）**：从新节点随机模拟到游戏结束
4. **回溯（Backpropagation）**：将结果回传更新路径上的节点

**UCT 公式**：

$$UCT(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

其中：
- $Q(s, a)$：动作价值的平均值（利用项）
- $N(s)$：状态访问次数
- $N(s, a)$：动作选择次数
- $c$：探索常数（通常取 $\sqrt{2}$）

**为什么这样设计**：

UCT 公式的设计基于**多臂老虎机**理论。

第一项 $Q(s, a)$ 鼓励选择已知好的动作（利用）。
第二项鼓励选择探索少的动作（探索）。

这个平衡由探索常数 $c$ 控制：
- $c$ 大：更多探索
- $c$ 小：更多利用

理论证明，UCT 公式能保证**渐近收敛到最优策略**（当模拟次数趋于无穷时）。

---

（第一部分完成，待续...）
## 📐 公式推导

### 推导一：策略评估的收敛性

**问题设定**：

我们想要证明策略评估的迭代法收敛到 $V_\pi$。

**贝尔曼期望算子**：

定义算子 $T^\pi$：

$$(T^\pi V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]$$

**压缩映射证明**：

要证明 $T^\pi$ 是压缩映射，需要证明：

$$||T^\pi V_1 - T^\pi V_2||_\infty \leq \gamma ||V_1 - V_2||_\infty$$

**证明**：

$$\begin{aligned}
|(T^\pi V_1)(s) - (T^\pi V_2)(s)| 
&= \left| \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) \gamma [V_1(s') - V_2(s')] \right| \\
&\leq \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) \gamma |V_1(s') - V_2(s')| \\
&\leq \gamma \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) ||V_1 - V_2||_\infty \\
&= \gamma ||V_1 - V_2||_\infty \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) \\
&= \gamma ||V_1 - V_2||_\infty
\end{aligned}$$

最后一步因为 $\sum_a \pi(a|s) \sum_{s'} P(s'|s, a) = 1$。

因此：

$$||T^\pi V_1 - T^\pi V_2||_\infty \leq \gamma ||V_1 - V_2||_\infty$$

**收敛性**：

由巴拿赫不动点定理：
1. $T^\pi$ 是压缩映射（压缩系数 $\gamma < 1$）
2. 完备度量空间中的压缩映射有唯一不动点
3. 迭代 $V_{k+1} = T^\pi V_k$ 收敛到不动点 $V_\pi$

**误差上界**：

$$||V_k - V_\pi||_\infty \leq \frac{\gamma^k}{1-\gamma} ||V_1 - V_0||_\infty$$

### 推导二：策略改进定理

**定理陈述**：

给定策略 $\pi$ 和贪婪策略 $\pi'$：

$$\pi'(s) = \arg\max_a Q_\pi(s, a)$$

则 $V_{\pi'} \geq V_\pi$。

**证明**：

第一步，写出贪婪策略的定义：

$$Q_\pi(s, \pi'(s)) \geq V_\pi(s) = Q_\pi(s, \pi(s))$$

第二步，考虑 $V_{\pi'}$ 的贝尔曼方程：

$$V_{\pi'}(s) = \sum_a \pi'(a|s) Q_{\pi'}(s, a)$$

第三步，用归纳法。假设对于所有 $s'$，$V_{\pi'}(s') \geq V_\pi(s')$。

第四步，比较 $Q$ 函数：

$$\begin{aligned}
Q_{\pi'}(s, a) &= \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_{\pi'}(s')] \\
&\geq \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V_\pi(s')] \\
&= Q_\pi(s, a)
\end{aligned}$$

第五步，结合：

$$V_{\pi'}(s) = Q_{\pi'}(s, \pi'(s)) \geq Q_\pi(s, \pi'(s)) \geq V_\pi(s)$$

**关键理解**：

策略改进定理的核心是**单调性**——每次改进都不会变差。

### 推导三：价值迭代的收敛性

**问题设定**：

我们想要证明价值迭代收敛到 $V^*$。

**贝尔曼最优化算子**：

定义算子 $T^*$：

$$(T^* V)(s) = \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]$$

**压缩映射证明**：

类似策略评估，可以证明 $T^*$ 也是压缩映射：

$$||T^* V_1 - T^* V_2||_\infty \leq \gamma ||V_1 - V_2||_\infty$$

**证明**：

$$\begin{aligned}
|(T^* V_1)(s) - (T^* V_2)(s)| 
&= \left| \max_a \sum_{s'} P(s'|s, a) \gamma [V_1(s') - V_2(s')] \right| \\
&\leq \max_a \sum_{s'} P(s'|s, a) \gamma |V_1(s') - V_2(s')| \\
&\leq \gamma ||V_1 - V_2||_\infty
\end{aligned}$$

**收敛性**：

由压缩映射定理，$T^*$ 有唯一不动点 $V^*$，且迭代 $V_{k+1} = T^* V_k$ 收敛到 $V^*$。

**收敛速度**：

$$||V_k - V^*||_\infty \leq \gamma^k ||V_0 - V^*||_\infty$$

### 推导四：MCTS 的 UCT 公式

**问题设定**：

MCTS 需要平衡探索和利用。UCT 公式如何实现这个平衡？

**多臂老虎机类比**：

MCTS 中的每个状态可以看作一个多臂老虎机：
- 每个动作是一个"臂"
- 选择动作获得奖励（模拟结果）
- 目标：最大化总奖励

**UCB1 公式**：

对于 K 臂老虎机，UCB1 公式为：

$$UCB1(a) = \bar{X}_a + \sqrt{\frac{2 \ln n}{n_a}}$$

其中：
- $\bar{X}_a$：臂 a 的平均奖励
- $n$：总选择次数
- $n_a$：臂 a 的选择次数

**UCT 公式**：

将 UCB1 应用到树搜索：

$$UCT(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

**为什么这样设计**：

第一项 $Q(s, a)$ 是利用——选择已知好的动作。
第二项是探索——选择访问少的动作。

**探索常数 c 的选择**：

理论分析表明，当 $c = \sqrt{2}$ 时，UCB1 的 regret 上界最优。

但在实践中，c 需要调参：
- $c = 0$：纯贪婪（只利用）
- $c = 1$：平衡
- $c = 2$：更多探索

---

## 💻 算法实现

### 实现一：策略迭代算法

```python
"""
策略迭代算法

通过交替进行策略评估和策略改进，找到最优策略
"""

import numpy as np
from typing import Tuple, Optional


def policy_evaluation(
    mdp,
    policy: np.ndarray,
    V: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = False,
) -> np.ndarray:
    """
    策略评估：计算给定策略 π 的价值函数 V_π
    
    使用迭代法求解贝尔曼期望方程：
        V_{k+1}(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_k(s')]
    
    收敛条件：max_s |V_{k+1}(s) - V_k(s)| < tol
    
    算法复杂度：
        - 时间：O(k * |S|^2 * |A|)，其中 k 是迭代次数
        - 空间：O(|S|)
    
    Args:
        mdp: 表格型 MDP，包含以下属性：
             - nS: 状态数
             - nA: 动作数
             - P: 转移字典 P[s][a] = [(prob, next_s, reward, done), ...]
             - gamma: 折扣因子
        policy: 策略 π(a|s)，形状 (nS, nA)
                policy[s, a] 表示在状态 s 选择动作 a 的概率
        V: 初始价值函数（可选），形状 (nS,)
        tol: 收敛容差
        max_iter: 最大迭代次数
        verbose: 是否打印迭代信息
        
    Returns:
        V: 收敛后的价值函数，形状 (nS,)
        
    Example:
        >>> mdp = TabularMDP(n_states=16, n_actions=4)
        >>> policy = np.ones((16, 4)) * 0.25  # 随机策略
        >>> V = policy_evaluation(mdp, policy)
    """
    # 初始化价值函数
    if V is None:
        V = np.zeros(mdp.nS)
    
    for iteration in range(max_iter):
        # 保存旧价值函数（用于检查收敛）
        V_old = V.copy()
        
        # 贝尔曼期望方程更新
        # 对每个状态 s，计算在新策略下的期望价值
        for s in range(mdp.nS):
            v_s = 0.0
            
            # 对每个动作求和
            for a in range(mdp.nA):
                # 对每个可能的转移求和
                for prob, next_s, reward, done in mdp.P[s][a]:
                    # 贝尔曼方程：即时奖励 + 折扣后的未来价值
                    future_value = 0.0 if done else V[next_s]
                    v_s += policy[s, a] * prob * (reward + mdp.gamma * future_value)
            
            V[s] = v_s
        
        # 检查收敛：最大变化量小于容差
        delta = np.max(np.abs(V - V_old))
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"策略评估迭代 {iteration + 1}: delta = {delta:.6f}")
        
        if delta < tol:
            if verbose:
                print(f"策略评估在 {iteration + 1} 次迭代后收敛")
            break
    
    return V


def policy_improvement(
    mdp,
    V: np.ndarray,
) -> np.ndarray:
    """
    策略改进：从价值函数提取贪婪策略
    
    对于每个状态，选择能最大化 Q 值的动作：
        π(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
    
    Args:
        mdp: 表格型 MDP
        V: 价值函数，形状 (nS,)
        
    Returns:
        policy: 确定性策略（one-hot 编码），形状 (nS, nA)
                policy[s, a] = 1 表示在状态 s 选择动作 a
    """
    if V.shape != (mdp.nS,):
        raise ValueError(f"V 的形状 {V.shape} 不匹配，期望 ({mdp.nS},)")
    
    # 初始化策略为 one-hot 编码
    policy = np.zeros((mdp.nS, mdp.nA))
    
    for s in range(mdp.nS):
        q_values = []
        
        # 计算每个动作的 Q 值
        for a in range(mdp.nA):
            q_sa = 0.0
            
            for prob, next_s, reward, done in mdp.P[s][a]:
                future_value = 0.0 if done else V[next_s]
                q_sa += prob * (reward + mdp.gamma * future_value)
            
            q_values.append(q_sa)
        
        # 选择 Q 值最大的动作（贪婪）
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return policy


def policy_iteration(
    mdp,
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    策略迭代算法
    
    算法流程：
    1. 初始化任意策略 π_0
    2. 策略评估：计算 V_{π_k}
    3. 策略改进：π_{k+1} = greedy(V_{π_k})
    4. 如果 π_{k+1} = π_k，返回；否则 k ← k+1，回到步骤 2
    
    收敛性：
        - 对于有限 MDP，策略迭代在有限步内收敛到最优策略
        - 每次迭代都严格改进（或保持不变）
    
    算法复杂度：
        - 时间：O(k * |S|^2 * |A|)，其中 k 是策略迭代次数
        - 空间：O(|S|)
    
    Args:
        mdp: 表格型 MDP
        tol: 策略评估的收敛容差
        max_iter: 最大策略迭代次数
        verbose: 是否打印迭代信息
        
    Returns:
        V: 最优价值函数，形状 (nS,)
        policy: 最优策略（one-hot 编码），形状 (nS, nA)
        n_iters: 实际迭代次数
        
    Example:
        >>> mdp = GridWorldMDP(size=5)
        >>> V, policy, n_iters = policy_iteration(mdp, verbose=True)
        >>> # 提取确定性策略
        >>> actions = np.argmax(policy, axis=1)
    """
    # 初始化随机策略
    policy = np.ones((mdp.nS, mdp.nA)) / mdp.nA
    
    for iteration in range(max_iter):
        if verbose:
            print(f"\n策略迭代 {iteration + 1}/{max_iter}")
        
        # 策略评估
        V = policy_evaluation(mdp, policy, tol=tol, verbose=verbose)
        
        # 策略改进
        new_policy = policy_improvement(mdp, V)
        
        # 检查策略是否收敛
        if np.array_equal(policy, new_policy):
            if verbose:
                print(f"策略迭代在 {iteration + 1} 次迭代后收敛")
            break
        
        policy = new_policy
    
    return V, policy, iteration + 1
```

### 实现二：价值迭代算法

```python
"""
价值迭代算法

通过迭代贝尔曼最优化方程，直接找到最优价值函数
"""

import numpy as np
from typing import Tuple, Optional


def value_iteration(
    mdp,
    V: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    价值迭代算法
    
    算法流程：
    1. 初始化 V_0(s) = 0
    2. 贝尔曼最优化更新：
       V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_k(s')]
    3. 如果 max_s |V_{k+1}(s) - V_k(s)| < tol，返回
    4. 否则 k ← k+1，回到步骤 2
    
    与策略迭代的区别：
        - 策略迭代：显式维护策略，交替评估和改进
        - 价值迭代：只维护价值函数，隐式改进策略
        - 价值迭代每次迭代更快，但可能需要更多迭代
    
    收敛性：
        - 收敛到最优价值函数 V*
        - 收敛速度：O(γ^k)
    
    算法复杂度：
        - 时间：O(k * |S|^2 * |A|)，其中 k 是迭代次数
        - 空间：O(|S|)
    
    Args:
        mdp: 表格型 MDP
        V: 初始价值函数（可选），形状 (nS,)
        tol: 收敛容差
        max_iter: 最大迭代次数
        verbose: 是否打印迭代信息
        
    Returns:
        V: 最优价值函数，形状 (nS,)
        policy: 最优策略（one-hot 编码），形状 (nS, nA)
        n_iters: 实际迭代次数
        
    Example:
        >>> mdp = GridWorldMDP(size=5)
        >>> V, policy, n_iters = value_iteration(mdp, verbose=True)
    """
    # 初始化价值函数
    if V is None:
        V = np.zeros(mdp.nS)
    
    for iteration in range(max_iter):
        # 保存旧价值函数（用于检查收敛）
        V_old = V.copy()
        
        # 贝尔曼最优化方程更新
        # 对每个状态 s，选择最优动作
        for s in range(mdp.nS):
            q_values = []
            
            # 计算每个动作的 Q 值
            for a in range(mdp.nA):
                q_sa = 0.0
                
                for prob, next_s, reward, done in mdp.P[s][a]:
                    future_value = 0.0 if done else V[next_s]
                    q_sa += prob * (reward + mdp.gamma * future_value)
                
                q_values.append(q_sa)
            
            # 取最大 Q 值 ⭐ 关键：最优化而不是期望
            V[s] = max(q_values)
        
        # 检查收敛
        delta = np.max(np.abs(V - V_old))
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"价值迭代 {iteration + 1}: delta = {delta:.6f}")
        
        if delta < tol:
            if verbose:
                print(f"价值迭代在 {iteration + 1} 次迭代后收敛")
            break
    
    # 从最优价值函数提取策略
    policy = extract_greedy_policy(mdp, V)
    
    return V, policy, iteration + 1


def extract_greedy_policy(
    mdp,
    V: np.ndarray,
) -> np.ndarray:
    """
    从价值函数提取贪婪策略
    
    Args:
        mdp: 表格型 MDP
        V: 价值函数，形状 (nS,)
        
    Returns:
        policy: 确定性策略（one-hot 编码），形状 (nS, nA)
    """
    policy = np.zeros((mdp.nS, mdp.nA))
    
    for s in range(mdp.nS):
        q_values = []
        
        for a in range(mdp.nA):
            q_sa = 0.0
            
            for prob, next_s, reward, done in mdp.P[s][a]:
                future_value = 0.0 if done else V[next_s]
                q_sa += prob * (reward + mdp.gamma * future_value)
            
            q_values.append(q_sa)
        
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return policy
```

### 实现三：MCTS 算法

```python
"""
蒙特卡洛树搜索 (MCTS)

用于大规模 MDP 的规划算法，通过聚焦搜索找到近似最优解
"""

import numpy as np
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """
    MCTS 搜索树节点
    
    Attributes:
        state: 状态表示
        parent: 父节点
        children: 子节点字典 {action: MCTSNode}
        N: 访问次数
        Q: 累计奖励
        available_actions: 可用动作列表
    """
    state: any
    parent: Optional['MCTSNode'] = None
    children: Dict = field(default_factory=dict)
    N: int = 0  # 访问次数
    Q: float = 0.0  # 累计奖励
    available_actions: List = field(default_factory=list)
    
    def is_fully_expanded(self) -> bool:
        """检查是否所有动作都已探索"""
        return len(self.children) == len(self.available_actions)
    
    def is_terminal(self) -> bool:
        """检查是否是终止状态"""
        return len(self.available_actions) == 0
    
    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        """
        选择 UCT 值最大的子节点
        
        UCT 公式：
            UCT(s, a) = Q(s, a) / N(s, a) + c * sqrt(ln N(s) / N(s, a))
        
        Args:
            c_param: 探索参数（默认 √2 ≈ 1.414）
        """
        choices_weights = []
        
        for action, child in self.children.items():
            # 避免除零
            if child.N == 0:
                uct_value = float('inf')
            else:
                # UCT 公式
                exploitation = child.Q / child.N
                exploration = c_param * math.sqrt(math.log(self.N) / child.N)
                uct_value = exploitation + exploration
            
            choices_weights.append(uct_value)
        
        # 选择 UCT 值最大的子节点
        best_action = list(self.children.keys())[np.argmax(choices_weights)]
        return self.children[best_action]


class MCTS:
    """
    蒙特卡洛树搜索
    
    核心流程：
    1. 选择：从根节点出发，用 UCT 公式选择子节点
    2. 扩展：添加一个新的子节点
    3. 模拟：从新节点随机模拟到终止
    4. 回溯：将结果回传更新路径上的节点
    """
    
    def __init__(self, c_param: float = 1.4, n_iterations: int = 1000):
        """
        初始化 MCTS
        
        Args:
            c_param: UCT 公式中的探索参数
                     - 大：更多探索
                     - 小：更多利用
                     - 理论最优：√2 ≈ 1.414
            n_iterations: MCTS 迭代次数
        """
        self.c_param = c_param
        self.n_iterations = n_iterations
    
    def search(self, initial_state, env) -> MCTSNode:
        """
        执行 MCTS 搜索
        
        Args:
            initial_state: 初始状态
            env: 环境（需要提供 step 和 get_available_actions 方法）
        
        Returns:
            root: 搜索树的根节点
        """
        # 创建根节点
        root = MCTSNode(
            state=initial_state,
            available_actions=env.get_available_actions(initial_state)
        )
        
        for _ in range(self.n_iterations):
            node = root
            
            # 1. 选择：一直选择到叶子节点
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.c_param)
            
            # 2. 扩展：如果不是终止状态，添加一个子节点
            if not node.is_terminal():
                action = self._select_untried_action(node)
                next_state = env.step(node.state, action)
                
                child = MCTSNode(
                    state=next_state,
                    parent=node,
                    available_actions=env.get_available_actions(next_state)
                )
                node.children[action] = child
                node = child
            
            # 3. 模拟：随机模拟到终止
            reward = self._simulate(node, env)
            
            # 4. 回溯：更新路径上的节点
            while node is not None:
                node.N += 1
                node.Q += reward
                node = node.parent
        
        return root
    
    def _select_untried_action(self, node: MCTSNode):
        """选择一个未探索的动作"""
        tried_actions = set(node.children.keys())
        untried = [a for a in node.available_actions if a not in tried_actions]
        return np.random.choice(untried)
    
    def _simulate(self, node: MCTSNode, env) -> float:
        """
        随机模拟到终止
        
        Returns:
            reward: 模拟得到的总奖励
        """
        state = node.state
        total_reward = 0.0
        
        while not env.is_terminal(state):
            action = np.random.choice(env.get_available_actions(state))
            state, reward = env.step(state, action)
            total_reward += reward
        
        return total_reward
    
    def get_best_action(self, root: MCTSNode):
        """
        从搜索树中选择最佳动作
        
        选择访问次数最多的动作（最稳健的选择）
        
        Args:
            root: 搜索树根节点
        
        Returns:
            best_action: 最佳动作
        """
        if not root.children:
            return np.random.choice(root.available_actions)
        
        # 选择访问次数最多的子节点
        visit_counts = [child.N for child in root.children.values()]
        best_action = list(root.children.keys())[np.argmax(visit_counts)]
        
        return best_action
```

（第二部分完成，待续...）
## 🔬 算法对比

### 策略迭代 vs 价值迭代

| 维度 | 策略迭代 | 价值迭代 |
|------|----------|----------|
| **核心思想** | 交替评估和改进策略 | 直接迭代贝尔曼最优化方程 |
| **维护变量** | 策略 π + 价值 V | 仅价值 V |
| **每次迭代** | 策略评估（多步）+ 策略改进（一步） | 贝尔曼最优更新（一步） |
| **迭代次数** | 少（通常 5-20 次） | 多（可能 100+ 次） |
| **每次迭代成本** | 高（需要完整策略评估） | 低（一步更新） |
| **总计算时间** | 中小规模问题更快 | 大规模问题可能更快 |
| **内存需求** | O(|S| + |S|×|A|) | O(|S|) |
| **收敛保证** | 有限步收敛到最优 | 渐近收敛到最优 |
| **实现复杂度** | 中等 | 简单 |

### 数学对比

**策略迭代**：
```
重复直到收敛：
    # 策略评估（多次迭代直到收敛）
    重复：
        V_{k+1}(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R + γV_k(s')]
    直到：收敛
    
    # 策略改进（一步）
    π'(s) ← argmax_a Σ_{s'} P(s'|s,a) [R + γV(s')]
```

**价值迭代**：
```
重复直到收敛：
    # 贝尔曼最优更新（一步）
    V_{k+1}(s) ← max_a Σ_{s'} P(s'|s,a) [R + γV_k(s')]
```

### MCTS vs 动态规划

| 维度 | 动态规划 | MCTS |
|------|----------|------|
| **状态空间** | 需要枚举所有状态 | 只搜索访问的状态 |
| **适用规模** | 中小规模（|S| < 10^6） | 大规模（|S| > 10^100） |
| **计算方式** | 系统备份 | 随机采样 |
| **最优性** | 保证最优 | 渐近最优 |
| ** anytime 性质** | 否（需要完整计算） | 是（可随时停止） |
| **并行化** | 困难 | 容易 |
| **典型应用** | 网格世界、资源调度 | 围棋、国际象棋 |

### 选择建议

**选择策略迭代如果**：
- 状态空间较小（< 1000）
- 需要精确的最优策略
- 策略评估可以快速收敛
- 内存充足

**选择价值迭代如果**：
- 状态空间中等（1000 - 10^6）
- 实现简单优先
- 可以接受近似最优

**选择 MCTS 如果**：
- 状态空间巨大（> 10^6）
- 有好的启发式（如神经网络）
- 需要 anytime 算法
- 可以并行计算

---

## 🧪 动手实验

### 实验一：策略迭代 vs 价值迭代收敛对比

**任务描述**：

在同一个 MDP 上对比策略迭代和价值迭代的收敛行为。

**环境设置**：

- 5×5 网格世界
- 起点 (0, 0)，终点 (4, 4)
- 每步奖励 -1
- 终点奖励 +10
- 折扣因子 γ = 0.99

**实验步骤**：

1. 创建 GridWorld MDP
2. 运行策略迭代，记录每次迭代的 max|V_{k+1} - V_k|
3. 运行价值迭代，记录同样指标
4. 绘制收敛曲线对比

**分析指标**：
- 收敛所需迭代次数
- 每次迭代的 CPU 时间
- 最终策略是否相同
- 总计算时间

**预期结果**：
- 策略迭代次数少但每次慢
- 价值迭代次数多但每次快
- 两者收敛到相同最优策略

**思考题**：
1. 为什么策略迭代收敛更快（迭代次数少）？
2. 在什么情况下价值迭代可能总时间更短？
3. 如果 γ = 0.9 而不是 0.99，收敛速度如何变化？

### 实验二：折扣因子对策略的影响

**任务描述**：

研究折扣因子 γ 如何影响最优策略。

**参数设置**：

| 组别 | γ | 预期行为 |
|------|---|----------|
| A | 0.5 | 短视，只看眼前 |
| B | 0.9 | 平衡 |
| C | 0.99 | 远视，重视长期 |
| D | 1.0 | 完全不折扣（仅终止问题） |

**实验步骤**：

1. 在同一个 MDP 上用不同 γ 值运行价值迭代
2. 提取每个 γ 对应的最优策略
3. 可视化策略差异
4. 比较价值函数分布

**预期现象**：
- γ 小：选择即时奖励高的动作
- γ 大：可能绕远路换取长期高奖励
- γ = 1：只适用于有终止状态的问题

### 实验三：MCTS 在井字棋上的应用

**任务描述**：

实现 MCTS 并让它学会玩井字棋（Tic-Tac-Toe）。

**井字棋 MDP**：
- 状态：3×3 棋盘（每个格子空/X/O）
- 动作：在空位落子
- 奖励：赢 +1，输 -1，平局 0
- 状态数：约 5000 种合法局面

**实现步骤**：

1. 实现井字棋环境（get_available_actions, step, is_terminal）
2. 实现 MCTS 算法
3. 让 MCTS 自我对弈
4. 分析胜率

**分析指标**：
- MCTS 迭代次数对棋力的影响
- 不同 c_param 值的对比
- 与极小化极大算法的对比

**扩展挑战**：
- 实现 UCT 公式的变体
- 添加领域知识（如优先选择中心位置）
- 与神经网络结合（类似 AlphaGo）

### 实验四：异步价值迭代

**任务描述**：

实现异步价值迭代（Asynchronous Value Iteration），对比与标准价值迭代的差异。

**标准价值迭代**：
```python
# 同步更新：先计算所有新值，再统一更新
V_new = V.copy()
for s in range(nS):
    V_new[s] = max_a Σ P(s'|s,a) [R + γV(s')]
V = V_new
```

**异步价值迭代**：
```python
# 异步更新：就地更新，立即使用新值
for s in range(nS):
    V[s] = max_a Σ P(s'|s,a) [R + γV(s')]  # 注意这里用的是最新的 V
```

**实验步骤**：

1. 实现异步价值迭代
2. 在同一个 MDP 上对比两种方法
3. 分析收敛速度差异

**预期结果**：
- 异步更新通常收敛更快
- 因为新信息立即传播

---

## ❓ 常见问题

### Q1: 动态规划和强化学习的根本区别是什么？

**A**: 核心区别在于**MDP 是否已知**。

**动态规划**：
- 假设：MDP 完全已知（知道 P 和 R）
- 方法：解析/迭代求解贝尔曼方程
- 典型算法：策略迭代、价值迭代
- 适用：规划问题（如游戏 AI、路径规划）

**强化学习**：
- 假设：MDP 未知
- 方法：通过与环境交互采样学习
- 典型算法：Q-Learning、SARSA、PPO
- 适用：学习问题（如机器人控制、推荐系统）

**关键洞察**：强化学习 = 动态规划 + 采样近似

蒙特卡洛方法用采样回报近似期望。
时序差分用自举 + 采样近似贝尔曼方程。

### Q2: 策略迭代和价值迭代哪个更好？

**A**: 没有绝对答案，取决于问题特点。

**策略迭代优势**：
- 迭代次数少（通常几次到几十次）
- 每次迭代有明确的策略改进
- 适合中小规模问题

**策略迭代劣势**：
- 每次迭代需要完整的策略评估（可能很慢）
- 需要存储策略和价值函数

**价值迭代优势**：
- 实现简单（就一行更新）
- 内存需求低
- 适合中等规模问题

**价值迭代劣势**：
- 迭代次数多
- 收敛速度慢（但每次迭代快）

**经验法则**：
- |S| < 1000：策略迭代
- 1000 < |S| < 10^6：价值迭代
- |S| > 10^6：MCTS 或近似方法

### Q3: MCTS 为什么能在围棋上成功？

**A**: MCTS 成功的关键是**聚焦搜索**。

围棋的状态空间是 $10^{170}$，传统动态规划完全不可行。MCTS 的成功在于：

**1. 只搜索相关状态**：
- 不计算所有局面的价值
- 只搜索从当前局面出发的可能着法
- 大幅减少计算量

**2. 智能探索**：
- UCT 公式平衡探索和利用
- 优先搜索有希望的着法
- 避免浪费计算在明显差的着法上

**3. 与神经网络结合**（AlphaGo）：
- 策略网络指导搜索方向
- 价值网络评估局面
- MCTS 在神经网络基础上精细化

**4. anytime 性质**：
- 可以随时停止搜索
- 搜索时间越长，棋力越强
- 适合实际比赛的时间限制

### Q4: 为什么价值迭代用 max 而不是期望？

**A**: 因为价值迭代的目标是找到**最优策略**。

回顾贝尔曼方程：

**贝尔曼期望方程**（给定策略 π）：
$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R + \gamma V_\pi(s')]$$

**贝尔曼最优化方程**（最优策略 π*）：
$$V^*(s) = \max_a \sum_{s'} P(s'|s, a) [R + \gamma V^*(s')]$$

关键区别：
- 期望方程：对动作求平均（因为策略可能随机）
- 最优化方程：对动作取 max（因为最优策略总是选最好的）

价值迭代直接用 max，隐含假设是"总是选择最优动作"。这正是我们想要的！

### Q5: MCTS 的迭代次数怎么选择？

**A**: 取决于时间约束和精度要求。

**理论分析**：
- MCTS 渐近收敛到最优策略
- 收敛速度是 $O(1/\sqrt{N})$，其中 N 是迭代次数
- 要让误差减半，需要 4 倍迭代次数

**实践建议**：

**有时间限制**（如比赛）：
- 用满所有时间
- 每秒能跑多少次迭代 × 可用时间

**无时间限制**：
- 从 1000 次开始
- 观察价值估计的变化
- 如果变化小于阈值，停止

**精度要求**：
- 粗略估计：100-1000 次
- 中等精度：1000-10000 次
- 高精度：10000+ 次

**调参技巧**：
- 绘制"迭代次数 vs 价值估计"曲线
- 找到收益递减点
- 在该点附近选择

---

## 📚 延伸阅读

### 核心教材

1. **Sutton & Barto, Chapter 4**: Dynamic Programming
   - https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
   - 最权威的 RL 教材，DP 章节讲得非常清晰

2. **Bertsekas, D. P. (2017). Dynamic Programming and Optimal Control**
   - DP 领域的经典教材
   - 理论深入，适合想深入理解的同学

### MCTS 专题

1. **Browne et al. (2012). A Survey of Monte Carlo Tree Search Methods**
   - https://ieeexplore.ieee.org/document/5800888
   - MCTS 的综述论文，覆盖所有变体

2. **Silver et al. (2016). Mastering the game of Go with deep neural networks**
   - https://www.nature.com/articles/nature16961
   - AlphaGo 的原始论文，MCTS + 神经网络的经典应用

### 代码资源

| 文件 | 内容 | 行数 |
|------|------|------|
| `policy_iteration.py` | 策略迭代算法 | ~150 |
| `value_iteration.py` | 价值迭代算法 | ~120 |
| `mcts.py` | MCTS 算法 | ~200 |
| `games/gridworld_nav.py` | 网格寻路环境 | ~150 |
| `games/warehouse_robot.py` | 仓库机器人环境 | ~180 |

---

## ✅ 本章检查清单

学完本章后，你应该能够：

**概念理解**：
- [ ] 解释动态规划的核心假设
- [ ] 区分策略迭代和价值迭代
- [ ] 解释 MCTS 的四步流程
- [ ] 解释 UCT 公式的含义

**数学推导**：
- [ ] 推导策略评估的收敛性
- [ ] 推导策略改进定理
- [ ] 推导价值迭代的收敛性
- [ ] 解释 UCT 公式的理论基础

**代码实现**：
- [ ] 实现策略迭代算法
- [ ] 实现价值迭代算法
- [ ] 实现 MCTS 算法
- [ ] 实现网格世界环境

**实验分析**：
- [ ] 对比策略迭代和价值迭代
- [ ] 分析折扣因子的影响
- [ ] 实现 MCTS 玩井字棋
- [ ] 实现异步价值迭代

**应用判断**：
- [ ] 根据问题规模选择算法
- [ ] 合理设置超参数
- [ ] 诊断收敛问题

---

## 📝 课后测验（10 分钟）

### 基础题（必答）

**1. 策略迭代和价值迭代的核心区别是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

| 维度 | 策略迭代 | 价值迭代 |
|------|----------|----------|
| 维护变量 | 策略 π + 价值 V | 仅价值 V |
| 每次迭代 | 策略评估（多步）+ 改进（一步） | 贝尔曼最优更新（一步） |
| 迭代次数 | 少（5-20 次） | 多（100+ 次） |
| 每次成本 | 高 | 低 |
</details>

---

**2. MCTS 的四步流程是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：
1. **选择（Selection）**：用 UCT 公式选择子节点
2. **扩展（Expansion）**：添加新子节点
3. **模拟（Simulation）**：随机模拟到终止
4. **回溯（Backpropagation）**：更新路径节点

UCT 公式：$UCT(s,a) = Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}$
</details>

---

**3. 为什么动态规划需要 MDP 完全已知？**

<details>
<summary>点击查看答案</summary>

**答案**：

DP 的更新规则需要：
- 转移概率 P(s'|s,a)：计算期望
- 奖励函数 R(s,a,s')：计算即时回报

如果未知，无法计算贝尔曼方程的右边。

对比：RL 方法（如 Q-Learning）用采样近似期望，不需要知道 P 和 R。
</details>

---

### 进阶题（选答）

**4. 证明策略迭代在有限步内收敛到最优策略**

<details>
<summary>点击查看答案</summary>

**证明思路**：
1. 每次策略改进都严格改进（或保持不变）
2. 有限 MDP 的策略数量有限（|A|^|S| 个）
3. 价值函数有上界
4. 因此必然在有限步内收敛
</details>

---

**5. UCT 公式中探索常数 c 如何选择？**

<details>
<summary>点击查看答案</summary>

**答案**：

理论分析：c = √2 时 regret 上界最优。

实践建议：
- c = 0：纯贪婪（只利用）
- c = 1：平衡
- c = 2：更多探索

需要根据具体问题调参。
</details>

---

### 编程题（实践）

**6. 实现异步价值迭代，并与标准价值迭代对比**

<details>
<summary>查看提示</summary>

**提示**：
1. 标准 VI：先计算所有新值，再统一更新
2. 异步 VI：就地更新，立即使用新值
3. 比较收敛速度

异步更新通常更快，因为新信息立即传播。
</details>

---

## 🚀 下一章预告

**第 3 章：蒙特卡洛方法**

当 MDP 未知时，如何学习最优策略？本章将介绍：
- 蒙特卡洛预测（用采样估计价值）
- 蒙特卡洛控制（用采样优化策略）
- On-Policy vs Off-Policy 学习
- 应用：21 点游戏、老虎机

**预告实验**：训练一个能赢 21 点的 AI！🃏

---

*最后更新：2026-04-22*  
*作者：Hermes <neko_yukirin@qq.com>*
