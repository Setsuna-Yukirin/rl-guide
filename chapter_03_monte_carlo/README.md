# 第 3 章：蒙特卡洛方法 (Monte Carlo Methods)

> **当 MDP 未知时** —— 用采样估计价值，从经验中学习最优策略

---

## 🎯 本章要解决什么问题

在第 2 章中，我们学习了动态规划。但动态规划有一个**强假设**：MDP 完全已知（知道所有转移概率 P 和奖励 R）。这个假设在实际应用中往往不成立。

想象这些场景：
- **训练机器人走路**：你不知道电机的精确动力学模型
- **推荐系统**：你不知道用户看到推荐后会如何反应
- **玩 Atari 游戏**：你不知道游戏的内部机制（只知道像素和分数）

这些场景有一个共同点：**MDP 未知，但可以与环境交互**。

这就是蒙特卡洛方法（Monte Carlo Methods）要解决的问题。

### 蒙特卡洛方法的核心洞察

蒙特卡洛方法的核心洞察可以用一句话概括：

> **与其用模型计算期望，不如用采样近似期望。**

这个洞察看似简单，但它带来了革命性的变化：
- 不再需要知道转移概率 P
- 不再需要知道奖励函数 R
- 只需要能与环境交互，采样轨迹

**关键区别**：

动态规划计算期望：
$$V(s) = \mathbb{E}[R + \gamma V(s') | s] = \sum_{s'} P(s'|s, \pi(s)) [R + \gamma V(s')]$$

蒙特卡洛用采样近似：
$$V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i$$

其中 $G_i$ 是第 i 次访问状态 s 后获得的实际回报。

### 蒙特卡洛 vs 动态规划

理解这个区别至关重要：

| 维度 | 动态规划 | 蒙特卡洛 |
|------|----------|----------|
| MDP 已知 | 是（需要 P 和 R） | 否（只需要采样） |
| 更新时机 | 每步更新（自举） | episode 结束更新 |
| 偏差 - 方差 | 有偏、低方差 | 无偏、高方差 |
| 适用任务 | 任何 MDP | 仅回合制任务 |
| 计算方式 | 解析/迭代 | 采样平均 |
| 典型应用 | 规划问题 | 学习问题 |

**关键洞察**：蒙特卡洛是强化学习从"规划"走向"学习"的关键一步。

### 为什么叫"蒙特卡洛"？

这个名字来源于摩纳哥的蒙特卡洛赌场。20 世纪 40 年代，冯·诺依曼等人在洛斯阿拉莫斯国家实验室研究核武器时，需要求解复杂的中子扩散方程。他们发现，用随机采样来近似解比解析方法更有效。因为涉及随机性，就用赌场名字"蒙特卡洛"来命名。

如今，蒙特卡洛方法广泛应用于：
- 强化学习（本章内容）
- 贝叶斯统计（MCMC 采样）
- 计算机图形学（光线追踪）
- 金融工程（期权定价）
- 物理学（粒子模拟）

学完本章后，你将能够：
- 理解蒙特卡洛方法的核心思想和数学原理
- 掌握第一访问 MC 和每次访问 MC 的区别
- 掌握蒙特卡洛控制算法（用 MC 找最优策略）
- 理解探索 - 利用权衡（ε-greedy 策略）
- 理解离策略学习（重要性采样）
- 为后续学习时序差分（第 4 章）打下基础

---

## 📖 场景描述：从赌场到 21 点游戏

### 场景一：赌场老虎机

想象你走进赌场，看到一排老虎机。每台老虎机：
- 拉一次手臂需要 1 美元
- 可能吐出 0 美元、2 美元、10 美元、甚至 100 美元
- 每台机器的 payout 概率不同

**问题**：你应该如何选择机器，让**总收益最大**？

这是一个经典的**多臂老虎机**（Multi-Armed Bandit）问题。

**策略一：纯探索**
- 每台机器都试 100 次
- 统计平均每台机器的回报
- 选择平均回报最高的机器

问题：浪费太多钱在差的机器上。

**策略二：纯利用**
- 第一台机器试一次
- 如果赢了，继续玩这台
- 如果输了，换下一台

问题：可能错过更好的机器。

**策略三：探索 - 利用权衡**
- 大部分时间玩当前最好的机器（利用）
- 偶尔试试其他机器（探索）
- 随着时间推移，逐渐减少探索

这就是蒙特卡洛方法的核心挑战之一：**探索与利用的权衡**。

### 场景二：21 点游戏（Blackjack）

21 点是蒙特卡洛方法的经典应用场景。

**游戏规则**：
- 目标：牌的点数接近 21 点，但不能超过
- 玩家先行动：可以选择"要牌"（Hit）或"停牌"（Stand）
- 玩家爆牌（>21 点）立即输
- 玩家停牌后，庄家行动（固定规则：16 点以下必须拿牌）
- 比较点数，大者赢

**状态空间**：
- 玩家当前点数（12-21 点，小于 12 点必须拿牌）
- 庄家明牌（2-11 点）
- 玩家是否有可用 A（可算作 1 或 11）

总共约 200 种状态。

**问题**：如何找到最优策略？

**用动态规划？**
- 需要知道发牌概率（可以计算）
- 需要知道庄家策略（已知）
- 理论上可行，但计算复杂

**用蒙特卡洛？**
- 不需要知道发牌概率
- 只需要玩很多局
- 统计每个状态下各种行动的回报
- 选择回报最高的行动

**蒙特卡洛的学习过程**：

```
第 1 局：状态 (16, 庄家 6, 无 A) → 要牌 → 爆牌 → 回报 -1
第 2 局：状态 (16, 庄家 6, 无 A) → 停牌 → 庄家爆牌 → 回报 +1
第 3 局：状态 (16, 庄家 6, 无 A) → 停牌 → 庄家 20 点 → 回报 -1
...
第 N 局后：
  Q((16, 6, 无 A), Hit) = 平均回报 = -0.8
  Q((16, 6, 无 A), Stand) = 平均回报 = -0.2
  → 最优行动：停牌
```

**关键洞察**：蒙特卡洛方法通过**大量采样**来估计价值，而不是通过**解析计算**。

### 场景三：从自我对弈中学习

想象你要训练一个 AI 玩井字棋（Tic-Tac-Toe）。

**方法一：动态规划**
- 枚举所有棋局状态（约 5000 种）
- 用极小化极大算法计算每种状态的价值
- 需要知道游戏规则（如何转移）

**方法二：蒙特卡洛**
- AI 自己和自己下棋（自我对弈）
- 记录每局的结果（赢/输/平）
- 统计每个状态下各种行动的胜率
- 选择胜率最高的行动

**蒙特卡洛的优势**：
- 不需要知道游戏规则
- 只需要能模拟游戏
- 可以处理更复杂的游戏（如围棋）

AlphaGo 的早期版本就用到了蒙特卡洛树搜索！

---

## 🧠 核心概念详解

### 概念一：用采样近似期望

**直觉理解**：

蒙特卡洛方法的核心思想非常直观：

> **如果你想知道某个随机变量的期望值，就采样很多次，然后取平均。**

比如你想知道一个骰子的期望点数：
- 解析方法：$\mathbb{E}[X] = \frac{1+2+3+4+5+6}{6} = 3.5$
- 蒙特卡洛方法：掷 1000 次骰子，计算平均值

当问题复杂时（如强化学习中的期望回报），解析计算可能很困难，但采样相对容易。

**形式化定义**：

在强化学习中，我们想要估计状态价值：

$$V_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

其中 $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$ 是折扣回报。

蒙特卡洛方法用采样平均近似这个期望：

$$V_\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

其中 $N(s)$ 是访问状态 s 的次数，$G_t^{(i)}$ 是第 i 次访问后的回报。

**为什么这样设计**：

蒙特卡洛估计有以下性质：

**1. 无偏性**：

$$\mathbb{E}\left[\frac{1}{N} \sum_{i=1}^{N} G^{(i)}\right] = \mathbb{E}[G] = V_\pi(s)$$

当采样次数趋于无穷时，估计值收敛到真实期望。

**2. 大数定律**：

根据大数定律：

$$\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^{N} G^{(i)} = V_\pi(s) \quad \text{(几乎必然)}$$

**3. 收敛速度**：

蒙特卡洛估计的误差以 $O(1/\sqrt{N})$ 的速度收敛。

这意味着要让误差减半，需要 4 倍的采样次数。

### 概念二：第一访问 vs 每次访问

**直觉理解**：

在一个 episode 中，同一个状态可能被访问多次。如何处理这些重复访问？

**第一访问 MC（First-visit MC）**：
- 只考虑第一次访问该状态的回报
- 忽略后续访问

**每次访问 MC（Every-visit MC）**：
- 考虑每次访问的回报
- 所有访问都用于估计

**例子**：

考虑一个 episode：
```
状态序列：A → B → C → B → D → 终止
回报：    -1  -1  -1  -2  -1
```

对于状态 B：
- 第一访问 MC：只用第一次访问的回报 $G = -1 + (-1) + (-2) + (-1) = -5$
- 每次访问 MC：用两次访问的回报
  - 第一次：$G_1 = -5$
  - 第二次：$G_2 = -2 + (-1) = -3$
  - 平均：$(-5 + (-3)) / 2 = -4$

**形式化定义**：

**第一访问 MC**：
$$V(s) = \frac{1}{N_1(s)} \sum_{i=1}^{N_1(s)} G_t^{(i)} \quad \text{其中 t 是第一次访问 s 的时间}$$

**每次访问 MC**：
$$V(s) = \frac{1}{N(s)} \sum_{\text{所有访问}} G_t$$

**为什么这样设计**：

两种方法都收敛到 $V_\pi(s)$，但有以下区别：

**第一访问 MC**：
- 理论分析更简单
- 在理论上更常用
- 实际中可能浪费数据（忽略后续访问）

**每次访问 MC**：
- 利用更多数据
- 在同一个 episode 中多次访问同一状态时更有效
- 理论上分析稍复杂

**实践建议**：
- 大多数情况下，两者表现相近
- 每次访问 MC 实现更简单（不需要检查是否第一次访问）
- 推荐使用每次访问 MC

### 概念三：蒙特卡洛控制

**直觉理解**：

蒙特卡洛预测解决了**策略评估**问题（给定策略 π，估计 $V_\pi$）。

蒙特卡洛控制要解决**策略优化**问题（找到最优策略 π*）。

核心思想与动态规划相同：**策略迭代**。

```
策略评估（用 MC） → 策略改进 → 新策略 → 再评估 → 再改进 → ... → 最优策略
```

**关键挑战**：

在动态规划中，策略改进需要计算：

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]$$

这需要知道转移概率 P 和奖励 R。但蒙特卡洛方法假设这些未知！

**解决方案**：学习动作价值函数 Q(s, a) 而不是状态价值函数 V(s)。

$$Q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

有了 Q 函数，策略改进变得简单：

$$\pi'(s) = \arg\max_a Q(s, a)$$

不需要知道 P 和 R！

**形式化定义**：

蒙特卡洛控制算法：

```
初始化 Q(s, a) = 0, π(s) = 随机
重复：
    1. 用策略 π 生成一个 episode
    2. 对于 episode 中的每个 (s, a)：
       - 计算回报 G
       - 更新 Q(s, a) ← Q(s, a) + α(G - Q(s, a))
    3. 策略改进：π(s) = argmax_a Q(s, a)
直到：收敛
```

**为什么这样设计**：

蒙特卡洛控制的关键是**用采样代替模型**。

动态规划的策略改进：
$$\pi'(s) = \arg\max_a \mathbb{E}[R + \gamma V(s') | s, a]$$

蒙特卡洛的策略改进：
$$\pi'(s) = \arg\max_a Q(s, a) \approx \arg\max_a \frac{1}{N(s,a)} \sum G^{(i)}$$

### 概念四：探索 - 利用权衡

**直觉理解**：

蒙特卡洛控制面临一个根本性挑战：

> **如果总是选择当前认为最好的动作（利用），可能错过更好的动作（需要探索）。**

想象你在餐厅点菜：
- **纯利用**：每次都点同样的菜（可能错过更好吃的）
- **纯探索**：每次都点新菜（可能点到很难吃的）
- **权衡**：大部分点喜欢的，偶尔尝试新菜

在强化学习中，这个权衡通过 **ε-greedy 策略** 实现。

**ε-greedy 策略**：

$$\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{如果 } a = \arg\max_{a'} Q(s, a') \text{（最优动作）} \\
\frac{\epsilon}{|A|} & \text{其他动作}
\end{cases}$$

其中：
- $\epsilon \in [0, 1]$ 是探索率
- $|A|$ 是动作数量

**行为解释**：
- 以概率 $1 - \epsilon + \epsilon/|A|$ 选择最优动作
- 以概率 $\epsilon/|A|$ 随机选择每个动作

**例子**：
- $\epsilon = 0.1$，3 个动作：
  - 最优动作：90% + 3.3% = 93.3%
  - 其他动作：各 3.3%

**为什么这样设计**：

ε-greedy 策略的设计基于以下考虑：

**1. 保证探索**：
- 所有动作都有非零概率被选择
- 随着采样次数增加，每个动作都会被充分探索

**2. 渐进最优**：
- 随着学习进行，可以逐渐减小 ε
- 最终收敛到贪婪策略（纯利用）

**3. 实现简单**：
- 只需要一个参数 ε
- 容易实现和调试

**GLIE 条件**：

理论上，为了保证收敛到最优策略，需要满足 GLIE 条件：

**GLIE** (Greedy in the Limit with Infinite Exploration)：
1. **无限探索**：$\lim_{k \to \infty} N_k(s, a) = \infty$（所有状态 - 动作对访问无限次）
2. **贪婪极限**：$\lim_{k \to \infty} \pi_k(a|s) = 1$ 如果 $a = \arg\max_{a'} Q(s, a')$

**实现方式**：
$$\epsilon_k = \frac{1}{k}$$

其中 k 是 episode 编号。

### 概念五：离策略学习（重要性采样）

**直觉理解**：

到目前为止，我们讨论的都是**在策略**（on-policy）学习：
- 用策略 π 生成数据
- 学习策略 π 的价值

**离策略**（off-policy）学习要解决不同的问题：
- 用策略 b（行为策略）生成数据
- 学习策略 π（目标策略）的价值

**为什么要离策略学习？**

1. **从历史数据学习**：
   - 行为策略：过去使用的策略
   - 目标策略：想要评估的新策略

2. **从人类专家学习**：
   - 行为策略：人类专家的策略
   - 目标策略：想要学习的 AI 策略

3. **探索更充分**：
   - 行为策略：高探索率（如 ε=0.5）
   - 目标策略：低探索率（如 ε=0.01）

**核心挑战**：

两个策略产生的数据分布不同，如何修正这个偏差？

**解决方案：重要性采样（Importance Sampling）**。

**重要性采样比率**：

$$\rho = \frac{P(\text{episode} | \pi)}{P(\text{episode} | b)}$$

这个比率衡量：同一个 episode，在目标策略下出现的概率与在行为策略下出现的概率之比。

**加权重要性采样**：

$$V(s) = \frac{\sum_{i=1}^{N} \rho_i G_i}{\sum_{i=1}^{N} \rho_i}$$

**为什么这样设计**：

重要性采样的核心思想是**加权平均**。

如果一个 episode 在目标策略下更可能出现（ρ > 1），就给它的回报更高权重。
如果一个 episode 在目标策略下不太可能出现（ρ < 1），就给它的回报更低权重。

这样，加权平均就能近似目标策略的期望回报。

---

（第一部分完成，待续...）
## 📐 公式推导

### 推导一：蒙特卡洛估计的收敛性

**问题设定**：

我们想要证明蒙特卡洛估计收敛到真实价值 $V_\pi(s)$。

**定义回顾**：

蒙特卡洛估计：
$$\hat{V}_n(s) = \frac{1}{n} \sum_{i=1}^{n} G^{(i)}$$

其中 $G^{(i)}$ 是第 i 次访问状态 s 后的回报。

**大数定律**：

根据强大数定律（Strong Law of Large Numbers）：

如果 $G^{(1)}, G^{(2)}, ...$ 是独立同分布的随机变量，且 $\mathbb{E}[|G|] < \infty$，则：

$$\lim_{n \to \infty} \hat{V}_n(s) = \mathbb{E}[G] = V_\pi(s) \quad \text{(几乎必然)}$$

**证明思路**：

第一步，注意到每次访问状态 s 后的回报 $G^{(i)}$ 是独立同分布的：
- 独立性：不同 episode 的回报独立
- 同分布：都遵循策略 π 下的回报分布

第二步，验证期望有限：
$$\mathbb{E}[|G|] = \mathbb{E}\left[\left|\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right|\right] \leq \sum_{t=0}^{\infty} \gamma^t \mathbb{E}[|R_{t+1}|] \leq \frac{R_{\max}}{1-\gamma} < \infty$$

第三步，应用大数定律：
$$\lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} G^{(i)} = V_\pi(s)$$

**收敛速度**：

根据中心极限定理，估计误差的分布渐近正态：

$$\sqrt{n}(\hat{V}_n(s) - V_\pi(s)) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

其中 $\sigma^2 = \text{Var}(G)$ 是回报的方差。

这意味着误差以 $O(1/\sqrt{n})$ 的速度收敛。

### 推导二：第一访问 vs 每次访问的等价性

**问题设定**：

我们想要证明第一访问 MC 和每次访问 MC 都收敛到 $V_\pi(s)$。

**第一访问 MC**：

$$\hat{V}_n^{\text{first}}(s) = \frac{1}{n} \sum_{i=1}^{n} G^{(i)}_{\text{first}}$$

其中 $G^{(i)}_{\text{first}}$ 是第 i 个 episode 中第一次访问 s 的回报。

**每次访问 MC**：

$$\hat{V}_n^{\text{every}}(s) = \frac{1}{\sum_{i=1}^{n} K_i} \sum_{i=1}^{n} \sum_{j=1}^{K_i} G^{(i,j)}$$

其中 $K_i$ 是第 i 个 episode 中访问 s 的次数，$G^{(i,j)}$ 是第 j 次访问的回报。

**等价性证明**：

第一步，注意到两种估计都是无偏的：
$$\mathbb{E}[\hat{V}_n^{\text{first}}(s)] = \mathbb{E}[\hat{V}_n^{\text{every}}(s)] = V_\pi(s)$$

第二步，两种估计都满足大数定律的条件（独立同分布、期望有限）。

第三步，因此两者都收敛到 $V_\pi(s)$。

**方差比较**：

每次访问 MC 的方差通常更小，因为它利用了更多数据。

但在同一个 episode 中，多次访问的回报是相关的（不是独立的），所以理论分析更复杂。

### 推导三：ε-greedy 策略的改进定理

**问题设定**：

我们想要证明，用 ε-greedy 策略进行策略改进，策略会变好。

**策略改进定理（ε-greedy 版本）**：

给定策略 $\pi$ 和 ε-greedy 策略 $\pi'$（相对于 $Q_\pi$）：

$$\pi'(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{如果 } a = \arg\max_{a'} Q_\pi(s, a') \\
\frac{\epsilon}{|A|} & \text{其他}
\end{cases}$$

则 $V_{\pi'} \geq V_\pi$。

**证明**：

第一步，写出 ε-greedy 策略的价值：

$$V_{\pi'}(s) = \sum_a \pi'(a|s) Q_\pi(s, a)$$

第二步，展开：

$$V_{\pi'}(s) = (1 - \epsilon + \frac{\epsilon}{|A|}) \max_a Q_\pi(s, a) + \frac{\epsilon}{|A|} \sum_{a \neq a^*} Q_\pi(s, a)$$

第三步，与 $V_\pi$ 比较：

$$V_\pi(s) = \sum_a \pi(a|s) Q_\pi(s, a)$$

第四步，如果 $\pi$ 也是 ε-greedy（或更差），则：

$$V_{\pi'}(s) \geq V_\pi(s)$$

**关键理解**：

ε-greedy 策略改进定理保证了**单调改进**。

即使有探索，策略也不会变差。

### 推导四：重要性采样的无偏性

**问题设定**：

我们想要证明，加权重要性采样估计是无偏的。

**定义回顾**：

重要性采样比率：
$$\rho(\tau) = \frac{P(\tau | \pi)}{P(\tau | b)} = \prod_{t=0}^{T} \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$$

其中 $\tau = (s_0, a_0, s_1, a_1, ..., s_T)$ 是一个 episode。

**普通重要性采样**：

$$\hat{V}_{\text{IS}}(s) = \frac{1}{n} \sum_{i=1}^{n} \rho(\tau_i) G(\tau_i)$$

**无偏性证明**：

$$\begin{aligned}
\mathbb{E}_b[\hat{V}_{\text{IS}}(s)] 
&= \mathbb{E}_b\left[\frac{1}{n} \sum_{i=1}^{n} \rho(\tau_i) G(\tau_i)\right] \\
&= \mathbb{E}_b[\rho(\tau) G(\tau)] \\
&= \sum_{\tau} P(\tau | b) \cdot \frac{P(\tau | \pi)}{P(\tau | b)} \cdot G(\tau) \\
&= \sum_{\tau} P(\tau | \pi) G(\tau) \\
&= \mathbb{E}_\pi[G(\tau)] \\
&= V_\pi(s)
\end{aligned}$$

**关键理解**：

重要性采样的无偏性来自于**概率测度的变换**。

通过乘以重要性比率，我们把在行为策略 b 下的期望转换成了在目标策略 π 下的期望。

**加权重要性采样的偏差**：

加权重要性采样：
$$\hat{V}_{\text{WIS}}(s) = \frac{\sum_{i=1}^{n} \rho_i G_i}{\sum_{i=1}^{n} \rho_i}$$

这个估计**有偏**，但方差更小。

偏差随着采样次数增加而趋于 0（渐近无偏）。

---

## 💻 算法实现

### 实现一：蒙特卡洛预测

```python
"""
蒙特卡洛预测算法

用采样估计给定策略的价值函数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class MCPrediction:
    """
    蒙特卡洛预测
    
    支持第一访问和每次访问两种模式
    
    Attributes:
        gamma (float): 折扣因子
        first_visit (bool): 是否使用第一访问 MC
        V (dict): 状态价值字典 {state: value}
        returns (dict): 回报列表 {state: [returns]}
    """
    
    def __init__(self, gamma: float = 0.99, first_visit: bool = True):
        """
        初始化 MC 预测器
        
        Args:
            gamma: 折扣因子
                   - 0.9-0.99：重视长期回报
                   - 0.5-0.9：平衡短期和长期
            first_visit: 是否使用第一访问 MC
                        - True: 第一访问 MC
                        - False: 每次访问 MC
        """
        self.gamma = gamma
        self.first_visit = first_visit
        
        # 状态价值估计
        self.V = defaultdict(float)
        
        # 回报列表（用于计算平均）
        self.returns = defaultdict(list)
        
        # 访问计数（用于分析）
        self.visit_counts = defaultdict(int)
    
    def compute_return(self, rewards: List[float], start_idx: int) -> float:
        """
        计算从指定位置开始的折扣回报
        
        公式：G_t = R_{t+1} + γ*R_{t+2} + γ²*R_{t+3} + ...
        
        Args:
            rewards: 奖励序列 [R_1, R_2, ..., R_T]
            start_idx: 起始索引 t
            
        Returns:
            G_t: 折扣回报
        """
        G = 0.0
        gamma_power = 1.0
        
        for t in range(start_idx, len(rewards)):
            G += gamma_power * rewards[t]
            gamma_power *= self.gamma
        
        return G
    
    def update(self, episode: List[Tuple]) -> None:
        """
        用一个 episode 更新价值估计
        
        Args:
            episode: episode 数据 [(s_0, a_0, r_1), (s_1, a_1, r_2), ..., (s_T, a_T, r_{T+1})]
        """
        # 提取奖励序列
        rewards = [step[2] for step in episode]
        
        # 记录每个状态是否已访问（用于第一访问 MC）
        visited = set()
        
        # 对 episode 中的每个状态进行更新
        for t, (state, action, _) in enumerate(episode):
            # 第一访问 MC：跳过已访问的状态
            if self.first_visit and state in visited:
                continue
            
            if self.first_visit:
                visited.add(state)
            
            # 计算回报 G_t
            G = self.compute_return(rewards, t)
            
            # 记录回报
            self.returns[state].append(G)
            self.visit_counts[state] += 1
            
            # 更新价值估计（增量式平均）
            # V_n = V_{n-1} + (1/n) * (G_n - V_{n-1})
            n = len(self.returns[state])
            self.V[state] += (1.0 / n) * (G - self.V[state])
    
    def predict(self, state) -> float:
        """
        预测状态价值
        
        Args:
            state: 状态
            
        Returns:
            V(state): 状态价值估计
        """
        return self.V[state]
    
    def get_value_function(self) -> Dict:
        """
        获取完整的价值函数
        
        Returns:
            V: 状态价值字典
        """
        return dict(self.V)
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            stats: 包含访问次数等信息
        """
        return {
            'n_states': len(self.V),
            'total_visits': sum(self.visit_counts.values()),
            'visit_counts': dict(self.visit_counts),
        }


def mc_prediction(
    env,
    policy,
    n_episodes: int = 10000,
    gamma: float = 0.99,
    first_visit: bool = True,
    verbose: bool = False,
) -> MCPrediction:
    """
    蒙特卡洛预测主函数
    
    Args:
        env: 环境（需要 reset 和 step 方法）
        policy: 策略函数 policy(state) -> action
        n_episodes: 训练的 episode 数量
        gamma: 折扣因子
        first_visit: 是否使用第一访问 MC
        verbose: 是否打印进度
        
    Returns:
        mc: 训练好的 MC 预测器
        
    Example:
        >>> mc = mc_prediction(env, policy, n_episodes=10000)
        >>> V = mc.get_value_function()
    """
    mc = MCPrediction(gamma=gamma, first_visit=first_visit)
    
    for episode_idx in range(n_episodes):
        # 生成一个 episode
        episode = []
        state, _ = env.reset()
        
        while True:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))
            
            state = next_state
            
            if done:
                break
        
        # 更新价值估计
        mc.update(episode)
        
        if verbose and (episode_idx + 1) % 1000 == 0:
            stats = mc.get_statistics()
            print(f"Episode {episode_idx + 1}: "
                  f"访问状态数 = {stats['n_states']}, "
                  f"总访问次数 = {stats['total_visits']}")
    
    return mc
```

### 实现二：蒙特卡洛控制

```python
"""
蒙特卡洛控制算法

用蒙特卡洛方法学习最优策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class MCControl:
    """
    蒙特卡洛控制
    
    使用 ε-greedy 策略进行探索
    
    Attributes:
        gamma (float): 折扣因子
        epsilon (float): 探索率
        Q (dict): 动作价值字典 {state: {action: value}}
        returns (dict): 回报列表 {state: {action: [returns]}}
        policy (dict): 当前策略 {state: {action: probability}}
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        n_actions: Optional[int] = None,
    ):
        """
        初始化 MC 控制器
        
        Args:
            gamma: 折扣因子
            epsilon: 探索率（ε-greedy）
            n_actions: 动作数量（可选，用于自动检测）
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        
        # 动作价值估计
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # 回报列表
        self.returns = defaultdict(lambda: defaultdict(list))
        
        # 访问计数
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        
        # 当前策略（ε-greedy）
        self.policy = {}
    
    def get_epsilon_greedy_policy(self, state):
        """
        获取 ε-greedy 策略
        
        Args:
            state: 状态
            
        Returns:
            policy: 动作概率字典 {action: probability}
        """
        # 自动检测动作数量
        if self.n_actions is None:
            raise ValueError("需要指定 n_actions 或先访问所有动作")
        
        # 获取最优动作
        if state in self.Q and len(self.Q[state]) > 0:
            best_action = max(self.Q[state].keys(), 
                            key=lambda a: self.Q[state][a])
        else:
            best_action = 0
        
        # 构建 ε-greedy 策略
        policy = {}
        for a in range(self.n_actions):
            if a == best_action:
                policy[a] = 1 - self.epsilon + self.epsilon / self.n_actions
            else:
                policy[a] = self.epsilon / self.n_actions
        
        return policy
    
    def select_action(self, state) -> int:
        """
        根据 ε-greedy 策略选择动作
        
        Args:
            state: 状态
            
        Returns:
            action: 选择的动作
        """
        policy = self.get_epsilon_greedy_policy(state)
        actions = list(policy.keys())
        probs = [policy[a] for a in actions]
        return np.random.choice(actions, p=probs)
    
    def compute_return(self, rewards: List[float], start_idx: int) -> float:
        """计算折扣回报"""
        G = 0.0
        gamma_power = 1.0
        
        for t in range(start_idx, len(rewards)):
            G += gamma_power * rewards[t]
            gamma_power *= self.gamma
        
        return G
    
    def update(self, episode: List[Tuple]) -> None:
        """
        用一个 episode 更新动作价值
        
        Args:
            episode: episode 数据 [(s_0, a_0, r_1), (s_1, a_1, r_2), ...]
        """
        rewards = [step[2] for step in episode]
        visited = set()
        
        for t, (state, action, _) in enumerate(episode):
            # 第一访问 MC
            if (state, action) in visited:
                continue
            visited.add((state, action))
            
            # 计算回报
            G = self.compute_return(rewards, t)
            
            # 记录回报
            self.returns[state][action].append(G)
            self.visit_counts[state][action] += 1
            
            # 更新 Q 值（增量式平均）
            n = len(self.returns[state][action])
            self.Q[state][action] += (1.0 / n) * (G - self.Q[state][action])
        
        # 更新策略
        self._update_policy()
    
    def _update_policy(self):
        """更新 ε-greedy 策略"""
        # 策略是隐式的，由 Q 值和 epsilon 决定
        pass
    
    def get_greedy_policy(self) -> Dict:
        """
        获取贪婪策略（用于评估）
        
        Returns:
            policy: {state: best_action}
        """
        greedy_policy = {}
        
        for state in self.Q:
            if len(self.Q[state]) > 0:
                best_action = max(self.Q[state].keys(),
                                key=lambda a: self.Q[state][a])
                greedy_policy[state] = best_action
        
        return greedy_policy
    
    def get_action_value(self, state, action) -> float:
        """获取动作价值"""
        return self.Q[state][action]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        n_states = len(self.Q)
        n_pairs = sum(len(actions) for actions in self.Q.values())
        total_visits = sum(
            sum(actions.values()) 
            for actions in self.visit_counts.values()
        )
        
        return {
            'n_states': n_states,
            'n_state_action_pairs': n_pairs,
            'total_visits': total_visits,
        }


def mc_control(
    env,
    n_episodes: int = 10000,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    verbose: bool = False,
) -> MCControl:
    """
    蒙特卡洛控制主函数
    
    Args:
        env: 环境
        n_episodes: 训练的 episode 数量
        gamma: 折扣因子
        epsilon: 探索率
        verbose: 是否打印进度
        
    Returns:
        mc: 训练好的 MC 控制器
        
    Example:
        >>> mc = mc_control(env, n_episodes=10000, epsilon=0.1)
        >>> policy = mc.get_greedy_policy()
    """
    # 自动检测动作数量
    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else None
    
    mc = MCControl(gamma=gamma, epsilon=epsilon, n_actions=n_actions)
    
    for episode_idx in range(n_episodes):
        # 生成一个 episode（用当前策略）
        episode = []
        state, _ = env.reset()
        
        while True:
            action = mc.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))
            
            state = next_state
            
            if done:
                break
        
        # 更新
        mc.update(episode)
        
        if verbose and (episode_idx + 1) % 1000 == 0:
            stats = mc.get_statistics()
            print(f"Episode {episode_idx + 1}: "
                  f"状态数 = {stats['n_states']}, "
                  f"状态 - 动作对 = {stats['n_state_action_pairs']}")
    
    return mc
```

### 实现三：GLIE 蒙特卡洛控制

```python
"""
GLIE 蒙特卡洛控制

使用衰减的 ε 实现 Greedy in the Limit with Infinite Exploration
"""

import numpy as np
from typing import Optional


class GLIEMCControl(MCControl):
    """
    GLIE MC 控制
    
    ε 随时间衰减：ε_k = 1/k
    
    保证：
    1. 所有状态 - 动作对访问无限次
    2. 策略最终收敛到贪婪策略
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        n_actions: Optional[int] = None,
    ):
        super().__init__(gamma=gamma, epsilon=1.0, n_actions=n_actions)
        self.episode_count = 0
    
    def select_action(self, state) -> int:
        """
        根据当前 ε 选择动作
        
        ε_k = 1/k，随 episode 增加而减小
        """
        # 更新 episode 计数
        self.episode_count += 1
        
        # 计算当前 ε
        current_epsilon = 1.0 / self.episode_count
        self.epsilon = current_epsilon
        
        return super().select_action(state)
    
    def get_current_epsilon(self) -> float:
        """获取当前 ε 值"""
        return self.epsilon
```

（第二部分完成，待续...）
## 🔬 算法对比

### 第一访问 MC vs 每次访问 MC

| 维度 | 第一访问 MC | 每次访问 MC |
|------|-------------|-------------|
| **数据处理** | 只用第一次访问的回报 | 用所有访问的回报 |
| **实现复杂度** | 需要记录已访问状态 | 更简单 |
| **数据利用率** | 低（忽略后续访问） | 高（利用所有数据） |
| **方差** | 较高 | 较低 |
| **收敛性** | 收敛到 V_π | 收敛到 V_π |
| **理论分析** | 更简单 | 稍复杂 |
| **推荐场景** | 理论研究 | 实际应用 |

### MC vs DP

| 维度 | 动态规划 | 蒙特卡洛 |
|------|----------|----------|
| **MDP 要求** | 需要完整模型 | 只需采样 |
| **更新时机** | 每步更新（自举） | episode 结束更新 |
| **偏差 - 方差** | 有偏、低方差 | 无偏、高方差 |
| **适用任务** | 任何 MDP | 仅回合制任务 |
| **样本效率** | 高（用模型） | 低（需大量采样） |
| **计算效率** | 每步计算量大 | 每步计算量小 |
| **典型应用** | 规划问题 | 学习问题 |

### On-Policy vs Off-Policy MC

| 维度 | On-Policy | Off-Policy |
|------|-----------|------------|
| **数据源** | 当前策略生成 | 任意策略生成 |
| **学习目标** | 当前策略的价值 | 目标策略的价值 |
| **重要性采样** | 不需要 | 需要 |
| **方差** | 较低 | 较高（因为权重） |
| **应用场景** | 标准 MC 控制 | 从历史数据学习 |

---

## 🧪 动手实验

### 实验一：第一访问 vs 每次访问 MC 对比

**任务描述**：

在 Blackjack 环境上对比两种 MC 方法的收敛行为。

**环境设置**：
- Blackjack 环境
- 初始策略：随机策略（要牌/停牌各 50%）
- 训练 10000 个 episode

**实验步骤**：

1. 用第一访问 MC 训练，记录每个 state 的 V(s) 估计
2. 用每次访问 MC 训练，同样记录
3. 每 1000 个 episode 计算一次估计误差（与真实值比较）
4. 绘制收敛曲线

**分析指标**：
- 收敛速度（达到目标误差所需的 episode 数）
- 最终误差
- 方差（多次运行的标准差）

**预期结果**：
- 每次访问 MC 收敛更快（利用更多数据）
- 两者最终收敛到相同值
- 每次访问 MC 方差更小

**思考题**：
1. 为什么每次访问 MC 方差更小？
2. 在什么情况下第一访问 MC 可能更好？
3. 如果 episode 很长，两种方法的差异如何变化？

### 实验二：ε 值对 MC 控制的影响

**任务描述**：

研究探索率 ε 对 MC 控制学习效果的影响。

**参数设置**：

| 组别 | ε | 预期行为 |
|------|---|----------|
| A | 0.01 | 几乎不探索，可能陷入次优 |
| B | 0.1 | 平衡探索和利用 |
| C | 0.5 | 大量探索，学习慢 |
| D | 1.0/k | GLIE 衰减 |

**实验步骤**：

1. 对每组 ε，运行 MC 控制
2. 每 1000 个 episode 评估一次当前策略（用贪婪策略玩 100 局）
3. 绘制学习曲线（胜率 vs episode）

**分析指标**：
- 最终胜率
- 收敛速度
- 策略稳定性

**预期结果**：
- ε 太小：收敛快但可能次优
- ε 太大：收敛慢但最终可能更好
- GLIE：理论上最优，但实际中固定 ε 可能足够

### 实验三：Blackjack 最优策略可视化

**任务描述**：

用 MC 控制学习 Blackjack 最优策略，并可视化。

**实现步骤**：

1. 用 MC 控制训练 100000 个 episode
2. 提取贪婪策略
3. 对于每个 (玩家点数，庄家明牌) 组合，显示最优行动
4. 绘制策略热力图

**可视化示例**：
```
庄家明牌 →
玩家点数 ↓    2    3    4    5    6    7    8    9   10    A
     12       S    S    S    S    S    H    H    H    H    H
     13       S    S    S    S    S    H    H    H    H    H
     14       S    S    S    S    S    H    H    H    H    H
     15       S    S    S    S    S    H    H    H    H    H
     16       S    S    S    S    S    H    H    H    H    H
     17       S    S    S    S    S    S    S    H    H    H
     18       S    S    S    S    S    S    S    S    S    S
     19       S    S    S    S    S    S    S    S    S    S
     20       S    S    S    S    S    S    S    S    S    S
     21       S    S    S    S    S    S    S    S    S    S
```
（S=停牌，H=要牌）

**分析**：
- 对比已知的 Blackjack 基本策略
- 分析差异原因
- 讨论 MC 学习的局限性

### 实验四：重要性采样演示

**任务描述**：

实现重要性采样，演示如何从行为策略估计目标策略的价值。

**环境设置**：
- 简单网格世界（5×5）
- 行为策略：随机策略
- 目标策略：贪婪策略

**实验步骤**：

1. 用随机策略生成 1000 个 episode
2. 用普通 MC 估计随机策略的价值
3. 用重要性采样估计贪婪策略的价值
4. 对比真实值（用 DP 计算）

**分析指标**：
- 估计误差
- 重要性权重的分布
- 方差分析

**思考题**：
1. 为什么重要性采样的方差大？
2. 如何减小方差（如加权重要性采样）？
3. 在什么情况下重要性采样特别有用？

---

## ❓ 常见问题

### Q1: 蒙特卡洛方法为什么只适用于回合制任务？

**A**: 因为蒙特卡洛需要计算完整回报 G_t。

回报的定义：
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$$

对于回合制任务，episode 最终会终止，回报是有限项的和。

对于连续任务（没有终止状态），回报是无限项的和：
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

如果 γ < 1，这个级数收敛，理论上可以计算。但实际中无法等待无限步。

**解决方案**：
1. 用时序差分（第 4 章）—— 不需要等 episode 结束
2. 用截断的回报 —— 只计算 n 步
3. 设计终止条件 —— 强制回合制

### Q2: 蒙特卡洛方法的收敛速度有多快？

**A**: 收敛速度是 $O(1/\sqrt{N})$。

根据中心极限定理：
$$\sqrt{N}(\hat{V}_N - V) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

这意味着：
- 要让误差减半，需要 4 倍的采样次数
- 要让误差减少到 1/10，需要 100 倍的采样次数

**影响收敛速度的因素**：
1. **方差**：回报的方差 σ² 越大，收敛越慢
2. **折扣因子**：γ 越大，回报方差越大，收敛越慢
3. **探索率**：ε 影响采样分布，间接影响收敛

**加速技巧**：
1. 重要性采样（利用历史数据）
2. 控制变量法（减少方差）
3. 分层采样（确保充分探索）

### Q3: ε-greedy 策略中 ε 怎么选择？

**A**: 没有统一答案，取决于任务。

**经验法则**：
- 简单任务（状态少）：ε = 0.1-0.2
- 复杂任务（状态多）：ε = 0.05-0.1
- 非常复杂：ε = 0.01-0.05 + GLIE 衰减

**调参建议**：
1. 从 ε = 0.1 开始
2. 观察学习曲线
3. 如果收敛太慢 → 减小 ε
4. 如果性能上不去 → 增大 ε 或用 GLIE

**理论保证**：
GLIE 条件（ε_k = 1/k）保证收敛到最优，但实际中可能太慢。

### Q4: 重要性采样的权重为什么可能很大？

**A**: 因为重要性比率是概率的比值。

$$\rho = \frac{P(\tau | \pi)}{P(\tau | b)} = \prod_{t} \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$$

如果某个动作在目标策略下概率高（如 0.9），在行为策略下概率低（如 0.1），则：
$$\frac{\pi(a|s)}{b(a|s)} = \frac{0.9}{0.1} = 9$$

如果 episode 很长，多个这样的比率相乘，权重会指数级增长。

**问题**：
- 方差极大
- 少数样本主导估计

**解决方案**：
1. 加权重要性采样（归一化权重）
2. 截断重要性采样（限制最大权重）
3. 选择相近的行为策略和目标策略

### Q5: 蒙特卡洛方法在实际中有哪些应用？

**A**: 蒙特卡洛方法广泛应用于：

**游戏 AI**：
- 21 点、扑克等卡牌游戏
- 围棋（MCTS 的基础）
- 视频游戏（Atari）

**机器人学**：
- 从试错中学习控制策略
- 模仿学习（从人类演示学习）

**推荐系统**：
- 从用户交互历史学习
- A/B 测试数据分析

**金融**：
- 期权定价
- 风险评估

**物理学**：
- 粒子模拟
- 统计力学

---

## 📚 延伸阅读

### 核心教材

1. **Sutton & Barto, Chapter 5**: Monte Carlo Methods
   - https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
   - 最权威的 MC 教程

2. **Szepesvári (2010). Algorithms for Reinforcement Learning**
   - 第 2 章讲 MC 方法
   - 理论分析深入

### 应用论文

1. **Tesauro (1995). Temporal Difference Learning and TD-Gammon**
   - 用 TD(λ) 学习西洋双陆棋
   - MC 和 TD 的对比

2. **Silver et al. (2016). Mastering the game of Go with deep neural networks**
   - AlphaGo 的 MCTS
   - MC 在现代 RL 中的应用

### 代码资源

| 文件 | 内容 | 行数 |
|------|------|------|
| `mc_prediction.py` | MC 预测算法 | ~150 |
| `mc_control.py` | MC 控制算法 | ~180 |
| `epsilon_greedy.py` | ε-greedy 策略 | ~80 |
| `games/blackjack.py` | 21 点环境 | ~200 |

---

## ✅ 本章检查清单

学完本章后，你应该能够：

**概念理解**：
- [ ] 解释蒙特卡洛方法的核心思想
- [ ] 区分第一访问和每次访问 MC
- [ ] 解释探索 - 利用权衡
- [ ] 解释离策略学习的意义

**数学推导**：
- [ ] 推导 MC 估计的收敛性
- [ ] 推导重要性采样的无偏性
- [ ] 推导 ε-greedy 策略改进定理

**代码实现**：
- [ ] 实现 MC 预测算法
- [ ] 实现 MC 控制算法
- [ ] 实现 GLIE MC 控制
- [ ] 实现 21 点环境

**实验分析**：
- [ ] 对比第一访问和每次访问 MC
- [ ] 分析 ε 值的影响
- [ ] 可视化 Blackjack 策略
- [ ] 演示重要性采样

**应用判断**：
- [ ] 根据任务选择 MC 或 DP
- [ ] 合理设置 ε 参数
- [ ] 诊断 MC 学习中的问题

---

## 🚀 下一章预告

**第 4 章：时序差分学习**

结合动态规划和蒙特卡洛的优点：
- 像 DP 一样每步更新（无需等 episode 结束）
- 像 MC 一样无需模型（直接采样学习）
- SARSA、Q-Learning 等核心算法

**预告实验**：悬崖行走 —— 对比 SARSA 和 Q-Learning 的行为差异！🏃

---

*最后更新：2026-04-22*  
*作者：Hermes <neko_yukirin@qq.com>*
