# 第 7 章：高级策略优化与 LLM 后训练 ⭐

> **从经典 RL 到 LLM 对齐的完整桥梁** —— 理解 PPO/DPO/GRPO 如何重塑大模型

---

## ⏱️ TL;DR（30 秒速览）

**核心问题**：LLM 如何对齐人类偏好？

**核心思想**：
> 用 RL 优化策略，让模型输出符合人类价值观。

**关键算法**：
- **PPO**：截断策略比率，稳定训练
- **DPO**：直接从偏好数据优化，无需奖励模型
- **GRPO**：组内比较，自动产生学习信号
- **Offline RL**：从固定数据集学习

**学完这章你能**：
- ✅ 理解 PPO/DPO/GRPO 的原理
- ✅ 区分不同后训练方法
- ✅ 实现简单版本的 DPO
- ✅ 理解 LLM 对齐的 RL 基础

**常见误区**：
- ❌ DPO 不需要偏好数据 → ✅ 需要，只是不需要奖励模型
- ❌ PPO 只用于 LLM → ✅ 通用 RL 算法
- ❌ GRPO 适用于所有任务 → ✅ 仅适用于可评分任务

---

---

## 🎯 本章要解决什么问题

如果你关注大语言模型（LLM）的最新进展，你一定听说过这些术语：**RLHF**、**PPO**、**DPO**、**GRPO**。这些技术是 ChatGPT、Claude、Gemini 等模型能够"对齐"人类偏好的核心秘密。但问题是，大多数关于 LLM 后训练的教程都跳过了一个关键问题：**这些方法背后的强化学习原理到底是什么？**

直接学习 LLM 后训练会遇到几个根本性困难：

**第一，概念跳跃太大**。从监督微调（SFT）直接跳到 PPO 优化，中间缺少了策略梯度、优势函数、重要性采样等 RL 基础概念的铺垫。结果是，很多人能把 PPO 代码跑起来，但完全不理解为什么这样设计。

**第二，数学门槛高**。PPO 的截断目标函数、DPO 的隐式奖励、GRPO 的组内标准化，这些概念如果直接从论文中学习，需要很强的数学背景。但如果先从直观理解入手，再逐步形式化，门槛会大幅降低。

**第三，实践成本高**。在真实 LLM 上实验 PPO/DPO 需要大量 GPU 资源和 API 调用费用。但如果先在经典 RL 环境（如 CartPole、Pendulum）上理解算法核心，再迁移到 LLM 场景，成本会大幅降低。

本章的设计目标就是解决这三个问题。我们会：

1. **从经典 RL 算法出发**，逐步推导到 LLM 后训练方法
2. **用直观类比 + 完整数学**，讲透每个算法的设计动机
3. **提供可运行的代码示例**，先在简单环境验证，再讨论 LLM 迁移

学完本章后，你将能够：
- 深刻理解 PPO 为什么比 vanilla Policy Gradient 更稳定
- 理解 DPO 如何绕过奖励模型直接优化偏好
- 理解 GRPO 如何通过组内比较产生学习信号
- 能够根据任务特点选择合适的后训练方法
- 为阅读 LLM 对齐领域的前沿论文打下坚实基础

---

## 📖 场景描述：从游戏控制到对话生成

### 场景一：训练 CartPole 智能体

想象你要训练一个智能体玩 CartPole 游戏（倒立摆平衡）。一根杆子竖立在可移动的小车上，你的目标是通过左右移动小车来保持杆子不倒。

**状态空间**：小车位置、速度、杆子角度、角速度（4 维）
**动作空间**：向左推、向右推（2 维离散）
**奖励**：每存活一步 +1 分

用第 6 章学到的 REINFORCE 算法，你可以训练出一个能平衡几百步的智能体。但你会发现训练过程很不稳定：
- 有时学习曲线突然崩溃，性能急剧下降
- 需要仔细调学习率，太大就发散，太小就不学
- 每次训练结果差异很大，重复性差

**这就是策略梯度算法的核心问题：更新步长难以控制。**

PPO 的发明就是为了解决这个问题。它的核心洞察是：**与其让策略更新"正确"，不如让策略更新"可控"**。通过限制每次更新的幅度，PPO 保证了训练过程的稳定性。

### 场景二：训练 LLM 生成有用回答

现在把场景从游戏升级到 LLM 对话。假设你已经用 SFT（监督微调）训练了一个基础模型，它能回答各种问题，但有时候回答不够有用、不够安全、或者不符合人类偏好。

**状态空间**：当前生成的 token 序列（高维离散）
**动作空间**：词表中的所有 token（几万维离散）
**奖励**：人类偏好评分（需要从偏好数据学习）

这个场景比 CartPole 复杂几个数量级：
- 状态空间是指数级的（所有可能的 token 序列）
- 动作空间巨大（词表大小）
- 奖励稀疏且昂贵（需要人类标注或奖励模型）

**传统 RLHF 流程**：
1. 收集人类偏好数据（哪个回答更好）
2. 训练一个奖励模型来预测人类偏好
3. 用 PPO 优化 LLM 策略，最大化奖励模型输出

这个流程有效，但复杂且昂贵。DPO 和 GRPO 的出现，就是为了简化这个流程。

**DPO 的洞察**：为什么需要显式的奖励模型？我们可以直接从偏好数据推导出最优策略应该满足的条件。

**GRPO 的洞察**：对于有标准答案的任务（如数学题），为什么需要人类偏好？我们可以通过组内比较自动生成学习信号。

### 场景三：从离线数据学习

想象你有一个巨大的数据集，里面包含了人类专家的操作记录。你想从这些数据中学习一个智能策略，但你**不能**再与环境交互（可能是成本太高，或者环境已经不存在了）。

这就是**离线强化学习（Offline RL）**的场景。与在线 RL 不同，Offline RL 只能从固定数据集中学习，不能主动探索。

**核心挑战**：如果数据集没有覆盖某些状态 - 动作对，学习到的策略可能会选择这些"分布外"（OOD）动作，而 Q 函数对这些动作的估计会非常不准确（外推误差）。

**CQL 的解决方案**：让 Q 函数对 OOD 动作给出保守（低）估计，这样策略就不会选择这些动作。

**IQL 的解决方案**：用期望回归只学习数据集内的价值，然后用优势加权行为克隆提取策略。

这三个场景看似不同，但核心都是**策略优化**问题。接下来我们会深入每个算法的数学原理和实现细节。

---

## 🧠 核心概念详解

### 概念一：策略梯度的不稳定性问题

**直觉理解**：

策略梯度算法的核心思想很简单：如果某个动作带来了高奖励，那就增加选择它的概率；如果带来了低奖励，就降低概率。这听起来很合理，但实际操作中会遇到一个致命问题：**更新步长难以控制**。

想象你在下山（最小化损失），策略梯度告诉你应该往哪个方向走。但问题是，它不能准确告诉你应该走多远。如果你走得太小，下山会很慢；如果你走得太大，可能会直接冲过谷底，甚至跑到对面的山上去了。

在深度学习中，我们通常用学习率来控制步长。但策略梯度有个特殊问题：**策略更新会改变数据分布**。也就是说，你更新策略后，接下来采样到的数据也会变化。这导致了一个恶性循环：
1. 策略更新过大
2. 新策略采样的数据质量变差
3. 基于差数据的梯度估计不准确
4. 下一次更新方向错误
5. 策略性能崩溃

**PPO 的核心贡献**就是解决了这个问题。

**形式化分析**：

策略梯度的目标函数是：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$$

它的梯度是：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) A_t]$$

其中 $A_t$ 是优势函数（动作比平均好多少）。

关键问题在于，这个梯度公式假设数据是从当前策略 $\pi_\theta$ 采样的。但当我们用旧策略 $\pi_{\theta_{old}}$ 采样的数据来更新新策略时，需要用重要性采样修正：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}}[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) A_t]$$

这里的重要性权重 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 就是问题所在。如果新旧策略差异太大，这个权重会非常大或非常小，导致梯度估计的方差爆炸。

**TRPO（Trust Region Policy Optimization）**的解决方案是用约束优化限制策略更新幅度：

$$\max_\theta \mathbb{E}[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_t] \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$$

但 TRPO 需要求解复杂的二阶优化问题，实现困难。

**PPO 的突破**是用一个巧妙的截断技巧，用一阶优化实现了类似 TRPO 的效果。

### 概念二：PPO 的截断目标函数

**直觉理解**：

PPO 的核心创新是**截断策略比率**。策略比率 $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 衡量新旧策略的差异。如果这个比率远离 1，说明策略更新太大了。

PPO 的想法很简单：**如果策略比率超出 [1-ε, 1+ε] 范围，就把它截断到这个范围内**。这样，无论梯度告诉你应该更新多少，实际更新都不会超过这个界限。

这个技巧的妙处在于：
1. 实现简单（就一行代码）
2. 计算高效（不需要二阶导数）
3. 效果显著（训练稳定性大幅提升）

**形式化定义**：

PPO 的截断目标函数定义为：

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是策略比率
- $A_t$ 是优势函数估计
- $\epsilon$ 是截断参数（通常取 0.2）

**为什么这样设计**：

这个目标函数的设计非常巧妙。让我们分析几种情况：

**情况 1：优势为正（动作比平均好）**
- 我们希望增加这个动作的概率（$r_t > 1$）
- 如果 $r_t < 1+\epsilon$，目标函数是 $r_t A_t$（鼓励增加）
- 如果 $r_t > 1+\epsilon$，目标函数是 $(1+\epsilon) A_t$（截断，不再鼓励）

**情况 2：优势为负（动作比平均差）**
- 我们希望减少这个动作的概率（$r_t < 1$）
- 如果 $r_t > 1-\epsilon$，目标函数是 $r_t A_t$（鼓励减少，因为 A 为负）
- 如果 $r_t < 1-\epsilon$，目标函数是 $(1-\epsilon) A_t$（截断，不再鼓励）

关键是 $\min$ 操作。它保证了：**只有当策略更新在安全范围内时，目标函数才会被优化；一旦超出范围，目标函数就不再提供梯度**。这就像一个"安全阀"，防止策略更新过大。

### 概念三：DPO 的隐式奖励

**直觉理解**：

传统 RLHF 需要显式训练一个奖励模型，这个奖励模型用来预测人类偏好。但 DPO 提出了一个深刻的问题：**我们真的需要显式的奖励模型吗？**

DPO 的关键洞察来自对最优策略的数学分析。在给定奖励函数 r 的情况下，最优策略满足：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp(\frac{1}{\beta} r(x, y))$$

其中 $\pi_{ref}$ 是参考策略（通常是 SFT 模型），$Z(x)$ 是归一化常数。

这个公式可以反解出奖励函数：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**这就是 DPO 的核心：最优策略本身就编码了奖励函数！**

我们不需要显式学习 r(x, y)，只需要直接优化策略 $\pi_\theta$，让它满足上述关系。

**形式化推导**：

给定偏好数据 $(x, y_w, y_l)$，其中 $y_w$ 是优选回答，$y_l$ 是劣选回答。人类偏好满足 Bradley-Terry 模型：

$$P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

将奖励函数用策略表示：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

代入偏好概率：

$$P(y_w \succ y_l | x) = \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})$$

注意 $\log Z(x)$ 项抵消了！这意味着我们可以直接优化策略，而不需要知道归一化常数。

DPO 的损失函数就是负对数似然：

$$L_{DPO}(\theta) = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$$

**为什么这样设计**：

DPO 相比传统 RLHF 有几个关键优势：

1. **简化流程**：不需要训练和维护奖励模型
2. **计算高效**：只需要前向传播，不需要 PPO 那样的多轮采样
3. **稳定性好**：没有 PPO 的超参数（如 clip epsilon、GAE lambda）
4. **理论保证**：从最优性条件推导而来，不是启发式设计

但 DPO 也有局限：
- 需要高质量的偏好数据
- 参考策略 $\pi_{ref}$ 的质量影响最终结果
- 对于没有明确偏好的任务（如创意写作），效果可能不如 PPO

### 概念四：GRPO 的组内比较

**直觉理解**：

GRPO 针对的是一类特殊任务：**有明确正确答案的任务**，如数学题、代码生成、逻辑推理等。

对于这类任务，传统 RLHF 流程显得笨重：
1. 收集人类偏好数据（但答案对错是客观的，不需要人类判断）
2. 训练奖励模型（但可以用规则直接评分）
3. PPO 优化（这部分可以保留）

GRPO 的洞察是：**对于同一问题，生成多个答案，然后通过组内比较自动生成学习信号**。

具体做法：
1. 对每个问题 x，生成 G 个答案 $\{y_1, ..., y_G\}$
2. 用规则评分（如数学题看答案是否正确）得到奖励 $\{r_1, ..., r_G\}$
3. 计算组内标准化优势：$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$
4. 用 PPO 风格的目标函数更新策略

**为什么这样设计**：

组内标准化有几个关键好处：

1. **自动基线**：不需要额外的价值函数来估计基线
2. **方差降低**：组内比较消除了问题难度的影响
3. **样本高效**：一次生成多个答案，充分利用计算资源
4. **无需奖励模型**：对于有客观标准的任务，直接用规则评分

GRPO 特别适合以下场景：
- 数学问题求解（答案对错明确）
- 代码生成（可以通过测试用例评分）
- 逻辑推理（有标准答案）

但对于开放性任务（如创意写作、对话生成），GRPO 的效果可能不如 DPO 或 PPO，因为缺乏明确的评分标准。

### 概念五：Offline RL 的分布外挑战

**直觉理解**：

Offline RL 的核心挑战是**分布外（OOD）动作**。想象你从人类专家的数据集中学习开车。数据集中，人类专家在某个路口总是减速慢行。但你的策略可能学到：在这个路口加速也是可以的（因为 Q 函数对这个动作的估计不准确）。

为什么 Q 函数会不准确？因为 Q 函数是通过贝尔曼方程迭代学习的：

$$Q(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q(s', a')]$$

如果某个动作 $a'$ 在数据集中从未出现过，Q 函数对它的估计就是外推的，可能严重偏离真实值。更糟糕的是，策略会倾向于选择 Q 值高的动作，如果 Q 值被高估，策略就会选择 OOD 动作，形成恶性循环。

**CQL 的解决方案**：

CQL（Conservative Q-Learning）的核心思想是：**让 Q 函数对 OOD 动作给出保守（低）估计**。

具体做法是在 Q 学习的基础上增加一个正则化项：

$$L_{CQL}(Q) = L_{TD}(Q) + \alpha (\mathbb{E}_{s,a \sim \pi}[Q(s, a)] - \mathbb{E}_{s,a \sim D}[Q(s, a)])$$

其中第一项是标准的 TD 损失，第二项鼓励 Q 函数在策略动作上的期望值低于在数据集动作上的期望值。

直观上，这就像给 Q 函数加了一个"惩罚"：如果你给 OOD 动作高 Q 值，正则化项就会惩罚你。

**IQL 的解决方案**：

IQL（Implicit Q-Learning）采用了不同的策略：**完全避免 OOD 动作的问题**。

IQL 分两步：
1. 用期望回归学习值函数 V(s)，只关注数据集内的高 Q 值动作
2. 用优势加权行为克隆学习策略，权重是 $\exp(\beta(Q(s,a) - V(s)))$

关键技巧是期望回归的损失函数：

$$L_V = \mathbb{E}_{s,a \sim D}[(V(s) - Q(s,a))^2]$$

但这个损失只会让 V(s) 接近 Q 值的平均值，而 IQL 想要的是高 Q 值。所以 IQL 用了一个不对称的损失：

$$L_V = \mathbb{E}_{s,a \sim D}[L_\tau(V(s) - Q(s,a))]$$

其中 $L_\tau$ 是期望分位数损失，参数 $\tau$ 控制关注的分位数（如 $\tau=0.7$ 表示关注上 70% 分位数）。

这样学到的 V(s) 接近数据集中高 Q 值动作的价值，然后用优势加权 BC 提取策略。

---

（第一部分完成，待续...）
## 📐 公式推导

### 推导一：PPO 的截断目标函数

**问题设定**：

我们想要最大化策略的期望回报，但要用旧策略采样的数据来更新新策略。重要性采样给出的目标函数是：

$$J(\theta) = \mathbb{E}_{s \sim \rho_{old}, a \sim \pi_{old}}[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s, a)]$$

其中 $\rho_{old}$ 是旧策略的状态分布。

**TRPO 的约束优化**：

TRPO 通过约束策略更新幅度来保证稳定性：

$$\max_\theta \mathbb{E}[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s, a)] \quad \text{s.t.} \quad \mathbb{E}[D_{KL}(\pi_{old}(\cdot|s) || \pi_\theta(\cdot|s))] \leq \delta$$

这个约束优化问题需要求解二阶导数（Hessian 矩阵），计算成本高。

**PPO 的截断技巧**：

PPO 观察到，TRPO 的约束本质上是为了防止策略比率 $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$ 偏离 1 太远。那为什么不直接截断这个比率呢？

定义截断后的策略比率：

$$\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) = \begin{cases} 1-\epsilon & \text{if } r(\theta) < 1-\epsilon \\ r(\theta) & \text{if } 1-\epsilon \leq r(\theta) \leq 1+\epsilon \\ 1+\epsilon & \text{if } r(\theta) > 1+\epsilon \end{cases}$$

PPO 的目标函数是：

$$L^{CLIP}(\theta) = \mathbb{E}[\min(r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A)]$$

**关键分析**：

让我们分析这个目标函数在不同情况下的行为。

**情况 1：A > 0（动作优于平均）**

此时我们希望增加动作概率（$r(\theta) > 1$）。

- 如果 $r(\theta) < 1+\epsilon$：$\min(r(\theta) A, (1+\epsilon) A) = r(\theta) A$
  - 目标函数鼓励增加 $r(\theta)$
  
- 如果 $r(\theta) \geq 1+\epsilon$：$\min(r(\theta) A, (1+\epsilon) A) = (1+\epsilon) A$
  - 目标函数是常数，不再提供梯度
  - 策略更新被"截断"

**情况 2：A < 0（动作劣于平均）**

此时我们希望减少动作概率（$r(\theta) < 1$）。

- 如果 $r(\theta) > 1-\epsilon$：$\min(r(\theta) A, (1-\epsilon) A) = r(\theta) A$（注意 A 为负）
  - 目标函数鼓励减少 $r(\theta)$
  
- 如果 $r(\theta) \leq 1-\epsilon$：$\min(r(\theta) A, (1-\epsilon) A) = (1-\epsilon) A$
  - 目标函数是常数，不再提供梯度

**结论**：无论优势是正还是负，策略比率都被限制在 $[1-\epsilon, 1+\epsilon]$ 范围内。这就是 PPO 稳定性的来源。

**PPO-Clip 算法**：

```
输入：旧策略 π_old，截断参数 ε
初始化：策略参数 θ

for iteration = 1, 2, ...:
    # 1. 用旧策略采样
    采样轨迹 {τ_i}，计算优势 {A_t}
    
    # 2. 多次 PPO 更新
    for epoch = 1, ..., K:
        for 每个样本 (s, a, A_t):
            r(θ) = π_θ(a|s) / π_old(a|s)
            L^CLIP = min(r(θ) * A_t, clip(r(θ), 1-ε, 1+ε) * A_t)
            θ ← θ + α * ∇_θ L^CLIP
    
    # 3. 更新旧策略
    π_old ← π_θ
```

### 推导二：GAE（广义优势估计）

**问题设定**：

PPO 需要优势函数 $A(s, a) = Q(s, a) - V(s)$ 来计算更新方向。但 Q(s, a) 未知，需要估计。

**n 步回报**：

一个自然的想法是用 n 步回报来估计优势：

$$A_t^{(n)} = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)$$

- n=1：单步 TD 误差，低方差但有偏
- n=∞：蒙特卡洛回报，无偏但高方差

**GAE 的核心思想**：

GAE（Generalized Advantage Estimation）用指数加权平均所有 n 步回报：

$$A_t^{GAE} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} A_t^{(n)}$$

其中 $\lambda \in [0, 1]$ 控制方差 - 偏差权衡。

**简化形式**：

可以证明，GAE 有一个简洁的表达式：

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

**实际计算**：

对于有限长度的轨迹（长度 T），GAE 可以反向计算：

```
A_T = 0
for t = T-1, T-2, ..., 0:
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = δ_t + γ * λ * A_{t+1}
```

**为什么这样设计**：

GAE 的设计平衡了方差和偏差：
- $\lambda = 0$：GAE 退化为单步 TD 误差，低方差但有偏
- $\lambda = 1$：GAE 退化为蒙特卡洛优势，无偏但高方差
- $\lambda = 0.95$：经验上不错的折中

### 推导三：DPO 的最优性条件

**问题设定**：

我们想要从偏好数据中学习最优策略，但不想显式学习奖励函数。

**最大熵 RL 框架**：

在最大熵 RL 中，最优策略满足：

$$\pi^*(a|s) = \frac{1}{Z(s)} \pi_{ref}(a|s) \exp(\frac{1}{\beta} Q^*(s, a))$$

其中 $\pi_{ref}$ 是参考策略，$\beta$ 是温度参数，$Z(s)$ 是归一化常数。

对于语言模型，状态 s 是 prompt x，动作 a 是响应 y：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp(\frac{1}{\beta} r(x, y))$$

**反解奖励函数**：

从上述公式可以解出奖励函数：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Bradley-Terry 偏好模型**：

人类偏好满足：

$$P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

代入奖励函数表达式：

$$P(y_w \succ y_l | x) = \sigma(\beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)})$$

注意 $\log Z(x)$ 项抵消了！

**DPO 损失**：

最大化偏好数据的对数似然等价于最小化负对数似然：

$$L_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$$

**为什么这样设计**：

DPO 的关键优势是**不需要归一化常数 $Z(x)$**。在语言模型中，计算 $Z(x)$ 需要对所有可能的响应求和，这是不可能的。DPO 通过巧妙的数学变换绕过了这个问题。

### 推导四：GRPO 的优势标准化

**问题设定**：

对于有明确答案的任务，我们想要自动生成学习信号，而不需要奖励模型。

**组内采样**：

对每个问题 x，生成 G 个响应 $\{y_1, ..., y_G\}$，然后用规则评分得到奖励 $\{r_1, ..., r_G\}$。

**优势估计**：

标准的策略梯度需要优势函数 $A(y_i) = Q(x, y_i) - V(x)$。但 Q 和 V 都未知。

GRPO 的洞察是：**组内奖励的标准化可以作为优势的代理**。

定义组内标准化优势：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\}) + \epsilon}$$

**数学分析**：

假设奖励可以分解为：

$$r_i = V(x) + A(x, y_i) + \eta_i$$

其中：
- $V(x)$ 是问题 x 的基础难度（所有响应的平均奖励）
- $A(x, y_i)$ 是响应 $y_i$ 的优势（比平均好多少）
- $\eta_i$ 是噪声

组内均值：

$$\text{mean}(r) = V(x) + \text{mean}(A) + \text{mean}(\eta) \approx V(x)$$

（假设优势均值为 0，噪声均值为 0）

组内标准化：

$$\frac{r_i - \text{mean}(r)}{\text{std}(r)} \approx \frac{A(x, y_i)}{\text{std}(A)}$$

这说明组内标准化后的奖励与优势成正比。

**为什么这样设计**：

组内标准化有几个关键好处：

1. **消除问题难度影响**：难题和简单题的奖励可能差异很大，但组内标准化后都在同一尺度
2. **自动基线**：不需要额外的价值函数
3. **方差降低**：标准化后的优势方差约为 1

---

## 💻 算法实现

### 实现一：PPO 核心更新

```python
"""
PPO 核心更新逻辑

实现截断目标函数和 GAE 优势估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (GAE)
    
    公式：A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}
    其中 δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    
    Args:
        rewards: 奖励序列 [T]
        values: 价值估计 [T+1]（包含最后状态的 value）
        dones: 终止标志 [T]
        gamma: 折扣因子
        lam: GAE lambda 参数
        
    Returns:
        advantages: 优势估计 [T]
        returns: 回报估计 [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    
    gae = 0.0
    for t in reversed(range(T)):
        # 计算 TD 误差
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE：指数加权累积 TD 误差
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # 回报 = 优势 + 基线价值
    returns = advantages + values[:T]
    
    return advantages, returns


def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    """
    计算 PPO 截断损失
    
    公式：L^CLIP = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
    其中 r = exp(new_log_probs - old_log_probs)
    
    Args:
        old_log_probs: 旧策略的对数概率 [B]
        new_log_probs: 新策略的对数概率 [B]
        advantages: 优势函数估计 [B]
        clip_epsilon: 截断参数
        
    Returns:
        policy_loss: 策略损失（标量）
        clip_fraction: 被截断的样本比例（用于监控）
    """
    # 计算策略比率 r(θ) = π_θ(a|s) / π_old(a|s)
    # 用对数概率计算更稳定：r = exp(log_new - log_old)
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 计算截断后的比率
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    
    # 计算两个目标
    # surr1: 未截断的目标
    # surr2: 截断后的目标
    surr1 = ratio * advantages
    surr2 = ratio_clipped * advantages
    
    # PPO 损失：取两者的最小值
    # 这保证了策略更新不会超出截断范围
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 监控：计算被截断的样本比例
    clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()
    
    return policy_loss, clip_fraction


def ppo_value_loss(values, returns, clip_epsilon=0.2):
    """
    计算 PPO 价值函数损失（带截断）
    
    价值函数截断防止价值估计变化过大
    
    Args:
        values: 当前价值估计 [B]
        returns: 回报目标 [B]
        clip_epsilon: 截断参数
        
    Returns:
        value_loss: 价值损失（标量）
    """
    # 未截断的损失
    value_loss_unclipped = F.mse_loss(values, returns)
    
    # 截断的价值估计
    # 防止价值函数变化超过 clip_epsilon
    values_clipped = torch.clamp(
        values,
        returns - clip_epsilon,
        returns + clip_epsilon
    )
    
    # 截断后的损失
    value_loss_clipped = F.mse_loss(values_clipped, returns)
    
    # 取两者最大值（保守估计）
    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
    
    return value_loss


class PPOUpdate:
    """
    PPO 更新管理器
    
    封装完整的 PPO 更新流程
    """
    
    def __init__(self, actor_critic, optimizer, clip_epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        """
        初始化 PPO 更新器
        
        Args:
            actor_critic: Actor-Critic 网络
            optimizer: 优化器
            clip_epsilon: PPO 截断参数
            value_coef: 价值函数损失权重
            entropy_coef: 熵正则化权重
            max_grad_norm: 梯度裁剪范数
        """
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def update(self, states, actions, old_log_probs, returns, advantages, n_epochs=10, batch_size=64):
        """
        执行 PPO 更新
        
        Args:
            states: 状态序列 [T]
            actions: 动作序列 [T]
            old_log_probs: 旧策略的对数概率 [T]
            returns: 回报估计 [T]
            advantages: 优势估计 [T]
            n_epochs: PPO 更新轮数
            batch_size: 批次大小
            
        Returns:
            stats: 训练统计信息
        """
        # 标准化优势（重要技巧！）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'clip_fraction': 0}
        
        # 多次 PPO 更新
        for _ in range(n_epochs):
            # 打乱数据
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # 前向传播
                dist = self.actor_critic.get_distribution(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                values = self.actor_critic.get_value(batch_states)
                
                # 计算损失
                policy_loss, clip_frac = ppo_clip_loss(
                    batch_old_log_probs, new_log_probs, batch_advantages, self.clip_epsilon
                )
                value_loss = ppo_value_loss(values, batch_returns, self.clip_epsilon)
                
                # 熵正则化（鼓励探索）
                entropy_loss = -entropy
                
                # 总损失
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 累积统计
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.item()
                stats['clip_fraction'] += clip_frac.item()
        
        # 平均统计
        n_updates = n_epochs * (dataset_size // batch_size)
        for key in stats:
            stats[key] /= n_updates
        
        return stats
```

### 实现二：DPO 训练器

```python
"""
DPO (Direct Preference Optimization) 训练器

直接从偏好数据优化策略，无需奖励模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOTrainer:
    """
    DPO 训练器
    
    核心公式：
    L_DPO = -E[log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)))]
    """
    
    def __init__(self, policy_model, ref_model, beta=0.1, lr=1e-5):
        """
        初始化 DPO 训练器
        
        Args:
            policy_model: 要优化的策略模型（LLM）
            ref_model: 参考模型（通常是 SFT 模型，冻结参数）
            beta: 温度参数，控制偏离参考策略的惩罚
                  - 太大：策略更新保守
                  - 太小：可能偏离参考策略太多
                  - 推荐：0.1-0.5
            lr: 学习率
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
        
        # 冻结参考模型
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
    
    def compute_log_probs(self, model, input_ids, output_ids):
        """
        计算生成序列的对数概率
        
        Args:
            model: 语言模型
            input_ids: 输入 token IDs [B, L_in]
            output_ids: 输出 token IDs [B, L_out]
            
        Returns:
            log_probs: 对数概率 [B]
        """
        # 拼接输入和输出
        full_ids = torch.cat([input_ids, output_ids], dim=-1)
        
        # 获取模型输出
        outputs = model(full_ids)
        logits = outputs.logits
        
        # 计算输出部分的对数概率
        # 对于每个位置 t，计算 P(output_t | input, output_{<t})
        log_probs = []
        for i in range(len(input_ids)):
            # 输出部分的起始位置
            start = len(input_ids[i])
            end = len(full_ids[i])
            
            # 提取该样本的 logits
            sample_logits = logits[i, start-1:end-1]  # [L_out, vocab]
            sample_targets = output_ids[i, 1:end]  # [L_out]
            
            # 计算交叉熵（负对数概率）
            loss = F.cross_entropy(sample_logits, sample_targets, reduction='none')
            log_prob = -loss.sum()  # 整个序列的对数概率
            
            log_probs.append(log_prob)
        
        return torch.stack(log_probs)
    
    def dpo_loss(self, input_ids, chosen_ids, rejected_ids):
        """
        计算 DPO 损失
        
        Args:
            input_ids: 输入 prompt [B, L_in]
            chosen_ids: 优选回答 [B, L_out]
            rejected_ids: 劣选回答 [B, L_out]
            
        Returns:
            loss: DPO 损失（标量）
            stats: 统计信息
        """
        # 计算策略模型的对数概率
        log_pi_chosen = self.compute_log_probs(self.policy_model, input_ids, chosen_ids)
        log_pi_rejected = self.compute_log_probs(self.policy_model, input_ids, rejected_ids)
        
        # 计算参考模型的对数概率（no_grad）
        with torch.no_grad():
            log_ref_chosen = self.compute_log_probs(self.ref_model, input_ids, chosen_ids)
            log_ref_rejected = self.compute_log_probs(self.ref_model, input_ids, rejected_ids)
        
        # 计算对数概率比
        log_ratio_chosen = log_pi_chosen - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected
        
        # DPO 损失
        # logits = β * (log_ratio_chosen - log_ratio_rejected)
        # loss = -log σ(logits)
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()
        
        # 统计信息
        stats = {
            'loss': loss.item(),
            'log_ratio_chosen': log_ratio_chosen.mean().item(),
            'log_ratio_rejected': log_ratio_rejected.mean().item(),
            'margin': (log_ratio_chosen - log_ratio_rejected).mean().item(),
            'accuracy': (logits > 0).float().mean().item(),  # 偏好预测准确率
        }
        
        return loss, stats
    
    def train_step(self, input_ids, chosen_ids, rejected_ids):
        """
        单步训练
        
        Args:
            input_ids: 输入 prompt [B, L_in]
            chosen_ids: 优选回答 [B, L_out]
            rejected_ids: 劣选回答 [B, L_out]
            
        Returns:
            stats: 训练统计
        """
        self.policy_model.train()
        
        # 计算损失
        loss, stats = self.dpo_loss(input_ids, chosen_ids, rejected_ids)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return stats
```

（第二部分完成，待续...）
## 🔬 算法对比

### PPO vs SAC vs DPO vs GRPO

| 维度 | PPO | SAC | DPO | GRPO |
|------|-----|-----|-----|------|
| **策略类型** | On-Policy | Off-Policy | Off-Policy | On-Policy |
| **动作空间** | 连续/离散 | 连续 | 离散（文本） | 离散（文本） |
| **样本效率** | 中 | 高 | 高 | 中 |
| **稳定性** | 高 | 高 | 很高 | 高 |
| **需要奖励模型** | 是 | 是 | 否 | 否 |
| **需要价值函数** | 是 | 是 | 否 | 否 |
| **超参数数量** | 多 | 中 | 少 | 中 |
| **适用场景** | 通用 RL、LLM 对齐 | 连续控制 | LLM 偏好优化 | 数学/代码生成 |

### 详细对比分析

**PPO（Proximal Policy Optimization）**

优势：
- 稳定性好，训练曲线平滑
- 适用于各种任务（游戏、机器人、LLM）
- 实现相对简单

劣势：
- 样本效率不高（需要大量采样）
- 超参数多（clip epsilon、GAE lambda、价值系数等）
- 需要价值函数训练

适用场景：
- 有明确奖励函数的任务
- 需要稳定训练的任务
- LLM 对齐（配合奖励模型）

**SAC（Soft Actor-Critic）**

优势：
- 样本效率高（off-policy）
- 自动熵调节，探索充分
- 连续控制任务 SOTA

劣势：
- 只适用于连续动作
- 实现复杂（Twin Q、自动温度）
- 超参数敏感

适用场景：
- 机器人控制
- 连续控制任务
- 需要高效探索的任务

**DPO（Direct Preference Optimization）**

优势：
- 不需要奖励模型
- 实现简单（就是分类损失）
- 训练稳定，超参数少
- 计算效率高

劣势：
- 需要高质量偏好数据
- 参考模型质量影响结果
- 对于开放任务效果有限

适用场景：
- LLM 人类对齐
- 有明确偏好的任务
- 资源有限的场景

**GRPO（Group Relative Policy Optimization）**

优势：
- 不需要奖励模型
- 自动生成学习信号
- 适合有标准答案的任务
- 样本效率较高

劣势：
- 只适用于可评分任务
- 需要生成多个样本
- 开放任务不适用

适用场景：
- 数学问题求解
- 代码生成
- 逻辑推理

---

## 🧪 动手实验

### 实验一：PPO 超参数敏感性分析

**任务描述**：

研究 PPO 关键超参数对训练稳定性的影响。

**参数设置**：

| 组别 | clip_epsilon | 预期现象 |
|------|--------------|----------|
| A | 0.1 | 更新保守，收敛慢但稳定 |
| B | 0.2 | 标准设置，平衡 |
| C | 0.3 | 更新激进，可能震荡 |

**实验步骤**：

1. 在 CartPole-v1 环境上训练 PPO
2. 每组参数运行 5 次独立实验
3. 记录每个 episode 的总奖励
4. 绘制学习曲线（带标准差）

**分析指标**：
- 收敛速度：达到 450 平均奖励的 episode 数
- 稳定性：收敛后奖励的标准差
- 最终性能：最后 100 个 episode 的平均奖励

**预期结果**：
- clip_epsilon=0.2 应该是最佳平衡
- clip_epsilon 太小会导致学习慢
- clip_epsilon 太大会导致训练不稳定

**思考题**：
1. 为什么 clip_epsilon 影响稳定性？
2. 如果任务奖励尺度变化，clip_epsilon 需要调整吗？
3. GAE lambda 参数如何影响训练？

### 实验二：DPO vs PPO 对比

**任务描述**：

在简化 LLM 对齐任务上对比 DPO 和 PPO。

**实验设置**：

用一个小规模语言模型（如 GPT-2 small），在偏好数据集上训练。

| 方法 | 需要组件 | 训练时间 |
|------|----------|----------|
| PPO | 策略模型 + 奖励模型 + 价值模型 | 长 |
| DPO | 策略模型 + 参考模型 | 短 |

**分析指标**：
- 偏好准确率（在测试偏好数据上）
- 训练时间
- GPU 显存占用
- 超参数调优难度

**预期结果**：
- DPO 训练更快，显存占用更低
- PPO 可能需要更多调参
- 两者最终性能接近（取决于任务）

### 实验三：GRPO 在数学问题上的应用

**任务描述**：

实现 GRPO 并在数学问题数据集上测试。

**实现步骤**：

1. 准备数学问题数据集（如 GSM8K 子集）
2. 实现组内采样和评分函数
3. 实现 GRPO 更新逻辑
4. 对比 SFT 基线

**评分函数示例**：

```python
def math_reward(problem, response, ground_truth):
    """
    数学问题评分
    
    Args:
        problem: 问题文本
        response: 模型回答
        ground_truth: 标准答案
        
    Returns:
        reward: 0 或 1
    """
    # 从回答中提取最终答案
    answer = extract_answer(response)
    
    # 比较答案
    if answer == ground_truth:
        return 1.0
    else:
        return 0.0
```

**分析指标**：
- 解题准确率
- 组大小 G 的影响
- 与 SFT 的对比

### 实验四：Offline RL 算法对比

**任务描述**：

在 D4RL 基准上对比 BC、CQL、IQL。

**数据集**：
- halfcheetah-medium-v2
- hopper-medium-v2
- walker2d-medium-v2

**实验设置**：

| 算法 | 关键参数 | 预期性能 |
|------|----------|----------|
| BC | 无 | 基准性能 |
| CQL | alpha=1.0 | 优于 BC |
| IQL | tau=0.7, beta=3.0 | 最优 |

**分析指标**：
- 归一化分数（相对于 expert 和 random）
- 训练稳定性
- 对 OOD 动作的鲁棒性

---

## ❓ 常见问题

### Q1: PPO 为什么比 vanilla Policy Gradient 更稳定？

**A**: 从优化理论角度可以严格分析这个问题。

Vanilla Policy Gradient 的目标函数是：

$$J(\theta) = \mathbb{E}[\log \pi_\theta(a|s) A]$$

这个目标函数的问题在于，它假设数据分布不随 θ 变化。但实际上，策略更新后，采样到的数据也会变化。这导致了一个问题：**梯度估计的方向可能不准确**。

具体来说，如果策略更新过大，新策略采样的数据分布会远离旧策略的分布。这时用旧数据计算的梯度就不再适用于新策略，可能导致性能下降。

PPO 通过截断策略比率，保证了新旧策略的差异不会太大。从优化角度看，这相当于在每次更新时加了一个"信任域"约束，保证了梯度估计的有效性。

数学上可以证明，PPO 的截断目标函数是 TRPO 约束优化问题的一阶近似，但实现更简单。

### Q2: DPO 真的不需要奖励模型吗？那偏好信息从哪里来？

**A**: 这是一个常见的误解。DPO 确实不需要显式的奖励模型，但偏好信息来自**偏好数据集**。

关键区别在于：
- **传统 RLHF**：偏好数据 → 训练奖励模型 → PPO 优化
- **DPO**：偏好数据 → 直接优化策略

DPO 的数学洞察是：最优策略本身就编码了奖励信息。从最大熵 RL 的最优性条件可以推导出：

$$\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(\frac{1}{\beta} r(x, y))$$

这个公式可以反解出 r(x, y)，然后代入 Bradley-Terry 偏好模型，就得到了 DPO 的损失函数。

所以 DPO 不是"不需要偏好信息"，而是"不需要显式的奖励模型"。偏好数据本身就是监督信号。

### Q3: GRPO 和 PPO 的主要区别是什么？

**A**: 核心区别在于**优势函数的计算方式**。

**PPO**：
- 用 GAE 计算优势：$A_t = \sum (\gamma\lambda)^l \delta_{t+l}$
- 需要价值函数 V(s) 来估计 TD 误差
- 适用于有环境交互的场景

**GRPO**：
- 用组内标准化计算优势：$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$
- 不需要价值函数
- 适用于可以批量生成、自动评分的场景

从数学上看，GRPO 的组内标准化相当于用组内均值作为基线：

$$A_i = r_i - \text{mean}(r)$$

这与优势函数的定义 $A(s, a) = Q(s, a) - V(s)$ 是一致的，只是 V(s) 用组内均值代替了。

### Q4: Offline RL 为什么比 Online RL 难？

**A**: 核心挑战是**分布偏移（distribution shift）**。

Online RL 中，智能体可以主动探索，收集新数据。如果策略发现某个状态 - 动作对的 Q 值估计不准确，它可以再次尝试这个动作，获得更准确的估计。

Offline RL 中，数据是固定的。如果数据集没有覆盖某些状态 - 动作对，策略就无法获得这些动作的真实反馈。更糟糕的是，Q 函数的贝尔曼更新会传播误差：

$$Q(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q(s', a')]$$

如果 $Q(s', a')$ 被高估（因为 a'是 OOD 动作），那么 $Q(s, a)$ 也会被高估。这种误差会累积，导致策略选择 OOD 动作。

CQL 和 IQL 的核心思想都是**防止 Q 函数对 OOD 动作的高估**，只是方法不同：
- CQL：显式正则化，惩罚 OOD 动作的 Q 值
- IQL：隐式避免，只学习数据集内的价值

### Q5: 如何选择 PPO、DPO、GRPO？

**A**: 取决于你的任务特点和资源约束。

**选择 PPO 如果**：
- 有明确的奖励函数（或可以训练奖励模型）
- 需要与环境交互
- 任务复杂，需要稳定的训练

**选择 DPO 如果**：
- 有高质量偏好数据
- 想简化训练流程（不要奖励模型）
- 资源有限（计算、时间）

**选择 GRPO 如果**：
- 任务有明确答案（数学、代码）
- 可以自动评分
- 想避免偏好数据收集

**混合策略**：
实际应用中，经常组合使用：
- SFT → DPO：先用监督数据微调，再用偏好数据优化
- SFT → GRPO → DPO：先用 GRPO 提升推理能力，再用 DPO 对齐偏好

---

## 📚 延伸阅读

### 核心论文

1. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
   - https://arxiv.org/abs/1707.06347
   - PPO 原始论文，包含理论分析和实验

2. **SAC**: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
   - https://arxiv.org/abs/1801.01290
   - SAC 原始论文

3. **DPO**: Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
   - https://arxiv.org/abs/2305.18290
   - DPO 原始论文，包含完整推导

4. **GRPO**: Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
   - https://arxiv.org/abs/2402.03300
   - GRPO 在数学推理中的应用

5. **CQL**: Kumar et al. "Conservative Q-Learning for Offline RL" (2020)
   - https://arxiv.org/abs/2006.04779
   - CQL 原始论文

6. **IQL**: Kostrikov et al. "Offline RL with Implicit Q-Learning" (2022)
   - https://arxiv.org/abs/2110.06169
   - IQL 原始论文

### 教程与博客

1. **Spinning Up in Deep RL** (OpenAI)
   - https://spinningup.openai.com
   - PPO、SAC 的详细教程和代码

2. **Lil'Log Policy Gradient**
   - https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
   - 策略梯度方法的深入讲解

3. **DPO 详解** (Hugging Face)
   - https://huggingface.co/blog/dpo-trl
   - DPO 实践指南

### 代码资源

| 算法 | 官方实现 | 第三方实现 |
|------|----------|------------|
| PPO | https://github.com/openai/spinningup | CleanRL |
| SAC | https://github.com/haarnoja/sac | Stable Baselines3 |
| DPO | https://github.com/eric-mitchell/dpo | TRL (Hugging Face) |
| GRPO | DeepSeekMath 代码 | 本仓库 |
| CQL | https://github.com/aviralkumar2907/CQL | d3rlpy |
| IQL | https://github.com/ikostrikov/implicit_q_learning | d3rlpy |

---

## ✅ 本章检查清单

学完本章后，你应该能够：

**概念理解**：
- [ ] 解释 PPO 为什么比 vanilla PG 稳定
- [ ] 解释 DPO 如何绕过奖励模型
- [ ] 解释 GRPO 的组内比较机制
- [ ] 解释 Offline RL 的分布外挑战

**数学推导**：
- [ ] 推导 PPO 的截断目标函数
- [ ] 推导 GAE 的简化形式
- [ ] 推导 DPO 的损失函数
- [ ] 推导 GRPO 的优势标准化

**代码实现**：
- [ ] 实现 PPO 核心更新（含 GAE）
- [ ] 实现 DPO 训练器
- [ ] 实现 GRPO 组内采样
- [ ] 实现 CQL 正则化

**实验分析**：
- [ ] 完成 PPO 超参数敏感性实验
- [ ] 对比 DPO 和 PPO 的性能
- [ ] 在数学任务上测试 GRPO
- [ ] 在 D4RL 上测试 Offline RL 算法

**应用判断**：
- [ ] 根据任务选择合适的算法
- [ ] 合理设置超参数
- [ ] 诊断训练中的问题

---

## 📝 课后测验（10 分钟）

### 基础题（必答）

**1. PPO 为什么比 vanilla Policy Gradient 更稳定？**

<details>
<summary>点击查看答案</summary>

**答案**：

PG 的问题：更新步长难以控制，可能性能崩溃。

PPO 的解决：截断策略比率
- r(θ) = π_θ(a|s) / π_old(a|s)
- 如果 r 超出 [1-ε, 1+ε]，截断
- 保证更新幅度可控
</details>

---

**2. DPO 如何绕过奖励模型？**

<details>
<summary>点击查看答案</summary>

**答案**：

传统 RLHF：偏好数据 → 奖励模型 → PPO

DPO 的洞察：最优策略本身编码了奖励信息
- π*(y|x) ∝ π_ref(y|x) * exp(r(x,y)/β)
- 反解出 r(x,y)
- 直接优化策略，不需要显式奖励模型
</details>

---

**3. GRPO 的核心思想是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

对同一问题生成 G 个答案：
1. 用规则评分（如数学题答案对错）
2. 组内标准化：A_i = (r_i - mean(r)) / std(r)
3. PPO 风格更新

优势：无需奖励模型，自动产生学习信号。
</details>

---

### 进阶题（选答）

**4. Offline RL 的核心挑战是什么？**

<details>
<summary>点击查看答案</summary>

**答案**：

**分布外（OOD）动作**：
- 数据集没覆盖的状态 - 动作对
- Q 函数外推不准确
- 策略选择 OOD 动作 → 恶性循环

**解决方案**：
- CQL：惩罚 OOD 动作的 Q 值
- IQL：只学习数据集内的价值
</details>

---

**5. 如何选择 PPO、DPO、GRPO？**

<details>
<summary>点击查看答案</summary>

**答案**：

| 场景 | 推荐 | 原因 |
|------|------|------|
| 有奖励函数 | PPO | 标准方法 |
| 有偏好数据 | DPO | 简化流程 |
| 有标准答案 | GRPO | 自动评分 |
| 开放任务 | PPO/DPO | 需要人类反馈 |
</details>

---

### 编程题（实践）

**6. 用 DPO 训练一个简单的文本偏好模型**

<details>
<summary>查看提示</summary>

**提示**：
1. 准备偏好数据（prompt, chosen, rejected）
2. 用小模型（如 GPT-2 small）
3. 实现 DPO 损失
4. 对比 SFT 基线
</details>

---

## 🚀 下一步

学完本章，你已经掌握了：
- ✅ 经典 RL 基础（第 1-4 章）
- ✅ 深度 RL 方法（第 5-6 章）
- ✅ 高级策略优化（本章）

**可以继续深入**：
1. 阅读 LLM 对齐领域的前沿论文
2. 在真实 LLM 上实验 DPO/GRPO
3. 探索新的后训练方法

**回顾与巩固**：
- 重做各章实验
- 实现算法变体
- 应用到自己的项目

---

*最后更新：2026-04-22*  
*作者：Hermes <neko_yukirin@qq.com>*
