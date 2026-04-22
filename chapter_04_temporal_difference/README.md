# 第 4 章：时序差分学习 (Temporal Difference Learning)

## 📚 学习目标

- 理解时序差分 (TD) 学习的核心思想
- 掌握 TD(0) 预测算法
- 掌握 Q-Learning (离策略 TD 控制)
- 掌握 SARSA (在策略 TD 控制)
- 理解 Expected SARSA
- 比较在策略 vs 离策略学习
- 通过 Cliff Walking 和 Windy Gridworld 实践 TD 算法

## 🔑 核心概念

### 1. TD 学习基础

**核心思想**：
- 结合动态规划和蒙特卡洛的优点
- **自举 (Bootstrapping)**：用后续估计更新当前估计
- **采样**：从环境采样，不需要完整模型
- **单步更新**：每步都可以学习，无需等待 episode 结束

**TD 与 MC、DP 的比较**：

| 特性 | 动态规划 | 蒙特卡洛 | 时序差分 |
|------|----------|----------|----------|
| 环境模型 | 需要 | 不需要 | 不需要 |
| 更新时机 | 每步 | 回合结束 | 每步 |
| 自举 | ✓ | ✗ | ✓ |
| 采样 | ✗ | ✓ | ✓ |
| 方差 | 低 | 高 | 中等 |
| 偏差 | 有偏 | 无偏 | 有偏 |

### 2. TD(0) 预测

**目标**：估计给定策略 π 的价值函数 V_π

**TD 误差**：
```
δ_t = R_{t+1} + γ * V(S_{t+1}) - V(S_t)
```

**更新规则**：
```
V(S_t) ← V(S_t) + α * δ_t
```

其中：
- `α`: 学习率 (step size)
- `γ`: 折扣因子
- `δ_t`: TD 误差

**TD(0) 算法**：
```
初始化 V(s) 为任意值
对于每个 episode:
    初始化状态 S
    重复：
        A ← π(S)
        执行 A，观察 R, S'
        V(S) ← V(S) + α * [R + γ*V(S') - V(S)]
        S ← S'
        如果 S 是终止状态则退出
```

### 3. SARSA (在策略 TD 控制)

**名称来源**：State-Action-Reward-State-Action

**核心思想**：
- 在策略学习：用当前策略生成数据，同时更新该策略
- 学习的是实际执行的策略的价值

**更新规则**：
```
Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ*Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

**SARSA 算法**：
```
初始化 Q(s, a) 为任意值
对于每个 episode:
    初始化 S
    选择 A ← 策略 (如 ε-greedy)
    重复：
        执行 A，观察 R, S'
        选择 A' ← 策略 (S')
        Q(S, A) ← Q(S, A) + α * [R + γ*Q(S', A') - Q(S, A)]
        S ← S', A ← A'
        如果 S 是终止状态则退出
```

**特点**：
- 保守学习：考虑了探索动作的影响
- 更安全：在 Cliff Walking 等环境中表现更谨慎

### 4. Q-Learning (离策略 TD 控制)

**核心思想**：
- 离策略学习：行为策略 (探索) 和目标策略 (贪婪) 可以不同
- 直接学习最优策略，不考虑探索动作

**更新规则**：
```
Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ*max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

**Q-Learning 算法**：
```
初始化 Q(s, a) 为任意值
对于每个 episode:
    初始化 S
    重复：
        选择 A ← 策略 (如 ε-greedy)
        执行 A，观察 R, S'
        Q(S, A) ← Q(S, A) + α * [R + γ*max_a Q(S', a) - Q(S, A)]
        S ← S'
        如果 S 是终止状态则退出
```

**特点**：
- 激进学习：总是假设后续选择最优动作
- 收敛到最优策略（满足 GLIE 条件）
- 可能过度乐观

### 5. Expected SARSA

**核心思想**：
- 使用期望值代替最大值，减少方差
- 介于 SARSA 和 Q-Learning 之间

**更新规则**：
```
Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ*E[Q(S_{t+1}, A_{t+1})] - Q(S_t, A_t)]
```

其中期望值：
```
E[Q(S_{t+1}, A_{t+1})] = Σ_a π(a|S_{t+1}) * Q(S_{t+1}, a)
```

对于 ε-greedy 策略：
```
E[Q] = (1-ε)*max_a Q(S', a) + (ε/|A|)*Σ_a Q(S', a)
```

**特点**：
- 方差比 Q-Learning 小
- 计算成本稍高（需要求和）
- 性能通常优于 SARSA 和 Q-Learning

### 6. 在策略 vs 离策略

| 特性 | 在策略 (SARSA) | 离策略 (Q-Learning) |
|------|----------------|---------------------|
| 学习策略 | 当前行为策略 | 最优策略 |
| 探索影响 | 考虑在内 | 不考虑 |
| 安全性 | 更安全 | 可能冒险 |
| 收敛速度 | 较慢 | 较快 |
| 适用场景 | 安全关键 | 追求最优 |

### 7. n 步 TD 方法

**n 步回报**：
```
G_{t:t+n} = R_{t+1} + γ*R_{t+2} + ... + γ^{n-1}*R_{t+n} + γ^n*V(S_{t+n})
```

**TD(λ)**：
- 指数加权平均所有 n 步回报
- 资格迹 (Eligibility Trace) 实现

## 📁 文件结构

```
chapter_04_temporal_difference/
├── README.md                    # 本章文档
├── __init__.py                  # 模块初始化
├── td_prediction.py             # TD(0) 预测算法
├── q_learning.py                # Q-Learning 算法
├── sarsa.py                     # SARSA 算法
├── expected_sarsa.py            # Expected SARSA
└── games/
    ├── __init__.py
    ├── cliff_walking.py         # Cliff Walking 环境
    └── windy_gridworld.py       # Windy Gridworld 环境
```

## 🎮 实践项目

### 项目 1：Cliff Walking

**环境描述**：
- 4x12 网格
- 起点：左下角
- 终点：右下角
- 悬崖：底部一行（除起点终点）
- 掉下悬崖：奖励 -100，回到起点
- 每步奖励：-1

**学习目标**：
- 比较 SARSA 和 Q-Learning 的行为差异
- SARSA 学会绕远路（安全）
- Q-Learning 学会走最短路径（冒险）

### 项目 2：Windy Gridworld

**环境描述**：
- 7x10 网格
- 某些列有向上的风
- 风强度不同
- 目标：从起点到终点

**学习目标**：
- 学习在有外力干扰下的最优策略
- 理解状态转移的不确定性

## 🔬 实验练习

1. 比较 SARSA 和 Q-Learning 在 Cliff Walking 上的表现
2. 分析不同 ε 值对两种算法的影响
3. 实现 Expected SARSA 并与 SARSA、Q-Learning 比较
4. 研究学习率 α 对收敛的影响
5. 实现 n 步 SARSA

## 📖 参考资料

- Sutton & Barto, Chapter 6: Temporal-Difference Learning
- Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning.
