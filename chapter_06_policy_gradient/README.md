# 第 6 章：策略梯度 (Policy Gradient)

## 📚 学习目标

- 理解策略梯度方法的核心思想
- 掌握 REINFORCE 算法（蒙特卡洛策略梯度）
- 掌握 Actor-Critic 架构
- 掌握 A2C（Advantage Actor-Critic）
- 掌握 DDPG（深度确定性策略梯度）
- 掌握 TD3（Twin Delayed DDPG）
- 通过赛车、机械臂、Pong 游戏实践连续控制

## 🔑 核心概念

### 1. 策略梯度基础

**核心思想**：
- 直接优化策略函数 π(a|s; θ)
- 适用于连续动作空间
- 可以学习随机策略

**策略梯度定理**：
```
∇J(θ) = E[∇log π(a|s; θ) * Q(s,a)]
```

### 2. REINFORCE 算法

**蒙特卡洛策略梯度**：
```
∇J(θ) ≈ E[∇log π(a|s; θ) * G_t]
```

**更新规则**：
```
θ ← θ + α * ∇log π(a|s; θ) * G_t
```

**特点**：
- 无偏估计
- 高方差
- 需要完整的 episode

### 3. Actor-Critic 架构

**核心思想**：
- Actor：策略网络 π(a|s; θ)
- Critic：价值网络 V(s; w) 或 Q(s,a; w)
- 用 Critic 降低方差

**优势函数**：
```
A(s,a) = Q(s,a) - V(s)
```

**更新规则**：
```
Actor: θ ← θ + α * ∇log π(a|s; θ) * A(s,a)
Critic: w ← w + β * ∇[G - V(s; w)]²
```

### 4. A2C (Advantage Actor-Critic)

**同步优势 Actor-Critic**：
- 使用 n 步回报估计优势
- 同步更新多个并行环境

**n 步回报**：
```
G_t = R_t + γ*R_{t+1} + ... + γ^n*V(s_{t+n})
```

**优势估计**：
```
A_t = G_t - V(s_t)
```

### 5. DDPG (Deep Deterministic Policy Gradient)

**核心思想**：
- 确定性策略：a = μ(s; θ)
- 适用于连续控制
- 基于 Actor-Critic

**关键技术**：
- 经验回放
- 目标网络（Actor 和 Critic 都有）
- 软更新（Polyak 平均）

**更新规则**：
```
Critic: w ← w + α * [r + γ*Q(s', μ(s'); w') - Q(s,a; w)] * ∇Q
Actor: θ ← θ + α * ∇Q(s, μ(s); w) * ∇μ(s; θ)
```

### 6. TD3 (Twin Delayed DDPG)

**DDPG 的改进**：
1. **Clipped Double Q-Learning**：两个 Critic，取最小值
2. **Delayed Policy Updates**：Critic 更新多次后再更新 Actor
3. **Target Policy Smoothing**：在目标动作上加噪声

**优势**：
- 减少过估计偏差
- 更稳定的训练
- 更好的最终性能

## 📁 文件结构

```
chapter_06_policy_gradient/
├── README.md                    # 本章文档
├── __init__.py                  # 模块初始化
├── reinforce.py                 # REINFORCE 算法
├── actor_critic.py              # Actor-Critic 基础
├── a2c.py                       # A2C 算法
├── ddpg.py                      # DDPG 算法
├── td3.py                       # TD3 算法
└── games/
    ├── __init__.py
    ├── car_racing.py            # 赛车环境
    ├── robotic_arm.py           # 机械臂环境
    └── pong_simple.py           # Pong 游戏
```

## 🎮 实践项目

### 项目 1：Car Racing

**环境描述**：
- 2D 赛车游戏
- 连续控制：转向、油门、刹车
- 目标：最快完成赛道

**学习目标**：
- 使用 DDPG/TD3 学习连续控制
- 处理高维状态（图像或传感器）

### 项目 2：Robotic Arm

**环境描述**：
- 机械臂控制
- 状态：关节角度、角速度
- 动作：关节力矩
- 目标：抓取目标物体

**学习目标**：
- 精确的连续控制
- 多关节协调

### 项目 3：Pong Simple

**环境描述**：
- 简化版乒乓球游戏
- 动作：上/下移动
- 目标：击败对手

**学习目标**：
- 使用策略梯度学习对抗策略
- 理解自博弈概念

## 🔬 实验练习

1. 比较 REINFORCE 和 Actor-Critic 的收敛速度
2. 分析优势函数对方差的影响
3. 实现 A2C 并比较与 DDPG 的表现
4. 研究 TD3 相对于 DDPG 的改进
5. 在 CarRacing 上比较不同算法

## 📖 参考资料

- Sutton & Barto, Chapter 13: Policy Gradient Methods
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
- Mnih et al. (2016). Asynchronous methods for deep reinforcement learning.
- Lillicrap et al. (2016). Continuous control with deep reinforcement learning.
- Fujimoto et al. (2018). Addressing function approximation error in actor-critic methods.
