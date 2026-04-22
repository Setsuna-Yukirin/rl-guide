# 第 5 章：函数近似 (Function Approximation)

## 📚 学习目标

- 理解为什么需要函数近似
- 掌握线性函数近似方法
- 掌握神经网络 Q 函数 (DQN 基础)
- 掌握 DQN 算法及其变体
- 通过 CartPole、Breakout、Lunar Lander 实践深度 RL

## 🔑 核心概念

### 1. 为什么需要函数近似

**表格型方法的局限**：
- 状态空间过大（如 Atari 游戏：210x160x3 像素）
- 无法泛化到未见过的状态
- 内存需求过大

**函数近似优势**：
- 可以处理连续状态空间
- 能够泛化到相似状态
- 内存效率高

### 2. 线性函数近似

**价值函数表示**：
```
V(s, w) = w^T * φ(s) = Σ w_i * φ_i(s)
```

其中：
- `w`: 权重向量
- `φ(s)`: 状态特征向量

**TD 更新规则**：
```
w ← w + α * [R + γ*V(s', w) - V(s, w)] * ∇V(s, w)
```

对于线性近似：
```
w ← w + α * δ * φ(s)
```

### 3. 神经网络 Q 函数

**DQN 核心思想**：
- 用神经网络近似 Q 函数：Q(s, a; θ) ≈ Q*(s, a)
- 输入：状态（如图像）
- 输出：每个动作的 Q 值

**损失函数**：
```
L(θ) = E[(r + γ*max_a' Q(s', a'; θ) - Q(s, a; θ))^2]
```

### 4. DQN 关键技术

**经验回放 (Experience Replay)**：
- 存储转移 (s, a, r, s') 到回放缓冲区
- 随机采样打破相关性
- 提高数据效率

**目标网络 (Target Network)**：
- 使用独立的网络计算目标 Q 值
- 定期更新目标网络参数
- 提高训练稳定性

**DQN 更新**：
```
θ ← θ + α * [r + γ*max_a' Q(s', a'; θ^-) - Q(s, a; θ)] * ∇Q(s, a; θ)
```

### 5. DQN 变体

**Double DQN**：
- 解决 Q-Learning 的最大值偏差
- 用当前网络选择动作，目标网络评估

**Dueling DQN**：
- 分离状态价值和优势函数
- Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

**Prioritized Experience Replay**：
- 优先采样 TD 误差大的转移
- 提高学习效率

## 📁 文件结构

```
chapter_05_function_approximation/
├── README.md                    # 本章文档
├── __init__.py                  # 模块初始化
├── linear_approximation.py      # 线性函数近似
├── neural_network_q.py          # 神经网络 Q 函数
├── dqn.py                       # DQN 算法及变体
└── games/
    ├── __init__.py
    ├── cartpole_balance.py      # CartPole 环境
    ├── breakout_atari.py        # Breakout 游戏
    └── lunar_lander.py          # 月球着陆器
```

## 🎮 实践项目

### 项目 1：CartPole Balance

**环境描述**：
- 小车在轨道上移动
- 目标：保持杆子平衡
- 状态：(位置，速度，角度，角速度)
- 动作：向左/向右推

**学习目标**：
- 使用 DQN 学习平衡策略
- 理解连续状态空间的处理

### 项目 2：Breakout Atari

**环境描述**：
- 经典 Atari 打砖块游戏
- 输入：游戏画面（像素）
- 动作：左移/右移/发射

**学习目标**：
- 从像素输入学习
- 理解卷积神经网络在 RL 中的应用

### 项目 3：Lunar Lander

**环境描述**：
- 控制登月舱安全着陆
- 状态：位置、速度、角度、燃料等
- 动作：主引擎/侧向引擎

**学习目标**：
- 处理混合状态空间
- 学习精细控制策略

## 🔬 实验练习

1. 比较表格型 Q-Learning 和 DQN 在 CartPole 上的表现
2. 分析经验回放缓冲区大小对训练的影响
3. 实现 Double DQN 并与原始 DQN 比较
4. 研究目标网络更新频率的影响
5. 实现 Dueling DQN 架构

## 📖 参考资料

- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Sutton & Barto, Chapter 9: On-policy Prediction with Approximation
- Van Hasselt et al. (2016). Deep reinforcement learning with double Q-learning.
- Wang et al. (2016). Dueling network architectures for deep reinforcement learning.
