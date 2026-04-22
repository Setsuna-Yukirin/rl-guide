# 01 - 项目概述

## 📌 项目信息

| 项目 | 说明 |
|------|------|
| **名称** | rl-guide（强化学习指北） |
| **版本** | v0.1.0 |
| **作者** | Hermes <neko_yukirin@qq.com> |
| **许可证** | MIT |
| **GitHub** | https://github.com/Setsuna-Yukirin/rl-guide |

---

## 🎯 项目目标

### 核心目标

构建一个**游戏化的强化学习教程仓库**，帮助从 LLM 后训练（GRPO/DPO/PPO）入门的学习者**回溯理解 RL 底层原理**。

### 目标用户

1. **LLM 研究者/工程师** - 理解 PPO/DPO/GRPO 的 RL 基础
2. **RL 初学者** - 从零开始系统学习强化学习
3. **学生** - 配合大学课程学习
4. **工程师** - 将 RL 应用到实际项目

### 学习成果

完成本教程后，学习者应该能够：

- ✅ 理解 MDP、贝尔曼方程等 RL 数学基础
- ✅ 掌握经典算法（DP、MC、TD、Q-Learning 等）
- ✅ 理解深度 RL 算法（DQN、A2C、PPO 等）
- ✅ 理解连续控制算法（DDPG、TD3、SAC）
- ✅ 理解 PPO/DPO/GRPO 的 RL 原理
- ✅ 能够独立实现 RL 算法
- ✅ 能够将 RL 应用到实际问题

---

## 🎯 核心理念

### 1. 游戏化学习 🎮

**问题**：传统 RL 教程枯燥，难以坚持

**解决方案**：每章都有可玩的小游戏

| 章节 | 游戏示例 | 学习概念 |
|------|---------|---------|
| 第 1 章 | 🍱 午餐选择器 | MDP 建模 |
| 第 2 章 | 🗺️ 网格寻路 | 最优路径规划 |
| 第 3 章 | 🃏 21 点游戏 | MC 学习 |
| 第 4 章 | 🏃 悬崖行走 | SARSA vs Q-Learning |
| 第 5 章 | 🤹 CartPole | DQN 训练 |
| 第 6 章 | 🚗 赛车控制 | 连续控制 |
| 第 7 章 | 🤖 对话优化 | PPO + LLM |

### 2. 模块化设计 📦

**问题**：代码耦合，难以理解

**解决方案**：每个算法一个独立的 class

```python
# 示例：QLearning 类
class QLearning:
    """Q-Learning 算法实现"""
    
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.99):
        ...
    
    def update(self, state, action, reward, next_state, done):
        ...
    
    def get_action(self, state, epsilon=0.1):
        ...
```

**优点**：
- 每个 class 控制在 200 行以内
- 易于理解和复用
- 可以单独测试

### 3. 可视化优先 📊

**问题**：算法行为抽象，难以理解

**解决方案**：大量图表和动画

- 价值函数热力图
- 策略箭头图
- 学习曲线
- 训练过程动画

### 4. 连接前沿 🔗

**问题**：经典 RL 与现代 LLM 后训练脱节

**解决方案**：第 7 章专门建立连接

```
经典 RL          现代 LLM 后训练
    ↓                  ↓
PPO (CartPole)  →  PPO (LLM 生成)
    ↓                  ↓
价值函数        →   奖励模型
    ↓                  ↓
策略优化        →   DPO/GRPO
```

### 5. 中文友好 📖

**问题**：优质中文 RL 教程少

**解决方案**：
- 完整中文注释
- 生活化类比
- 中文文档

---

## 🔧 技术栈

### 核心依赖

```python
# 数值计算
numpy>=1.24.0

# RL 环境
gymnasium>=0.29.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 深度学习
torch>=2.0.0

# 游戏渲染
pygame>=2.0.0

# 交互式学习
ipywidgets>=8.0.0
jupyter>=1.0.0

# 开发工具
pytest>=7.0.0
black>=23.0.0
tqdm>=4.65.0
```

### 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.9+ |
| CUDA | 11.7+ (可选，推荐) |
| GPU | 推荐 NVIDIA RTX 3080 或更高 |
| 内存 | 最低 8GB，推荐 16GB+ |
| 存储 | 最低 5GB，推荐 10GB+ |

### 硬件利用策略

**GPU**: NVIDIA RTX 3080

- ✅ 深度学习模型训练（DQN、PPO 等）
- ✅ 批量环境并行采样
- ✅ 可视化渲染加速

**注意事项**：
- 训练参数可调，支持快速验证模式
- 默认配置：少跑几轮，确保能跑通
- 正式训练：可调整参数充分利用 GPU

```python
# 示例：可调参数
config = {
    "num_episodes": 100,      # 快速验证用 100，正式训练用 1000+
    "batch_size": 64,         # 根据显存调整
    "use_gpu": True,          # 可关闭 GPU 测试
}
```

---

## 📁 项目结构

```
rl-guide/
├── docs/                        # 架构书文档
│   ├── INDEX.md                 # 文档索引
│   ├── 01_architecture/         # 架构设计
│   ├── 02_chapters/             # 章节详细设计
│   ├── 03_api_reference/        # API 参考
│   ├── 04_tutorials/            # 教程
│   └── 05_design_decisions/     # 设计决策
│
├── chapter_01_mdp_fundamentals/ # 第 1 章
├── chapter_02_dynamic_programming/ # 第 2 章
├── chapter_03_monte_carlo/      # 第 3 章
├── chapter_04_temporal_difference/ # 第 4 章
├── chapter_05_function_approximation/ # 第 5 章
├── chapter_06_policy_gradient/  # 第 6 章
├── chapter_07_advanced_policy/  # 第 7 章
│
├── utils/                       # 工具函数
├── tests/                       # 测试
├── notebooks/                   # Jupyter 笔记本
│
├── README.md                    # 项目说明
├── requirements.txt             # 依赖
└── .gitignore                   # Git 忽略
```

---

## 📋 开发原则

### 代码质量

1. **每个算法一个 class** - 模块化，易理解
2. **200 行以内** - 保持简洁
3. **完整 docstring** - Google 风格
4. **类型注解** - 提高可读性
5. **行内注释** - 解释复杂逻辑

### 测试要求

1. **测试先行** - 先写测试再写代码
2. **回归测试** - 确保现有功能正常
3. **功能测试** - 验证新功能
4. **覆盖率** - 核心算法 80%+ 覆盖率

### 文档要求

1. **每章 README** - 学习目标、前置知识
2. **数学推导** - 关键公式
3. **代码示例** - 可直接运行
4. **可视化** - 图表辅助理解

### 提交规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

详见 [INDEX.md](INDEX.md#提交规范)

---

## 🚀 里程碑

| 版本 | 内容 | 预计时间 |
|------|------|---------|
| v0.1.0 | 架构书 + 基础框架 | 2026-04-22 |
| v0.2.0 | 第 1-2 章完成 | 2026-04-25 |
| v0.3.0 | 第 3-4 章完成 ⭐ | 2026-04-30 |
| v0.4.0 | 第 5-6 章完成 | 2026-05-05 |
| v0.5.0 | 第 7 章完成 ⭐ | 2026-05-10 |
| v1.0.0 | 正式发布 | 2026-05-15 |

---

## 📞 反馈与支持

- **Bug 报告**: https://github.com/Setsuna-Yukirin/rl-guide/issues
- **功能建议**: https://github.com/Setsuna-Yukirin/rl-guide/discussions
- **邮件**: neko_yukirin@qq.com

---

*文档版本：v0.1.0*  
*最后更新：2026-04-22*
