# 🎮 Hermes - 强化学习指北

> **从经典 RL 算法到 LLM 后训练的完整学习路径**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 📌 项目简介

**rl-guide** 是一个游戏化的强化学习教程仓库，帮助你从经典 RL 算法打底，系统理解 GRPO/DPO/PPO 等 LLM 后训练技术的底层原理。

### 🎯 核心理念

- **🎮 游戏化学习** - 每章都有可玩的小游戏，边玩边学
- **📦 模块化设计** - 每个算法一个独立的 class，便于理解和复用
- **📊 可视化优先** - 大量图表和动画帮助理解算法行为
- **🔗 连接前沿** - 从经典 RL 一直连接到 LLM 后训练（GRPO/DPO/PPO）
- **📖 中文友好** - 完整的中文解释和生活化类比

---

## 📚 学习路线

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        强化学习学习路径                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  基础篇          核心篇 ⭐           进阶篇              前沿篇 ⭐           │
│  ┌─────┐        ┌──────────┐       ┌──────────┐       ┌──────────┐        │
│  │第 1 章│   →    │第 2 章     │   →   │第 3-4 章   │   →   │第 5-7 章   │        │
│  │ MDP │        │ 动态规划  │       │ 蒙特卡洛  │       │ 函数近似  │        │
│  └─────┘        └──────────┘       │ 时序差分  │       │ 策略梯度  │        │
│                                     └──────────┘       │ 高级策略  │        │
│                                                         └──────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 章节详情

| 章节 | 主题 | 核心算法 | 游戏/应用 | 难度 | 前置 |
|------|------|----------|-----------|------|------|
| **第 1 章** | MDP 基础 | Bellman 方程、价值函数、策略迭代 | 🍱 午餐选择、🚇 路线规划 | ⭐ | 无 |
| **第 2 章** | 动态规划 | Policy Iteration、Value Iteration、MCTS | 🗺️ 网格寻路、🎯 AlphaGo | ⭐⭐ | 第 1 章 |
| **第 3 章** | 蒙特卡洛 | MC Prediction、MC Control、重要性采样 | 🃏 21 点、🎰 老虎机 | ⭐⭐ | 第 1 章 |
| **第 4 章** | 时序差分 ⭐ | SARSA、Q-Learning、Expected SARSA、6 种探索策略 | 🏃 悬崖行走、💎 迷宫、🐍 贪吃蛇 | ⭐⭐⭐ | 第 3 章 |
| **第 5 章** | 函数近似 | 线性近似、DQN、目标网络、经验回放 | 🤹 CartPole、🕹️ 打砖块、🚀 登月器 | ⭐⭐⭐ | 第 4 章 |
| **第 6 章** | 策略梯度 | REINFORCE、A2C、DDPG、TD3 | 🚗 赛车、🎯 机械臂、🎮 乒乓球 | ⭐⭐⭐⭐ | 第 5 章 |
| **第 7 章** | 高级策略 ⭐ | **PPO**、**SAC**、**DPO**、**GRPO**、Offline RL | 🤖 对话优化、📝 文本生成 | ⭐⭐⭐⭐⭐ | 第 6 章 |

### 关键里程碑

| 阶段 | 章节 | 学完后你能... | 预计时间 |
|------|------|---------------|----------|
| **🌱 入门** | 第 1-2 章 | 用 MDP 建模实际问题，理解最优策略 | 1-2 周 |
| **🌿 基础** | 第 3-4 章 | 实现经典 RL 算法，理解探索 - 利用权衡 | 2-3 周 |
| **🌳 进阶** | 第 5-6 章 | 训练深度 RL 智能体，处理连续控制 | 3-4 周 |
| **🚀 前沿** | 第 7 章 | 理解 LLM 后训练（DPO/GRPO）的 RL 原理 | 2-3 周 |

---

## 📁 目录结构

```
rl-guide/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── pyproject.toml                    # 项目配置
├── Makefile                          # 快捷命令
│
├── chapter_01_mdp_fundamentals/      # 第 1 章：MDP 基础
├── chapter_02_dynamic_programming/   # 第 2 章：动态规划
├── chapter_03_monte_carlo/           # 第 3 章：蒙特卡洛
├── chapter_04_temporal_difference/   # 第 4 章：时序差分 ⭐
├── chapter_05_function_approximation/# 第 5 章：函数近似
├── chapter_06_policy_gradient/       # 第 6 章：策略梯度
├── chapter_07_advanced_policy/       # 第 7 章：高级策略 ⭐
│   ├── ppo.py                        # PPO 算法
│   ├── sac.py                        # SAC 算法
│   ├── dpo.py                        # DPO（LLM 对齐）
│   ├── grpo.py                       # GRPO（推理优化）
│   └── offline_rl.py                 # Offline RL（BC/CQL/IQL）
│
├── rl_guide/                         # 核心库
│   ├── core/                         # 核心抽象
│   │   ├── mdp.py                    # MDP 五元组
│   │   ├── policy.py                 # 策略基类
│   │   ├── value_function.py         # 价值函数
│   │   └── replay_buffer.py          # 经验回放
│   └── utils/                        # 工具
│       ├── visualization.py          # 可视化
│       └── training_loop.py          # 训练循环
│
├── tests/                            # 单元测试（171 个测试）
│   ├── test_core.py
│   ├── test_utils.py
│   └── test_chapter_*.py
│
└── docs/                             # 文档
    └── 01_architecture/              # 架构设计
```

### 每章结构

每章都包含：
- 📖 `README.md` - 算法原理、数学公式、使用示例
- 🐍 `*.py` - 模块化算法实现（每个算法 ~200 行）
- 🎮 `games/` - 游戏化应用场景
- ✅ `tests/` - 单元测试

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 推荐：conda 或 venv 虚拟环境

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Setsuna-Yukirin/Hermes.git
cd Hermes

# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行第一个示例

```bash
# 第 1 章：午餐选择器
python chapter_01_mdp_fundamentals/games/lunch_decision.py

# 第 3 章：21 点游戏
python chapter_03_monte_carlo/games/blackjack.py

# 第 4 章：悬崖行走对比
python chapter_04_temporal_difference/games/cliff_walking.py
```

---

## 📖 学习建议

### 推荐学习顺序

1. **基础阶段**（第 1-2 章）
   - 理解 MDP 五元组
   - 掌握贝尔曼方程
   - 学会动态规划方法
   - 了解 MCTS 搜索

2. **核心阶段**（第 3-4 章）⭐
   - 蒙特卡洛 vs 时序差分
   - SARSA vs Q-Learning 的区别
   - 探索策略的重要性

3. **进阶阶段**（第 5-6 章）
   - 从表格方法到函数近似
   - DQN 的核心创新
   - 策略梯度方法
   - 连续控制算法（DDPG/TD3）

4. **前沿阶段**（第 7 章）⭐
   - TRPO/PPO 的原理
   - SAC 的优势
   - DPO 的直觉理解
   - 连接到 LLM 后训练

### 学习方式

1. **先玩游戏** - 每章先运行 games/ 下的示例，直观感受
2. **再读代码** - 理解算法实现
3. **最后推导** - 结合数学公式深入理解

---

## 🎮 游戏列表

| 章节 | 游戏 | 学习概念 |
|------|------|---------|
| 第 1 章 | 🍱 午餐选择器 | MDP 建模 |
| 第 1 章 | 🚇 下班路线规划 | 状态转移 |
| 第 2 章 | 🗺️ 网格寻路 | 最优路径 |
| 第 2 章 | 🎯 简化版 AlphaGo | MCTS 搜索 |
| 第 3 章 | 🃏 21 点游戏 | MC 学习 |
| 第 4 章 | 🏃 悬崖行走 | SARSA vs Q-Learning |
| 第 4 章 | 💎 迷宫寻宝 | TD 学习 |
| 第 5 章 | 🤹 CartPole | DQN 训练 |
| 第 6 章 | 🚗 赛车控制 | 连续控制 |
| 第 7 章 | 🤖 对话优化 | PPO + LLM |

---

## 🔗 参考资源

### 推荐阅读

- **Spinning Up in Deep RL** (OpenAI) - 深度学习 RL 入门
- **Reinforcement Learning: An Introduction** (Sutton & Barto) - RL 圣经
- **CleanRL** - 单文件高质量实现参考

### 相关课程

- **CS285** (Berkeley) - 深度强化学习
- **DeepMind x UCL** - RL 课程（YouTube）
- **莫烦 Python** - 中文 RL 教程

---

## 📊 项目状态

### 完成度

| Phase | 章节 | 状态 | 测试 | 代码行数 |
|-------|------|------|------|----------|
| **Phase 1** | 核心框架 | ✅ 完成 | 42 | ~800 |
| **Phase 2** | 第 1-2 章 | ✅ 完成 | 20 | ~1,200 |
| **Phase 3** | 第 3-4 章 ⭐ | ✅ 完成 | 45 | ~2,000 |
| **Phase 4** | 第 5-6 章 | ✅ 完成 | 44 | ~2,500 |
| **Phase 5** | 第 7 章 ⭐ | ✅ 完成 | 20 | ~2,900 |

**总计**: 171 个测试全部通过 ✅ | ~9,400 行代码

### 算法实现清单

- [x] 经典 RL 算法（MDP、DP、MC、TD）
- [x] 深度 RL（DQN、函数近似）
- [x] 策略梯度（REINFORCE、A2C、DDPG、TD3）
- [x] 前沿算法（**PPO**、**SAC**、**DPO**、**GRPO**）
- [x] Offline RL（BC、CQL、IQL）

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献方式

- 🐛 报告 Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🎨 添加新的游戏示例

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 👤 作者

**Hermes**

- GitHub: [@Setsuna-Yukirin](https://github.com/Setsuna-Yukirin)
- 项目主页：[rl-guide](https://github.com/Setsuna-Yukirin/rl-guide)

---

## 🌟 Star History

如果这个项目对你有帮助，欢迎给个 Star！⭐

---

*最后更新：2026-04-22*
