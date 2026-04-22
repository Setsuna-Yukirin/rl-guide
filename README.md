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

```
第 1 章          第 2 章          第 3 章          第 4 章          第 5 章          第 6 章          第 7 章
MDP 基础    →   动态规划    →   蒙特卡洛    →   时序差分    →   函数近似    →   策略梯度    →   高级策略
  🍱            🗺️            🃏            🏃            🤹            🚗            🤖
  午餐选择      网格寻路       21 点游戏      悬崖行走       CartPole      赛车控制      对话优化
                                              💎            🕹️            🎯            📝
                                              迷宫寻宝      打砖块        机械臂        文本生成
                                              🐍            🚀            🎮
                                              贪吃蛇        登月器        乒乓球
```

---

## 📁 目录结构

```
rl-guide/
├── README.md                        # 本文件
├── requirements.txt                 # Python 依赖
├── .gitignore                       # Git 忽略文件
│
├── chapter_01_mdp_fundamentals/     # 第 1 章：MDP 基础
│   ├── README.md                    # 章节说明
│   ├── 01_mdp_core.py               # MDP 核心类
│   ├── 02_bellman_equation.py       # 贝尔曼方程
│   ├── 03_value_function.py         # 价值函数
│   ├── 04_policy.py                 # 策略定义
│   └── games/                       # 应用场景
│       ├── lunch_decision.py        # 🍱 午餐选择器
│       └── commute_planner.py       # 🚇 下班路线规划
│
├── chapter_02_dynamic_programming/  # 第 2 章：动态规划 + 搜索
│   ├── README.md
│   ├── 01_policy_evaluation.py
│   ├── 02_policy_iteration.py
│   ├── 03_value_iteration.py
│   ├── 04_mcts.py                   # ⭐ 蒙特卡洛树搜索
│   └── games/
│       ├── gridworld_nav.py         # 🗺️ 网格寻路
│       ├── warehouse_robot.py       # 📦 仓库搬运
│       └── alphago_simple.py        # 🎯 简化版 AlphaGo
│
├── chapter_03_monte_carlo/          # 第 3 章：蒙特卡洛方法
│   ├── README.md
│   ├── 01_mc_prediction.py
│   ├── 02_mc_control.py
│   ├── 03_on_off_policy.py
│   └── games/
│       ├── blackjack.py             # 🃏 21 点游戏
│       ├── slot_machine.py          # 🎰 老虎机
│       └── flight_chess.py          # 🎲 飞行棋
│
├── chapter_04_temporal_difference/  # 第 4 章：时序差分学习 ⭐核心
│   ├── README.md
│   ├── 01_td_prediction.py
│   ├── 02_sarsa.py
│   ├── 03_q_learning.py
│   ├── 04_expected_sarsa.py
│   ├── 05_exploration_strategies.py # ⭐ 探索策略对比
│   └── games/
│       ├── cliff_walking.py         # 🏃 悬崖行走
│       ├── maze_treasure.py         # 💎 迷宫寻宝
│       └── snake_simple.py          # 🐍 贪吃蛇
│
├── chapter_05_function_approximation/ # 第 5 章：函数近似
│   ├── README.md
│   ├── 01_linear_approximation.py
│   ├── 02_neural_network_q.py
│   ├── 03_dqn.py
│   └── games/
│       ├── cartpole_balance.py      # 🤹 倒立摆平衡
│       ├── breakout_atari.py        # 🕹️ 打砖块
│       └── lunar_lander.py          # 🚀 登月器
│
├── chapter_06_policy_gradient/      # 第 6 章：策略梯度 + 连续控制
│   ├── README.md
│   ├── 01_reinforce.py
│   ├── 02_actor_critic.py
│   ├── 03_a2c.py
│   ├── 04_ddpg.py                   # ⭐ DDPG
│   ├── 05_td3.py                    # ⭐ TD3
│   └── games/
│       ├── car_racing.py            # 🚗 赛车控制
│       ├── robotic_arm.py           # 🎯 机械臂抓取
│       └── pong_simple.py           # 🎮 乒乓球
│
├── chapter_07_advanced_policy/      # 第 7 章：高级策略 + 离线 RL ⭐目标
│   ├── README.md
│   ├── 01_trpo.py
│   ├── 02_ppo.py
│   ├── 03_sac.py                    # ⭐ SAC
│   ├── 04_dpo_intuition.py          # DPO 直觉
│   ├── 05_offline_rl_intro.py       # ⭐ 离线 RL 简介
│   └── games/
│       ├── dialogue_optimization.py # 🤖 对话优化
│       ├── text_generation.py       # 📝 文本生成
│       └── multi_objective.py       # 🎯 多目标优化
│
├── utils/                           # 工具函数
│   ├── visualization.py             # 可视化工具
│   ├── metrics.py                   # 性能指标
│   └── env_wrapper.py               # 环境封装
│
├── tests/                           # 测试
│   ├── test_algorithms.py
│   └── test_convergence.py
│
└── notebooks/                       # Jupyter 笔记本
    ├── chapter_01_intro.ipynb
    └── ...
```

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

## 📝 项目状态

- [x] 项目规划完成
- [x] 仓库创建
- [ ] Phase 1: 基础框架（进行中）
- [ ] Phase 2: 第 1-2 章
- [ ] Phase 3: 第 3-4 章 ⭐核心
- [ ] Phase 4: 第 5-6 章
- [ ] Phase 5: 第 7 章 ⭐目标

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
