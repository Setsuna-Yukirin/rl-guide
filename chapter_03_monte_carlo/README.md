# 第 3 章：蒙特卡洛方法 (Monte Carlo Methods)

## 📚 学习目标

- 理解蒙特卡洛方法的核心思想
- 掌握蒙特卡洛预测算法（估计 V_π 和 Q_π）
- 掌握蒙特卡洛控制算法（寻找最优策略）
- 理解探索与利用的权衡（ε-greedy 策略）
- 通过 Blackjack 游戏实践 MC 算法

## 🔑 核心概念

### 1. 蒙特卡洛方法基础

**核心思想**：
- 从经验中学习（通过采样完整的 episode）
- 不需要环境的先验知识（模型无关）
- 只适用于**回合制任务**（有明确的开始和结束）

**与动态规划的区别**：
| 特性 | 动态规划 | 蒙特卡洛 |
|------|----------|----------|
| 环境模型 | 需要已知 | 不需要 |
| 更新时机 | 每步更新 | 回合结束更新 |
| 偏差/方差 | 低方差，有偏 | 高方差，无偏 |
| 计算复杂度 | 高 | 低 |

### 2. 蒙特卡洛预测 (MC Prediction)

**目标**：估计给定策略 π 的价值函数

**第一访问 MC (First-visit MC)**：
```
对于每个状态 s：
  1. 生成一个 episode
  2. 对于 episode 中第一次访问的每个状态 s：
     - 计算回报 G_t = R_{t+1} + γ*R_{t+2} + γ²*R_{t+3} + ...
     - 将 G_t 添加到 returns(s) 列表
  3. V(s) = average(returns(s))
```

**每次访问 MC (Every-visit MC)**：
- 与第一访问类似，但每次访问状态都记录回报

### 3. 蒙特卡洛控制 (MC Control)

**目标**：找到最优策略 π*

**基本思想**：
1. 策略评估：估计 Q_π(s, a)
2. 策略改进：π(s) = argmax_a Q(s, a)

**挑战**：如何保证充分探索？

### 4. ε-贪婪策略 (ε-Greedy Policy)

```
π(a|s) = {
  1 - ε + ε/|A|    如果 a = argmax_a' Q(s, a')  (贪婪动作)
  ε/|A|           其他动作
}
```

**作用**：
- 以概率 1-ε 选择最优动作（利用）
- 以概率 ε 随机选择动作（探索）
- 保证所有动作都有被选择的机会

### 5. GLIE 蒙特卡洛控制

**GLIE** (Greedy in the Limit with Infinite Exploration):
- 无限探索：所有状态 - 动作对被访问无限次
- 贪婪极限：策略最终收敛到贪婪策略

**实现方式**：
```
ε_k = 1/k  # 随时间衰减
```

### 6. 重要性采样 (Importance Sampling)

**离策略学习**：用行为策略 b 生成的数据评估目标策略 π

**重要性采样比率**：
```
ρ = P(episode | π) / P(episode | b)
```

**加权重要性采样**：
```
V(s) = Σ(ρ_i * G_i) / Σ(ρ_i)
```

## 📁 文件结构

```
chapter_03_monte_carlo/
├── README.md                    # 本章文档
├── __init__.py                  # 模块初始化
├── mc_prediction.py             # MC 预测算法
├── mc_control.py                # MC 控制算法
├── epsilon_greedy.py            # ε-贪婪策略
└── games/
    ├── __init__.py
    └── blackjack.py             # Blackjack 环境
```

## 🎮 实践项目：Blackjack

**环境描述**：
- 目标：点数接近 21 点且不超过
- 状态：(玩家点数，庄家明牌，是否有可用 A)
- 动作：要牌 (Hit) / 停牌 (Stand)
- 奖励：赢 +1，输 -1，平局 0

**学习目标**：
- 使用 MC 控制找到最优策略
- 可视化策略表面

## 🔬 实验练习

1. 比较第一访问 MC 和每次访问 MC 的收敛速度
2. 分析不同 ε 值对学习效果的影响
3. 实现 GLIE MC 控制
4. 比较 MC 和 DP 的样本效率

## 📖 参考资料

- Sutton & Barto, Chapter 5: Monte Carlo Methods
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction
