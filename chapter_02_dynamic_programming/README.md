# 第 2 章：动态规划 (Dynamic Programming)

## 📚 学习目标

- 理解动态规划在强化学习中的应用
- 掌握策略迭代 (Policy Iteration) 算法
- 掌握价值迭代 (Value Iteration) 算法
- 了解蒙特卡洛树搜索 (MCTS) 基础
- 通过网格寻路游戏实践 DP 算法

## 🔑 核心概念

### 1. 动态规划基础

动态规划 (DP) 是一类在**已知 MDP 模型**（转移概率和奖励函数）情况下求解最优策略的算法。

**关键假设**：
- 环境完全已知（转移概率 `P(s'|s,a)` 和奖励 `R(s,a)` 已知）
- 状态空间有限且可枚举

**核心思想**：
- 使用价值函数指导策略改进
- 通过"备份"(backup) 操作迭代更新价值

### 2. 策略评估 (Policy Evaluation)

计算给定策略 π 的价值函数 V_π：

```
V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a) + γ * V_k(s')]
```

**收敛条件**：`max_s |V_{k+1}(s) - V_k(s)| < θ`

### 3. 策略改进 (Policy Improvement)

基于当前价值函数改进策略：

```
π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a) + γ * V_π(s')]
```

**策略改进定理**：如果 V_π'(s) ≥ V_π(s) 对所有 s 成立，则 π' 至少和 π 一样好。

### 4. 策略迭代 (Policy Iteration)

交替进行策略评估和策略改进：

```
重复：
  1. 策略评估：计算 V_π
  2. 策略改进：π' = greedy(V_π)
直到：策略收敛（不再改变）
```

**特点**：
- 通常收敛很快（几次迭代即可）
- 每次迭代计算量大（需要完整策略评估）

### 5. 价值迭代 (Value Iteration)

将策略评估截断为一步更新的简化算法：

```
V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ * V_k(s')]
```

**特点**：
- 每次迭代计算量小
- 可能需要更多迭代次数收敛
- 实践中通常比策略迭代更快

### 6. 蒙特卡洛树搜索 (MCTS) 简介

MCTS 是一种**基于采样的规划算法**，核心思想：

```
重复 N 次：
  1. 选择 (Selection): 从根节点出发，用 UCT 公式选择子节点
  2. 扩展 (Expansion): 添加一个新的子节点
  3. 模拟 (Simulation): 从新节点随机模拟到终止
  4. 回溯 (Backpropagation): 将结果回传更新路径上的节点
```

**UCT 公式**：
```
UCT(s,a) = Q(s,a) + c * sqrt(ln N(s) / N(s,a))
```

其中：
- `Q(s,a)`: 动作价值的平均值
- `N(s)`: 状态访问次数
- `N(s,a)`: 动作选择次数
- `c`: 探索常数（通常取 √2）

## 📁 文件结构

```
chapter_02_dynamic_programming/
├── README.md                    # 本章文档
├── __init__.py                  # 模块初始化
├── policy_evaluation.py         # 策略评估算法
├── policy_iteration.py          # 策略迭代算法
├── value_iteration.py           # 价值迭代算法
├── mcts.py                      # MCTS 算法
└── games/
    ├── __init__.py
    └── gridworld_nav.py         # 网格寻路环境
```

## 🎮 实践项目

### 项目 1：网格寻路 (GridWorld Navigation)

**环境描述**：
- 5x5 网格世界
- 起点：(0, 0)
- 终点：(4, 4)，奖励 +10
- 陷阱：随机位置，奖励 -5
- 每步奖励：-1
- 动作：上、下、左、右

**目标**：使用策略迭代/价值迭代找到最优路径

### 项目 2：仓库机器人 (Warehouse Robot)

**环境描述**：
- 10x10 网格
- 多个货架位置
- 机器人需要收集物品并送到指定位置
- 碰撞惩罚：-10

**目标**：使用价值迭代规划最优路径

## 🔬 实验练习

1. 比较策略迭代和价值迭代的收敛速度
2. 分析不同折扣因子 γ 对最优策略的影响
3. 实现带异步更新的值迭代
4. 使用 MCTS 解决简单的棋类游戏

## 📖 参考资料

- Sutton & Barto, Chapter 4: Dynamic Programming
- Bertsekas, D. P. (2017). Dynamic Programming and Optimal Control
- Browne et al. (2012). A Survey of Monte Carlo Tree Search Methods
