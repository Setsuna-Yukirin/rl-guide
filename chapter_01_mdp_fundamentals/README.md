# 第 1 章：MDP 基础

> **马尔可夫决策过程 (Markov Decision Process)**

---

## 📌 学习目标

完成本章后，你将能够：

- [x] 理解 MDP 五元组 (S, A, P, R, γ)
- [x] 掌握贝尔曼方程（期望方程和最优化方程）
- [x] 计算价值函数 V(s) 和 Q(s,a)
- [x] 实现策略评估和策略改进
- [x] 使用策略迭代和价值迭代求解 MDP
- [x] 将实际问题建模为 MDP

---

## 📚 核心概念

### MDP 五元组

```
MDP = (S, A, P, R, γ)

S: 状态空间
A: 动作空间
P: 状态转移概率 P(s'|s,a)
R: 奖励函数 R(s,a,s')
γ: 折扣因子 (0 < γ ≤ 1)
```

### 贝尔曼方程

**期望方程**:
```
V_π(s) = Σ π(a|s) Σ P(s'|s,a) [R(s,a,s') + γV_π(s')]
```

**最优化方程**:
```
V*(s) = max Σ P(s'|s,a) [R(s,a,s') + γV*(s')]
```

---

## 💻 算法实现

### 价值函数计算

```python
from chapter_01_mdp_fundamentals import value_function

# 策略迭代
V, policy, iters = value_function.policy_iteration(mdp)

# 价值迭代
V, policy, iters = value_function.value_iteration(mdp)
```

### 运行游戏

```python
from chapter_01_mdp_fundamentals.games import lunch_decision

env = lunch_decision.LunchDecisionEnv(render_mode="human")
state, _ = env.reset(seed=42)

for t in range(10):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
```

---

## 🎮 游戏

### 🍱 午餐选择器

**状态**: 天气 × 预算 × 上次选择 (18 种)
**动作**: 食堂 / 外卖 / 带饭
**奖励**: 满意度 - 成本 + 多样性奖励

运行游戏:
```bash
python chapter_01_mdp_fundamentals/games/lunch_decision.py
```

---

## 🧪 测试

运行测试:
```bash
pytest tests/test_chapter_01.py -v
```

---

## 📝 练习

1. 修改午餐选择器的奖励函数，观察最优策略变化
2. 实现下班路线规划环境
3. 比较策略迭代和价值迭代的收敛速度

---

## 🔗 下一章

[第 2 章：动态规划 →](../chapter_02_dynamic_programming/README.md)
