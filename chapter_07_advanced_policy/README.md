# 第 7 章：高级策略优化

本章介绍现代强化学习的高级策略优化算法，包括 PPO、SAC 等主流算法，以及与 LLM 后训练相关的 DPO/GRPO 算法，最后介绍 Offline RL 基础。

## 📚 本章内容

### 7.1 PPO（Proximal Policy Optimization）

**核心思想**：通过限制策略更新幅度，实现稳定的策略梯度优化。

**关键特性**：
- **Clipped Surrogate Objective**：截断策略比率，防止过大更新
- **Generalized Advantage Estimation (GAE)**：低方差优势估计
- **Value Function Clipping**：价值函数裁剪

**数学公式**：
```
L^CLIP(θ) = E_t[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```

其中：
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` 是策略比率
- `A_t` 是优势函数估计
- `ε` 是裁剪参数（通常 0.2）

**算法流程**：
```python
for iteration in range(num_iterations):
    # 1. 采样轨迹
    trajectories = collect_trajectories(π_θ)
    
    # 2. 计算优势 (GAE)
    advantages = compute_gae(trajectories, V_φ)
    
    # 3. 多次 PPO 更新
    for _ in range(ppo_epochs):
        # 计算策略比率
        ratio = π_θ(a|s) / π_θ_old(a|s)
        
        # 截断目标
        surr1 = ratio * A
        surr2 = clip(ratio, 1-ε, 1+ε) * A
        loss = -min(surr1, surr2).mean()
        
        # 价值函数损失
        value_loss = (V_φ(s) - V_target)^2
        
        # 总损失
        total_loss = loss + c1 * value_loss - c2 * entropy
```

---

### 7.2 SAC（Soft Actor-Critic）

**核心思想**：在标准 RL 目标基础上增加最大熵项，鼓励探索。

**关键特性**：
- **Maximum Entropy**：最大化期望回报 + 策略熵
- **Off-Policy**：可使用经验回放
- **Twin Q-Networks**：两个 Q 网络防止过估计
- **Automatic Temperature Tuning**：自动调整熵系数

**数学公式**：
```
J(θ) = E[Σ γ^t (r(s_t, a_t) + α * H(π(·|s_t)))]
```

其中 `α` 是温度参数，控制熵的重要性。

**算法流程**：
```python
for each step:
    # 1. 选择动作（从策略采样）
    a ~ π_θ(·|s)
    
    # 2. 执行动作，存储经验
    (s, a, r, s', done) → replay_buffer
    
    # 3. 更新 Q 网络（两个）
    for Q in [Q1, Q2]:
        y = r + γ * (1-done) * (min(Q1', Q2') - α * log π)
        loss = MSE(Q(s,a), y)
    
    # 4. 更新策略
    J_π = E[α * log π_θ(a|s) - Q(s,a)]
    
    # 5. 更新温度参数（自动熵调节）
    J_α = -α * E[log π + H_target]
```

---

### 7.3 DPO（Direct Preference Optimization）

**核心思想**：直接从人类偏好数据优化策略，无需显式奖励模型。

**与 LLM 后训练的连接**：
- **传统 RLHF**：SFT → 奖励模型 → PPO 优化
- **DPO**：直接从偏好数据优化，简化流程

**数学公式**：
```
L_DPO(π_θ; π_ref) = -E[log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) 
                              - β * log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

其中：
- `(x, y_w, y_l)` 是偏好三元组（prompt, 优选回答，劣选回答）
- `π_ref` 是参考策略（通常是 SFT 模型）
- `β` 是温度参数

**算法流程**：
```python
for batch in preference_data:
    # 1. 计算对数概率比
    log_ratio_w = log(π_θ(y_w|x)) - log(π_ref(y_w|x))
    log_ratio_l = log(π_θ(y_l|x)) - log(π_ref(y_l|x))
    
    # 2. 计算 DPO 损失
    logits = β * (log_ratio_w - log_ratio_l)
    loss = -log_sigmoid(logits).mean()
    
    # 3. 更新策略
    loss.backward()
    optimizer.step()
```

---

### 7.4 GRPO（Group Relative Policy Optimization）

**核心思想**：对同一 prompt 生成多个回答，通过组内相对质量优化策略。

**与 LLM 后训练的连接**：
- **无需奖励模型**：通过组内比较自动产生监督信号
- **高效采样**：一次生成多个回答，减少 API 调用
- **适用于数学/代码任务**：可通过执行结果自动评分

**数学公式**：
```
对于每个 prompt x，生成 G 个回答 {y_1, ..., y_G}
计算每个回答的奖励 {r_1, ..., r_G}
优势估计：A_i = (r_i - mean(r)) / std(r)

L_GRPO(θ) = E[Σ_i min(r_i(θ) * A_i, clip(r_i(θ), 1-ε, 1+ε) * A_i)]
```

**算法流程**：
```python
for batch in data:
    # 1. 对每个 prompt 生成 G 个回答
    for x in batch:
        outputs = [model.generate(x) for _ in range(G)]
    
    # 2. 计算奖励（执行结果/规则评分）
    rewards = [compute_reward(x, y) for y in outputs]
    
    # 3. 标准化优势
    advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
    
    # 4. PPO 风格更新
    loss = compute_ppo_loss(outputs, advantages)
```

---

### 7.5 Offline RL（离线强化学习）

**核心思想**：从固定数据集中学习策略，无需与环境交互。

**关键挑战**：
- **分布外 (OOD) 动作**：策略可能选择数据集中未见的动作
- **外推误差**：Q 函数对 OOD 动作的估计不准确

**算法分类**：

#### 7.5.1 行为克隆 (Behavior Cloning, BC)
```
L_BC(θ) = -E[log π_θ(a|s)]  (监督学习)
```

#### 7.5.2 Conservative Q-Learning (CQL)
```
L_CQL = L_TD + α * (E[Q(s, a~π)] - E[Q(s, a~data)])
```
通过惩罚 OOD 动作的 Q 值，使 Q 函数保守。

#### 7.5.3 Implicit Q-Learning (IQL)
```
# 使用期望回归学习值函数
L_V = E[(V(s) - Q(s,a))^2]  (只考虑高 Q 值动作)
# 优势加权行为克隆
L_π = E[exp(β * (Q(s,a) - V(s))) * log π(a|s)]
```

**算法流程**：
```python
# CQL 训练
for batch in offline_dataset:
    # TD 损失
    td_loss = compute_td_loss(Q, batch)
    
    # CQL 正则化
    cql_loss = Q(s, π(s)).mean() - Q(s, a_data).mean()
    
    # 总损失
    loss = td_loss + α * cql_loss
```

---

## 🎯 算法对比

| 算法 | On/Off-Policy | 连续/离散动作 | 样本效率 | 稳定性 | 适用场景 |
|------|---------------|---------------|----------|--------|----------|
| **PPO** | On-Policy | 两者 | 中 | 高 | 通用、机器人控制 |
| **SAC** | Off-Policy | 连续 | 高 | 高 | 连续控制、需要探索 |
| **DPO** | Off-Policy | 离散 (文本) | 高 | 高 | LLM 对齐、偏好优化 |
| **GRPO** | On-Policy | 离散 (文本) | 中 | 高 | 数学/代码生成 |
| **CQL** | Off-Policy | 两者 | 高 | 中 | Offline RL、安全应用 |

---

## 📁 文件结构

```
chapter_07_advanced_policy/
├── README.md                 # 本章文档
├── ppo.py                    # PPO 算法实现
├── sac.py                    # SAC 算法实现
├── dpo.py                    # DPO 算法实现 (LLM 连接)
├── grpo.py                   # GRPO 算法实现 (LLM 连接)
├── offline_rl.py             # Offline RL (BC/CQL/IQL)
├── games/
│   └── __init__.py           # 环境创建函数
└── tests/
    └── test_chapter_07.py    # 单元测试
```

---

## 🚀 快速开始

### PPO 示例

```python
from chapter_07_advanced_policy.ppo import PPOAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = PPOAgent(state_dim=4, n_actions=2)

rewards = agent.train(env, n_episodes=100)
```

### SAC 示例

```python
from chapter_07_advanced_policy.sac import SACAgent
import gymnasium as gym

env = gym.make('Pendulum-v1')
agent = SACAgent(state_dim=3, action_dim=1, action_bounds=(-2.0, 2.0))

rewards = agent.train(env, n_episodes=100)
```

### DPO 示例（LLM 对齐）

```python
from chapter_07_advanced_policy.dpo import DPOTrainer

trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    beta=0.1
)

# 偏好数据：(prompt, chosen, rejected)
preference_data = [
    ("写一首诗", "好的诗", "差的诗"),
    ...
]

trainer.train(preference_data, n_epochs=10)
```

### GRPO 示例（数学问题）

```python
from chapter_07_advanced_policy.grpo import GRPOTrainer

trainer = GRPOTrainer(
    model=policy_model,
    group_size=8,
    reward_fn=math_reward_fn
)

# 数学问题数据集
math_problems = [
    "解方程 x^2 + 2x - 3 = 0",
    ...
]

trainer.train(math_problems, n_epochs=10)
```

### Offline RL 示例

```python
from chapter_07_advanced_policy.offline_rl import CQLAgent
import numpy as np

# 加载离线数据集
dataset = load_d4rl_dataset('halfcheetah-medium-v2')

agent = CQLAgent(state_dim=17, action_dim=6)
agent.train_offline(dataset, n_epochs=100)
```

---

## 🧪 运行测试

```bash
# 运行第 7 章测试
pytest tests/test_chapter_07.py -v

# 运行所有测试
make test
```

---

## 📖 参考文献

1. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. **SAC**: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
3. **DPO**: Rafailov et al. "Direct Preference Optimization" (2023)
4. **GRPO**: Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
5. **CQL**: Kumar et al. "Conservative Q-Learning for Offline RL" (2020)
6. **IQL**: Kostrikov et al. "Offline RL with Implicit Q-Learning" (2022)

---

*文档版本：v1.0.0*  
*最后更新：2026-04-22*
