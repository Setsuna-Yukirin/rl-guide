# 第 1 章：MDP 基础 - 详细设计

**版本**: v0.1.0  
**创建时间**: 2026-04-22  
**状态**: 🟡 设计中

---

## 📌 学习目标

完成本章后，学习者应该能够：

1. **理解 MDP 五元组** - 状态、动作、转移、奖励、折扣因子
2. **掌握贝尔曼方程** - 期望方程和最优化方程
3. **计算价值函数** - V(s) 和 Q(s,a)
4. **理解策略** - 策略评估和策略改进
5. **应用 MDP 建模** - 将实际问题建模为 MDP

---

## 📚 核心概念

### 1.1 MDP 五元组

```
MDP = (S, A, P, R, γ)

S: 状态空间 (State Space)
A: 动作空间 (Action Space)
P: 状态转移概率 (Transition Probability)
   P(s'|s,a) = 在状态 s 采取动作 a 后到达 s' 的概率
R: 奖励函数 (Reward Function)
   R(s,a,s') = 在状态 s 采取动作 a 到达 s' 的奖励
γ: 折扣因子 (Discount Factor)
   0 < γ ≤ 1，表示未来奖励的当前价值
```

### 1.2 回报 (Return)

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ γ^k R_{t+k+1}  (k=0 to ∞)
```

### 1.3 贝尔曼方程

**贝尔曼期望方程** (Bellman Expectation Equation):

```
V_π(s) = Σ π(a|s) Σ P(s'|s,a) [R(s,a,s') + γV_π(s')]
         a∈A    s'∈S
```

**贝尔曼最优化方程** (Bellman Optimality Equation):

```
V*(s) = max Σ P(s'|s,a) [R(s,a,s') + γV*(s')]
        a∈A s'∈S
```

### 1.4 价值函数

**状态价值函数** V(s):
- 从状态 s 开始，遵循策略π的期望回报
- V_π(s) = E_π[G_t | S_t = s]

**动作价值函数** Q(s,a):
- 在状态 s 采取动作 a，然后遵循策略π的期望回报
- Q_π(s,a) = E_π[G_t | S_t = s, A_t = a]

**关系**:
```
V_π(s) = Σ π(a|s) Q_π(s,a)
         a∈A
```

### 1.5 策略

**策略** π:
- 从状态到动作的映射
- 确定性策略：π(s) = a
- 随机性策略：π(a|s) = P(A=a|S=s)

**策略评估**:
- 计算给定策略π的价值函数 V_π

**策略改进**:
- 基于 V_π 改进策略
- 贪心策略：π'(s) = argmax_a Q_π(s,a)

**策略改进定理**:
- 如果 V_π(s) ≤ V_π'(s) 对所有 s 成立，则π' 优于或等于π

---

## 🎮 应用场景设计

### 场景 1: 🍱 午餐选择器

**问题描述**:
每天中午需要决定吃什么，考虑以下因素：
- 天气（晴天/雨天/冷天）
- 预算（充足/紧张）
- 上次选择（避免连续吃同样的）

**MDP 建模**:

```
状态空间 S:
- weather: {sunny, rainy, cold} (3 种)
- budget: {rich, poor} (2 种)
- last_meal: {cafeteria, delivery, bring} (3 种)
- 总状态数：3 × 2 × 3 = 18 种

动作空间 A:
- {cafeteria, delivery, bring} (3 种)

转移概率 P:
- weather: 根据天气转移矩阵
- budget: 根据花费更新
- last_meal:  deterministic（等于当前动作）

奖励函数 R:
- 满意度：cafeteria(5), delivery(7), bring(3)
- 成本惩罚：rich(0), poor(-cost)
- 多样性奖励：如果和上次不同 +2

折扣因子 γ: 0.9
```

**学习目标**:
- 理解状态空间的构建
- 理解奖励函数的设计
- 理解策略如何优化长期回报

**游戏设计**:
```python
class LunchDecisionEnv(gym.Env):
    """午餐选择环境"""
    
    def __init__(self):
        # 状态：[weather, budget, last_meal]
        self.state = self.reset()
    
    def step(self, action):
        # action: 0=cafeteria, 1=delivery, 2=bring
        # 返回：next_state, reward, done, info
        ...
    
    def reset(self):
        # 随机初始化状态
        ...
```

**可视化**:
- 状态转移图
- 不同策略的长期回报对比
- 最优策略可视化

---

### 场景 2: 🚇 下班路线规划

**问题描述**:
下班回家需要选择交通方式，考虑：
- 时间（高峰/非高峰）
- 费用
- 舒适度
- 天气影响

**MDP 建模**:

```
状态空间 S:
- time: {peak, off_peak} (2 种)
- weather: {good, bad} (2 种)
- location: {office, station_A, station_B, home} (4 种)
- 总状态数：2 × 2 × 4 = 16 种

动作空间 A:
- {subway, bus, taxi, walk} (4 种)

转移概率 P:
- location: 根据动作确定性地转移
- time: 保持不变
- weather: 小概率变化

奖励函数 R:
- 时间成本：subway(-10), bus(-15), taxi(-5), walk(-30)
- 费用成本：subway(-3), bus(-2), taxi(-30), walk(0)
- 舒适度：subway(2), bus(1), taxi(5), walk(1)
- 天气惩罚：bad_weather 时 bus 和 walk 额外 -3

折扣因子 γ: 0.95
```

**学习目标**:
- 理解多目标优化（时间/费用/舒适度）
- 理解天气等外部因素的影响
- 理解折扣因子的作用

**游戏设计**:
```python
class CommutePlannerEnv(gym.Env):
    """下班路线规划环境"""
    
    def __init__(self):
        self.locations = ['office', 'station_A', 'station_B', 'home']
        self.state = self.reset()
    
    def step(self, action):
        # 根据动作转移位置
        # 计算时间和费用
        # 返回奖励
        ...
```

**可视化**:
- 路线图
- 不同策略的路径对比
- 总成本对比

---

## 💻 算法实现设计

### 2.1 贝尔曼方程实现

```python
# chapter_01/02_bellman_equation.py

def bellman_expectation_V(
    mdp: TabularMDP,
    V: np.ndarray,
    policy: np.ndarray,
) -> np.ndarray:
    """
    贝尔曼期望方程 - 计算 V_π
    
    V_π(s) = Σ π(a|s) Σ P(s'|s,a) [R(s,a,s') + γV_π(s')]
    
    Args:
        mdp: 表格型 MDP
        V: 当前价值函数
        policy: 策略 π(a|s), 形状 (nS, nA)
    
    Returns:
        new_V: 更新后的价值函数
    """
    new_V = np.zeros_like(V)
    
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                new_V[s] += policy[s, a] * prob * (reward + mdp.gamma * V[next_s])
    
    return new_V


def bellman_optimality_Q(
    mdp: TabularMDP,
    Q: np.ndarray,
) -> np.ndarray:
    """
    贝尔曼最优化方程 - 计算 Q*
    
    Q*(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ max_a' Q*(s',a')]
    
    Args:
        mdp: 表格型 MDP
        Q: 当前 Q 函数
    
    Returns:
        new_Q: 更新后的 Q 函数
    """
    new_Q = np.zeros_like(Q)
    
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                new_Q[s, a] += prob * (reward + mdp.gamma * np.max(Q[next_s]))
    
    return new_Q
```

---

### 2.2 价值函数计算

```python
# chapter_01/03_value_function.py

def compute_V_pi(
    mdp: TabularMDP,
    policy: np.ndarray,
    epsilon: float = 1e-6,
    max_iterations: int = 1000,
) -> np.ndarray:
    """
    计算策略π的价值函数 V_π
    
    使用迭代策略评估
    
    Args:
        mdp: 表格型 MDP
        policy: 策略 π(a|s)
        epsilon: 收敛阈值
        max_iterations: 最大迭代次数
    
    Returns:
        V: 价值函数
    """
    V = np.zeros(mdp.nS)
    
    for i in range(max_iterations):
        delta = 0
        new_V = bellman_expectation_V(mdp, V, policy)
        
        delta = max(delta, np.abs(new_V - V).max())
        V = new_V
        
        if delta < epsilon:
            break
    
    return V


def compute_Q_pi(
    mdp: TabularMDP,
    V: np.ndarray,
) -> np.ndarray:
    """
    从 V_π 计算 Q_π
    
    Q_π(s,a) = Σ P(s'|s,a) [R(s,a,s') + γV_π(s')]
    
    Args:
        mdp: 表格型 MDP
        V: 状态价值函数
    
    Returns:
        Q: 动作价值函数
    """
    Q = np.zeros((mdp.nS, mdp.nA))
    
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_s, reward, done in mdp.P[s][a]:
                Q[s, a] += prob * (reward + mdp.gamma * V[next_s])
    
    return Q
```

---

### 2.3 策略评估与改进

```python
# chapter_01/04_policy.py

def evaluate_policy(
    mdp: TabularMDP,
    policy: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    策略评估：计算给定策略的价值函数
    
    Args:
        mdp: 表格型 MDP
        policy: 策略 π(a|s)
        epsilon: 收敛阈值
    
    Returns:
        V: 价值函数
    """
    return compute_V_pi(mdp, policy, epsilon)


def improve_policy(
    mdp: TabularMDP,
    V: np.ndarray,
) -> np.ndarray:
    """
    策略改进：基于价值函数改进策略
    
    π'(s) = argmax_a Q_π(s,a)
    
    Args:
        mdp: 表格型 MDP
        V: 价值函数
    
    Returns:
        new_policy: 改进后的策略（确定性）
    """
    Q = compute_Q_pi(mdp, V)
    new_policy = np.zeros((mdp.nS, mdp.nA))
    
    for s in range(mdp.nS):
        best_a = np.argmax(Q[s])
        new_policy[s, best_a] = 1.0
    
    return new_policy


def policy_improvement_theorem(
    mdp: TabularMDP,
    policy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    验证策略改进定理
    
    Args:
        mdp: 表格型 MDP
        policy: 当前策略
    
    Returns:
        V_old: 原策略价值
        V_new: 新策略价值
        improved: 是否改进
    """
    # 评估当前策略
    V_old = evaluate_policy(mdp, policy)
    
    # 改进策略
    new_policy = improve_policy(mdp, V_old)
    
    # 评估新策略
    V_new = evaluate_policy(mdp, new_policy)
    
    # 验证改进定理
    improved = np.all(V_new >= V_old - 1e-6)
    
    return V_old, V_new, improved
```

---

## 🧪 测试设计

### 单元测试

```python
# tests/test_chapter_01.py

@pytest.mark.unit
class TestBellmanEquation:
    """贝尔曼方程测试"""
    
    def test_bellman_expectation(self):
        """测试贝尔曼期望方程"""
        mdp = create_simple_mdp()
        V = np.zeros(mdp.nS)
        policy = np.ones((mdp.nS, mdp.nA)) / mdp.nA
        
        new_V = bellman_expectation_V(mdp, V, policy)
        
        assert new_V.shape == V.shape
        assert np.all(new_V >= 0)  # 奖励非负时价值非负
    
    def test_bellman_optimality(self):
        """测试贝尔曼最优化方程"""
        mdp = create_simple_mdp()
        Q = np.zeros((mdp.nS, mdp.nA))
        
        new_Q = bellman_optimality_Q(mdp, Q)
        
        assert new_Q.shape == Q.shape


@pytest.mark.unit
class TestValueFunction:
    """价值函数测试"""
    
    def test_compute_V_pi_convergence(self):
        """测试 V_π 计算收敛"""
        mdp = create_simple_mdp()
        policy = np.ones((mdp.nS, mdp.nA)) / mdp.nA
        
        V = compute_V_pi(mdp, policy)
        
        # 验证收敛
        assert np.all(np.isfinite(V))
    
    def test_Q_pi_from_V(self):
        """测试从 V 计算 Q"""
        mdp = create_simple_mdp()
        policy = np.ones((mdp.nS, mdp.nA)) / mdp.nA
        V = compute_V_pi(mdp, policy)
        
        Q = compute_Q_pi(mdp, V)
        
        assert Q.shape == (mdp.nS, mdp.nA)


@pytest.mark.unit
class TestPolicy:
    """策略测试"""
    
    def test_policy_improvement(self):
        """测试策略改进"""
        mdp = create_simple_mdp()
        policy = np.ones((mdp.nS, mdp.nA)) / mdp.nA
        
        V_old, V_new, improved = policy_improvement_theorem(mdp, policy)
        
        assert improved
        assert np.all(V_new >= V_old - 1e-6)
```

### 集成测试

```python
@pytest.mark.integration
class TestLunchEnv:
    """午餐环境测试"""
    
    def test_lunch_env_step(self):
        """测试环境 step"""
        env = LunchDecisionEnv()
        state, info = env.reset()
        
        next_state, reward, done, truncated, info = env.step(0)
        
        assert len(next_state) == len(state)
        assert isinstance(reward, float)
    
    def test_lunch_env_mdp_solve(self):
        """测试 MDP 求解"""
        env = LunchDecisionEnv()
        mdp = env.to_mdp()
        
        # 使用策略迭代求解
        policy = np.ones((mdp.nS, mdp.nA)) / mdp.nA
        
        for _ in range(100):
            V = evaluate_policy(mdp, policy)
            policy = improve_policy(mdp, V)
        
        # 验证策略有效
        assert np.all(policy.sum(axis=1) - 1 < 1e-6)
```

---

## 📊 可视化设计

### 1. 状态转移图

```python
def plot_mdp_transitions(mdp: TabularMDP, state: int):
    """绘制某状态的状态转移图"""
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.DiGraph()
    
    for a in range(mdp.nA):
        for prob, next_s, reward, done in mdp.P[state][a]:
            if prob > 0.01:  # 只画显著概率
                G.add_edge(f"s{state}", f"s{next_s}", 
                          label=f"a{a}: {prob:.2f}\nR={reward}")
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()
```

### 2. 价值函数热力图

```python
def plot_value_heatmap(V: np.ndarray, title: str = "Value Function"):
    """绘制价值函数热力图"""
    plt.figure(figsize=(8, 6))
    
    # 假设状态是网格状的
    height = int(np.sqrt(len(V)))
    width = len(V) // height
    
    V_grid = V[:height*width].reshape(height, width)
    
    plt.imshow(V_grid, cmap='viridis', aspect='auto')
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("State X")
    plt.ylabel("State Y")
    
    plt.show()
```

### 3. 策略对比图

```python
def compare_policies(V1: np.ndarray, V2: np.ndarray, title: str = "Policy Comparison"):
    """对比两个策略的价值函数"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 策略 1
    im1 = axes[0].imshow(V1.reshape(height, width), cmap='viridis')
    axes[0].set_title("Policy 1")
    plt.colorbar(im1, ax=axes[0])
    
    # 策略 2
    im2 = axes[1].imshow(V2.reshape(height, width), cmap='viridis')
    axes[1].set_title("Policy 2")
    plt.colorbar(im2, ax=axes[1])
    
    # 差异
    diff = V2 - V1
    im3 = axes[2].imshow(diff.reshape(height, width), cmap='RdBu')
    axes[2].set_title("Difference (Policy 2 - Policy 1)")
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

---

## 📋 实现清单

### 核心算法

- [ ] `bellman_expectation_V()` - 贝尔曼期望方程
- [ ] `bellman_optimality_Q()` - 贝尔曼最优化方程
- [ ] `compute_V_pi()` - 计算策略价值
- [ ] `compute_Q_pi()` - 计算动作价值
- [ ] `evaluate_policy()` - 策略评估
- [ ] `improve_policy()` - 策略改进
- [ ] `policy_improvement_theorem()` - 验证改进定理

### 游戏环境

- [ ] `LunchDecisionEnv` - 午餐选择环境
- [ ] `CommutePlannerEnv` - 路线规划环境
- [ ] 环境到 MDP 的转换

### 可视化

- [ ] `plot_mdp_transitions()` - 状态转移图
- [ ] `plot_value_heatmap()` - 价值热力图
- [ ] `compare_policies()` - 策略对比

### 测试

- [ ] 贝尔曼方程测试（3 个）
- [ ] 价值函数测试（3 个）
- [ ] 策略测试（3 个）
- [ ] 环境测试（4 个）
- [ ] 集成测试（2 个）

---

## 🔗 连接到后续章节

### 第 2 章：动态规划

- 本章的策略评估是 DP 的基础
- 策略改进是策略迭代的核心
- 价值迭代基于贝尔曼最优化方程

### 第 3 章：蒙特卡洛

- 本章的 V_π 和 Q_π 是 MC 估计的目标
- 策略评估概念在 MC 中延续

### 第 4 章：时序差分

- TD 是贝尔曼方程的采样版本
- Q-Learning 基于贝尔曼最优化方程

---

*文档版本：v0.1.0*  
*最后更新：2026-04-22*
