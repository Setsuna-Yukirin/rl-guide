# 02 - 系统架构

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层                               │
├─────────────────────────────────────────────────────────────┤
│  CLI 命令  │  Jupyter Notebook  │  Python API  │  游戏界面   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      应用层                                  │
├─────────────────────────────────────────────────────────────┤
│  章节管理器  │  游戏引擎  │  可视化系统  │  训练循环        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      算法层                                  │
├─────────────────────────────────────────────────────────────┤
│  Chapter 1  │  Chapter 2  │  Chapter 3  │  ...  │  Chapter 7│
│  MDP 基础   │  动态规划    │  蒙特卡洛    │       │  高级策略  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      核心层                                  │
├─────────────────────────────────────────────────────────────┤
│  MDP 类  │  Policy 类  │  ValueFunction 类  │  ReplayBuffer │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      环境层                                  │
├─────────────────────────────────────────────────────────────┤
│  Gymnasium  │  自定义环境  │  游戏环境  │  环境包装器        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      工具层                                  │
├─────────────────────────────────────────────────────────────┤
│  NumPy  │  PyTorch  │  Matplotlib  │  Pygame  │  其他工具   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 模块划分

### 核心层 (Core Layer)

**职责**：提供 RL 基础抽象

| 模块 | 文件 | 说明 |
|------|------|------|
| `mdp` | `utils/core/mdp.py` | MDP 五元组定义 |
| `policy` | `utils/core/policy.py` | 策略抽象基类 |
| `value_function` | `utils/core/value_function.py` | 价值函数抽象 |
| `replay_buffer` | `utils/core/replay_buffer.py` | 经验回放缓冲区 |

**示例**：
```python
from utils.core import MDP, Policy

class QLearningPolicy(Policy):
    """基于 Q 表的策略"""
    ...
```

---

### 算法层 (Algorithm Layer)

**职责**：实现具体 RL 算法

**组织方式**：按章节划分

```
chapter_01_mdp_fundamentals/
├── 01_mdp_core.py           # MDP 核心类
├── 02_bellman_equation.py   # 贝尔曼方程
├── 03_value_function.py     # 价值函数
└── 04_policy.py             # 策略

chapter_04_temporal_difference/
├── 01_td_prediction.py
├── 02_sarsa.py              # SARSA 类
├── 03_q_learning.py         # Q-Learning 类
├── 04_expected_sarsa.py
└── 05_exploration_strategies.py
```

**算法类设计规范**：

```python
class QLearning:
    """
    Q-Learning 算法实现
    
    Attributes:
        state_dim (int): 状态空间维度
        action_dim (int): 动作空间维度
        lr (float): 学习率
        gamma (float): 折扣因子
        epsilon (float): 探索率
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        """初始化 Q-Learning"""
        ...
    
    def get_action(self, state: np.ndarray) -> int:
        """ε-greedy 策略选择动作"""
        ...
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        更新 Q 表
        
        Returns:
            float: TD error
        """
        ...
    
    def save(self, path: str):
        """保存 Q 表"""
        ...
    
    def load(self, path: str):
        """加载 Q 表"""
        ...
```

---

### 应用层 (Application Layer)

**职责**：提供用户接口和应用

| 模块 | 说明 |
|------|------|
| `TrainingLoop` | 训练循环管理 |
| `GameEngine` | 游戏引擎 |
| `Visualization` | 可视化系统 |
| `ChapterManager` | 章节进度管理 |

**训练循环示例**：

```python
from utils import TrainingLoop

loop = TrainingLoop(
    algorithm=QLearning(...),
    env=gym.make("CliffWalking-v0"),
    num_episodes=1000,
    render=True,
)

stats = loop.run()
```

---

### 环境层 (Environment Layer)

**职责**：提供 RL 环境

| 类型 | 来源 | 示例 |
|------|------|------|
| **标准环境** | Gymnasium | CartPole, Pendulum |
| **自定义环境** | 自实现 | 午餐选择器、迷宫 |
| **游戏环境** | Pygame | 21 点、赛车 |
| **包装环境** | 自实现 | 归一化、帧堆叠 |

**自定义环境示例**：

```python
import gymnasium as gym

class LunchDecisionEnv(gym.Env):
    """
    午餐选择环境
    
    State: [天气，预算，上次选择]
    Action: [食堂，外卖，带饭]
    Reward: 满意度 - 成本
    """
    
    def __init__(self):
        super().__init__()
        self.weather = ["sunny", "rainy", "cold"]
        self.actions = ["cafeteria", "delivery", "bring"]
        
    def step(self, action):
        ...
    
    def reset(self):
        ...
```

---

## 🔄 数据流

### 训练流程

```
1. 初始化环境
   env = gym.make("CartPole-v1")
   
2. 初始化算法
   agent = DQN(...)
   
3. 训练循环
   for episode in range(num_episodes):
       state, _ = env.reset()
       
       for t in range(max_steps):
           # 选择动作
           action = agent.get_action(state)
           
           # 执行动作
           next_state, reward, done, _, _ = env.step(action)
           
           # 存储经验
           replay_buffer.add(state, action, reward, next_state, done)
           
           # 更新算法
           batch = replay_buffer.sample()
           agent.update(batch)
           
           state = next_state
           
           if done:
               break
       
       # 记录统计
       stats.append(episode_reward)
   
4. 可视化结果
   plot(stats)
```

### 游戏流程

```
1. 初始化游戏
   game = BlackjackGame()
   
2. 训练 AI
   agent = MCAgent()
   agent.train(game)
   
3. 人机对战
   while True:
       # 玩家回合
       player_action = human_play()
       
       # AI 回合
       ai_action = agent.get_action()
       
       # 更新游戏状态
       game.step(player_action, ai_action)
       
       # 渲染
       game.render()
```

---

## 🔌 接口设计

### 算法接口

所有算法类遵循统一接口：

```python
class RLAlgorithm(Protocol):
    """RL 算法协议"""
    
    def get_action(self, state: np.ndarray) -> Any:
        """选择动作"""
        ...
    
    def update(self, *experience) -> Dict[str, float]:
        """更新算法参数"""
        ...
    
    def save(self, path: str) -> None:
        """保存模型"""
        ...
    
    def load(self, path: str) -> None:
        """加载模型"""
        ...
```

### 环境接口

所有环境遵循 Gymnasium 接口：

```python
class RLEnv(Protocol):
    """RL 环境协议"""
    
    observation_space: gym.Space
    action_space: gym.Space
    
    def reset(self, seed=None) -> Tuple[Any, Dict]:
        """重置环境"""
        ...
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """执行动作"""
        ...
    
    def render(self) -> Any:
        """渲染环境"""
        ...
    
    def close(self) -> None:
        """关闭环境"""
        ...
```

---

## 📊 可视化架构

```
┌─────────────────────────────────────────┐
│         可视化系统                       │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────┐  ┌─────────────┐      │
│  │  实时图表    │  │  热力图      │      │
│  │  (学习曲线)  │  │  (价值函数)  │      │
│  └─────────────┘  └─────────────┘      │
│                                         │
│  ┌─────────────┐  ┌─────────────┐      │
│  │  策略图      │  │  动画        │      │
│  │  (箭头图)    │  │  (训练过程)  │      │
│  └─────────────┘  └─────────────┘      │
│                                         │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│         可视化工具                       │
├─────────────────────────────────────────┤
│  Matplotlib  │  Seaborn  │  Pygame     │
└─────────────────────────────────────────┘
```

---

## 🔐 安全设计

### 代码安全

1. **输入验证** - 所有外部输入验证
2. **异常处理** - 关键操作 try-except
3. **资源管理** - with 语句管理资源
4. **类型检查** - 类型注解 + 运行时检查

### 训练安全

1. **梯度裁剪** - 防止梯度爆炸
2. **早停机制** - 防止过拟合
3. **检查点** - 定期保存模型
4. **资源监控** - GPU 显存监控

---

## 📈 扩展性设计

### 添加新算法

1. 在对应章节创建文件
2. 实现统一接口
3. 编写测试
4. 更新文档

### 添加新环境

1. 继承 `gym.Env`
2. 实现 `reset()` 和 `step()`
3. 添加 `render()` (可选)
4. 编写测试

### 添加新游戏

1. 在 `games/` 目录创建文件
2. 实现游戏逻辑
3. 添加可视化
4. 编写说明文档

---

## 🎯 性能优化

### GPU 利用

```python
import torch

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型移动到 GPU
model = DQN().to(device)

# 数据移动到 GPU
state = torch.tensor(state).to(device)
```

### 批量处理

```python
# 批量采样
batch = replay_buffer.sample(batch_size=64)

# 批量计算
q_values = model(batch_states)  # [64, action_dim]
```

### 并行环境

```python
from gymnasium.vector import AsyncVectorEnv

# 创建 8 个并行环境
envs = AsyncVectorEnv([
    lambda: gym.make("CartPole-v1") for _ in range(8)
])

# 并行采样
observations, rewards, dones, _ = envs.step(actions)
```

---

## 📝 配置管理

### 配置文件

```yaml
# configs/default.yaml
algorithm:
  name: "q_learning"
  lr: 0.1
  gamma: 0.99
  epsilon: 0.1

environment:
  name: "CliffWalking-v0"
  max_steps: 100

training:
  num_episodes: 1000
  batch_size: 64
  use_gpu: true

visualization:
  render: true
  save_plots: true
  plot_interval: 10
```

### 加载配置

```python
import yaml

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

agent = QLearning(
    lr=config["algorithm"]["lr"],
    gamma=config["algorithm"]["gamma"],
)
```

---

*文档版本：v0.1.0*  
*最后更新：2026-04-22*
