# 03 - 目录结构与命名规范

## 📁 完整目录结构

```
rl-guide/
│
├── docs/                           # 架构书文档 ⭐
│   ├── INDEX.md                    # 文档索引（入口）
│   ├── 01_architecture/            # 架构设计
│   │   ├── 01-project-overview.md  # 项目概述
│   │   ├── 02-system-architecture.md # 系统架构
│   │   └── 03-directory-structure.md # 目录结构（本文件）
│   ├── 02_chapters/                # 章节详细设计
│   │   ├── chapter_01_mdp.md
│   │   ├── chapter_02_dp.md
│   │   └── ...
│   ├── 03_api_reference/           # API 参考
│   ├── 04_tutorials/               # 教程
│   └── 05_design_decisions/        # 设计决策
│
├── chapter_01_mdp_fundamentals/    # 第 1 章：MDP 基础
│   ├── README.md                   # 章节说明
│   ├── 01_mdp_core.py              # MDP 核心类
│   ├── 02_bellman_equation.py      # 贝尔曼方程
│   ├── 03_value_function.py        # 价值函数
│   ├── 04_policy.py                # 策略定义
│   └── games/                      # 应用场景
│       ├── lunch_decision.py       # 🍱 午餐选择器
│       └── commute_planner.py      # 🚇 下班路线规划
│
├── chapter_02_dynamic_programming/ # 第 2 章：动态规划
│   ├── README.md
│   ├── 01_policy_evaluation.py
│   ├── 02_policy_iteration.py
│   ├── 03_value_iteration.py
│   ├── 04_mcts.py                  # ⭐ MCTS
│   └── games/
│       ├── gridworld_nav.py
│       ├── warehouse_robot.py
│       └── alphago_simple.py
│
├── chapter_03_monte_carlo/         # 第 3 章：蒙特卡洛
│   ├── README.md
│   ├── 01_mc_prediction.py
│   ├── 02_mc_control.py
│   ├── 03_on_off_policy.py
│   └── games/
│       ├── blackjack.py
│       ├── slot_machine.py
│       └── flight_chess.py
│
├── chapter_04_temporal_difference/ # 第 4 章：时序差分 ⭐核心
│   ├── README.md
│   ├── 01_td_prediction.py
│   ├── 02_sarsa.py
│   ├── 03_q_learning.py
│   ├── 04_expected_sarsa.py
│   ├── 05_exploration_strategies.py
│   └── games/
│       ├── cliff_walking.py
│       ├── maze_treasure.py
│       └── snake_simple.py
│
├── chapter_05_function_approximation/ # 第 5 章：函数近似
│   ├── README.md
│   ├── 01_linear_approximation.py
│   ├── 02_neural_network_q.py
│   ├── 03_dqn.py
│   └── games/
│       ├── cartpole_balance.py
│       ├── breakout_atari.py
│       └── lunar_lander.py
│
├── chapter_06_policy_gradient/     # 第 6 章：策略梯度
│   ├── README.md
│   ├── 01_reinforce.py
│   ├── 02_actor_critic.py
│   ├── 03_a2c.py
│   ├── 04_ddpg.py                  # ⭐ DDPG
│   ├── 05_td3.py                   # ⭐ TD3
│   └── games/
│       ├── car_racing.py
│       ├── robotic_arm.py
│       └── pong_simple.py
│
├── chapter_07_advanced_policy/     # 第 7 章：高级策略 ⭐目标
│   ├── README.md
│   ├── 01_trpo.py
│   ├── 02_ppo.py
│   ├── 03_sac.py                   # ⭐ SAC
│   ├── 04_dpo_intuition.py
│   ├── 05_offline_rl_intro.py
│   └── games/
│       ├── dialogue_optimization.py
│       ├── text_generation.py
│       └── multi_objective.py
│
├── utils/                          # 工具函数
│   ├── __init__.py
│   ├── core/                       # 核心抽象
│   │   ├── __init__.py
│   │   ├── mdp.py
│   │   ├── policy.py
│   │   ├── value_function.py
│   │   └── replay_buffer.py
│   ├── visualization.py            # 可视化工具
│   ├── metrics.py                  # 性能指标
│   ├── env_wrapper.py              # 环境包装器
│   └── training_loop.py            # 训练循环
│
├── tests/                          # 测试
│   ├── __init__.py
│   ├── conftest.py                 # pytest 配置
│   ├── test_core.py                # 核心类测试
│   ├── test_chapter_01.py
│   ├── test_chapter_02.py
│   └── ...
│
├── notebooks/                      # Jupyter 笔记本
│   ├── chapter_01_intro.ipynb
│   ├── chapter_02_dp.ipynb
│   └── ...
│
├── configs/                        # 配置文件
│   ├── default.yaml
│   ├── chapter_01.yaml
│   └── ...
│
├── scripts/                        # 辅助脚本
│   ├── run_chapter.py              # 运行章节示例
│   ├── plot_results.py             # 绘制结果
│   └── download_models.py          # 下载模型
│
├── README.md                       # 项目说明
├── requirements.txt                # Python 依赖
├── setup.py                        # 安装配置（可选）
├── .gitignore                      # Git 忽略
├── LICENSE                         # 许可证
└── .hermes/                        # Hermes 工具
    └── plans/                      # 计划文档
        ├── 2026-04-22_RL-基础课程仓库计划.md
        └── 2026-04-22_RL-资源调研报告.md
```

---

## 📝 命名规范

### 目录命名

| 类型 | 规范 | 示例 |
|------|------|------|
| **章节目录** | `chapter_XX_topic_name/` | `chapter_01_mdp_fundamentals/` |
| **子目录** | `snake_case` | `games/`, `utils/` |
| **文档目录** | `XX_category/` | `01_architecture/` |

### 文件命名

| 类型 | 规范 | 示例 |
|------|------|------|
| **Python 模块** | `snake_case.py` | `q_learning.py`, `value_function.py` |
| **文档文件** | `kebab-case.md` | `project-overview.md` |
| **配置文件** | `snake_case.yaml` | `default.yaml` |
| **测试文件** | `test_*.py` | `test_chapter_01.py` |
| **Notebook** | `chapter_XX_*.ipynb` | `chapter_01_intro.ipynb` |

### 章节内文件编号

```
chapter_01_mdp_fundamentals/
├── 01_mdp_core.py              # 核心概念
├── 02_bellman_equation.py      # 核心理论
├── 03_value_function.py        # 核心算法
├── 04_policy.py                # 核心算法
└── games/
    ├── 01_lunch_decision.py    # 游戏示例（可选编号）
    └── 02_commute_planner.py
```

**编号规则**：
- `01-09`: 核心概念和理论
- `10-19`: 算法实现
- `20-29`: 扩展内容
- `games/`: 游戏示例（可编号可不编号）

---

### 类命名

| 类型 | 规范 | 示例 |
|------|------|------|
| **核心类** | `PascalCase` | `MDP`, `Policy`, `QLearning` |
| **异常类** | `PascalCase` + `Error` | `MLError`, `TrainingError` |
| **工具类** | `PascalCase` | `ReplayBuffer`, `TrainingLoop` |

**示例**：
```python
class MDP:
    """马尔可夫决策过程"""
    ...

class QLearning:
    """Q-Learning 算法"""
    ...

class ReplayBuffer:
    """经验回放缓冲区"""
    ...
```

---

### 函数命名

| 类型 | 规范 | 示例 |
|------|------|------|
| **公共函数** | `snake_case` | `get_action()`, `update()` |
| **私有函数** | `_snake_case` | `_sample_batch()`, `_compute_td_error()` |
| **魔术方法** | `__snake_case__` | `__init__()`, `__call__()` |

**示例**：
```python
class QLearning:
    def __init__(self, state_dim, action_dim):
        ...
    
    def get_action(self, state):
        """选择动作（公共方法）"""
        ...
    
    def update(self, experience):
        """更新 Q 表（公共方法）"""
        ...
    
    def _compute_td_error(self, batch):
        """计算 TD 误差（私有方法）"""
        ...
```

---

### 变量命名

| 类型 | 规范 | 示例 |
|------|------|------|
| **普通变量** | `snake_case` | `state`, `action`, `reward` |
| **常量** | `UPPER_CASE` | `MAX_STEPS`, `LEARNING_RATE` |
| **私有变量** | `_snake_case` | `_q_table`, `_epsilon` |
| **类变量** | `_snake_case` | `_instance_count` |

**示例**：
```python
# 常量
MAX_EPISODES = 1000
LEARNING_RATE = 0.1

# 普通变量
state = env.reset()
action = agent.get_action(state)

# 私有变量
class QLearning:
    def __init__(self):
        self._q_table = {}
        self._epsilon = 0.1
```

---

## 📋 代码组织规范

### 模块结构

```python
"""
模块名称：q_learning.py

简要描述：Q-Learning 算法实现

详细说明：
    本模块实现了经典的 Q-Learning 算法，包括：
    - Q 表初始化
    - ε-greedy 策略
    - TD 更新规则
    
示例:
    >>> from chapter_04.q_learning import QLearning
    >>> agent = QLearning(state_dim=10, action_dim=4)
    >>> action = agent.get_action(state)
"""

# 标准库导入
import numpy as np
from typing import Dict, Tuple, Optional

# 第三方库导入
import torch

# 本地导入
from utils.core import Policy
from utils.visualization import plot_learning_curve

# 常量定义
DEFAULT_LR = 0.1
DEFAULT_GAMMA = 0.99

# 类定义
class QLearning(Policy):
    """Q-Learning 算法实现"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = DEFAULT_LR,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = 0.1,
    ):
        """
        初始化 Q-Learning
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: 探索率
        """
        ...
    
    def get_action(self, state: np.ndarray) -> int:
        """选择动作"""
        ...
    
    def update(self, experience: Tuple) -> float:
        """更新 Q 表"""
        ...


# 辅助函数
def create_q_table(state_dim: int, action_dim: int) -> np.ndarray:
    """创建 Q 表"""
    ...


# 测试代码（在 if __name__ == "__main__" 中）
if __name__ == "__main__":
    # 简单测试
    agent = QLearning(state_dim=10, action_dim=4)
    print("QLearning 测试通过！")
```

---

### Docstring 规范

**Google 风格**：

```python
def function_name(arg1: str, arg2: int) -> float:
    """
    函数简短描述
    
    详细描述（可选）：
        可以更详细地解释函数的功能和用途
        
    Args:
        arg1: 参数 1 的描述
        arg2: 参数 2 的描述
        
    Returns:
        返回值的描述
        
    Raises:
        ValueError: 当 arg1 为空时
        TypeError: 当 arg2 不是整数时
        
    Example:
        >>> result = function_name("hello", 42)
        >>> print(result)
        0.5
        
    Note:
        注意事项（可选）
        
    Reference:
        Sutton, R. S., & Barto, A. G. (2018). 
        Reinforcement learning: An introduction. MIT press.
    """
    ...
```

---

## 📁 特殊目录说明

### `.hermes/` 目录

**用途**：Hermes Agent 工具和计划文档

```
.hermes/
└── plans/
    ├── 2026-04-22_RL-基础课程仓库计划.md  # 项目计划
    └── 2026-04-22_RL-资源调研报告.md      # 调研报告
```

**注意**：此目录在 `.gitignore` 中，不提交到 Git

---

### `notebooks/` 目录

**用途**：Jupyter 笔记本，交互式学习

```
notebooks/
├── chapter_01_intro.ipynb      # 第 1 章交互教程
├── chapter_02_dp.ipynb         # 第 2 章交互教程
└── ...
```

**内容**：
- 代码示例
- 可视化图表
- 练习题
- 解答

---

### `configs/` 目录

**用途**：配置文件

```
configs/
├── default.yaml                # 默认配置
├── chapter_01.yaml             # 第 1 章配置
├── chapter_02.yaml             # 第 2 章配置
└── ...
```

**配置示例**：
```yaml
# configs/chapter_04.yaml
algorithm:
  name: "q_learning"
  lr: 0.1
  gamma: 0.99
  epsilon: 0.1
  epsilon_decay: 0.995
  epsilon_min: 0.01

environment:
  name: "CliffWalking-v0"
  max_steps: 100

training:
  num_episodes: 1000
  seed: 42
  use_gpu: false

visualization:
  render: true
  save_plots: true
  plot_interval: 10
```

---

### `scripts/` 目录

**用途**：辅助脚本

```
scripts/
├── run_chapter.py              # 运行章节示例
├── plot_results.py             # 绘制结果
├── download_models.py          # 下载预训练模型
└── benchmark.py                # 性能测试
```

**使用示例**：
```bash
# 运行第 4 章示例
python scripts/run_chapter.py chapter_04 --game cliff_walking

# 绘制学习曲线
python scripts/plot_results.py --output figures/learning_curve.png

# 下载预训练模型
python scripts/download_models.py --chapter chapter_05
```

---

## 🔧 Git 工作流

### 分支策略

```
main (保护分支)
  ↑
  ├── develop (开发分支)
  │     ↑
  │     ├── feature/chapter-01 (功能分支)
  │     ├── feature/chapter-02
  │     └── fix/bug-001 (修复分支)
  │
  └── docs/architecture (文档分支)
```

### 提交规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**type**:
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具

**scope**:
- `chapter_01`, `chapter_02`, ...
- `utils`, `tests`, `docs`
- `config`, `scripts`

**示例**：
```
feat(chapter_04): 实现 Q-Learning 算法和悬崖行走游戏

- 添加 QLearning 类
- 实现 ε-greedy 策略
- 添加悬崖行走游戏示例
- 添加可视化功能
- 编写单元测试

Closes #4
```

---

## 📊 文件大小指导

| 文件类型 | 最大行数 | 说明 |
|---------|---------|------|
| **算法类** | 200 行 | 保持简洁，单一职责 |
| **游戏文件** | 300 行 | 可包含游戏逻辑 + 渲染 |
| **工具模块** | 400 行 | 可包含多个辅助函数 |
| **测试文件** | 500 行 | 可包含多个测试用例 |
| **文档文件** | 1000 行 | 如过长考虑拆分 |

---

*文档版本：v0.1.0*  
*最后更新：2026-04-22*
