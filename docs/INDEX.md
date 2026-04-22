# rl-guide 架构书

> **强化学习指北 - 从经典算法到 LLM 后训练的完整学习路径**

**版本**：v0.1.0  
**创建时间**：2026-04-22  
**作者**：Hermes <neko_yukirin@qq.com>  
**状态**：📝 编写中

---

## 📚 文档目录

### 第一部分：架构设计

| 文档 | 说明 | 状态 |
|------|------|------|
| [01-项目概述](01_architecture/01-project-overview.md) | 项目目标、核心理念、技术栈 | ✅ 已完成 |
| [02-系统架构](01_architecture/02-system-architecture.md) | 整体架构、模块划分、数据流 | 📝 编写中 |
| [03-目录结构](01_architecture/03-directory-structure.md) | 文件组织、命名规范 | 📝 编写中 |

### 第二部分：章节详细设计

| 文档 | 说明 | 状态 |
|------|------|------|
| [第 1 章：MDP 基础](02_chapters/chapter_01_mdp.md) | MDP 五元组、贝尔曼方程、价值函数 | 📝 待编写 |
| [第 2 章：动态规划](02_chapters/chapter_02_dp.md) | 策略迭代、价值迭代、MCTS | 📝 待编写 |
| [第 3 章：蒙特卡洛](02_chapters/chapter_03_mc.md) | MC 预测、MC 控制、On/Off-policy | 📝 待编写 |
| [第 4 章：时序差分](02_chapters/chapter_04_td.md) | SARSA、Q-Learning、探索策略 | 📝 待编写 |
| [第 5 章：函数近似](02_chapters/chapter_05_fa.md) | 线性近似、DQN、经验回放 | 📝 待编写 |
| [第 6 章：策略梯度](02_chapters/chapter_06_pg.md) | REINFORCE、Actor-Critic、DDPG、TD3 | 📝 待编写 |
| [第 7 章：高级策略](02_chapters/chapter_07_advanced.md) | TRPO、PPO、SAC、DPO、离线 RL | 📝 待编写 |

### 第三部分：API 参考

| 文档 | 说明 | 状态 |
|------|------|------|
| [核心类 API](03_api_reference/core_classes.md) | MDP、Policy、ValueFunction 等核心类 | 📝 待编写 |
| [算法 API](03_api_reference/algorithms.md) | 各算法类的接口文档 | 📝 待编写 |
| [环境 API](03_api_reference/environments.md) | 游戏环境的接口文档 | 📝 待编写 |
| [工具函数 API](03_api_reference/utils.md) | 可视化、指标计算等工具 | 📝 待编写 |

### 第四部分：教程

| 文档 | 说明 | 状态 |
|------|------|------|
| [快速开始](04_tutorials/quickstart.md) | 安装、配置、运行第一个示例 | 📝 待编写 |
| [开发指南](04_tutorials/development-guide.md) | 如何添加新算法、新环境 | 📝 待编写 |
| [测试指南](04_tutorials/testing-guide.md) | 如何编写测试、运行测试 | 📝 待编写 |

### 第五部分：设计决策

| 文档 | 说明 | 状态 |
|------|------|------|
| [ADR-001: 为什么选择 Gymnasium](05_design_decisions/adr-001-gymnasium.md) | 环境库选型理由 | 📝 待编写 |
| [ADR-002: Class 模块化设计](05_design_decisions/adr-002-class-design.md) | 为什么每个算法一个 class | 📝 待编写 |
| [ADR-003: 游戏化学习](05_design_decisions/adr-003-gamification.md) | 为什么每章都要有游戏 | 📝 待编写 |

---

## 🎯 开发工作流

### 迭代流程

```
1. 阅读架构书章节设计
   ↓
2. 明确本次迭代目标（写在 issue/plan 中）
   ↓
3. 编写测试（测试先行）
   ├── 回归测试（确保现有功能正常）
   └── 功能测试（验证新功能）
   ↓
4. 实现代码
   ├── 完整功能实现
   ├── 清晰注释（docstring + 行内注释）
   └── 可调参数（快速验证模式）
   ↓
5. 运行测试
   ├── 所有测试通过
   └── 代码能跑通
   ↓
6. Git 提交
   ├── git add .
   ├── git commit -m "清晰的提交信息"
   └── git push
   ↓
7. 更新架构书（如有变更）
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

**示例**:
```
feat(chapter_01): 实现 MDP 核心类和午餐选择器游戏

- 添加 MDP、State、Action 类
- 实现贝尔曼期望方程
- 添加午餐选择器游戏示例
- 添加可视化功能

Closes #1
```

---

## 📋 当前进度

| Phase | 内容 | 状态 | 预计时间 |
|-------|------|------|---------|
| Phase 0 | 架构书编写 | 🟡 进行中 | 1 天 |
| Phase 1 | 基础框架 | ⚪ 未开始 | 1-2 天 |
| Phase 2 | 第 1-2 章 | ⚪ 未开始 | 3-4 天 |
| Phase 3 | 第 3-4 章 ⭐ | ⚪ 未开始 | 4-5 天 |
| Phase 4 | 第 5-6 章 | ⚪ 未开始 | 4-5 天 |
| Phase 5 | 第 7 章 ⭐ | ⚪ 未开始 | 5-6 天 |

---

## 🔧 技术栈

### 核心依赖

| 库 | 版本 | 用途 |
|----|------|------|
| Python | 3.9+ | 编程语言 |
| NumPy | >=1.24.0 | 数值计算 |
| Gymnasium | >=0.29.0 | RL 环境 |
| Matplotlib | >=3.7.0 | 可视化 |
| PyTorch | >=2.0.0 | 深度学习 |

### 开发工具

| 工具 | 用途 |
|------|------|
| pytest | 测试框架 |
| black | 代码格式化 |
| Jupyter | 交互式笔记本 |
| Git + GitHub | 版本控制 |

### 硬件环境

- **GPU**: NVIDIA RTX 3080
- **用途**: 深度学习模型训练
- **注意**: 训练参数可调，支持快速验证模式

---

## 📞 联系方式

- **GitHub**: https://github.com/Setsuna-Yukirin/rl-guide
- **Issues**: https://github.com/Setsuna-Yukirin/rl-guide/issues

---

*最后更新：2026-04-22*
