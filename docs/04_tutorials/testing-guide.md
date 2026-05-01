# 测试指南

## 📋 测试框架

本项目使用 **pytest** 作为测试框架，包含以下测试类型：

| 测试类型 | 说明 | 标记 |
|---------|------|------|
| **单元测试** | 测试单个函数/类的功能 | `@pytest.mark.unit` |
| **集成测试** | 测试多个组件的协同工作 | `@pytest.mark.integration` |
| **回归测试** | 确保现有功能不被破坏 | `@pytest.mark.integration` |

---

## 🚀 快速开始

### 安装测试工具

```bash
make install
```

或手动安装：

```bash
pip install pytest pytest-cov black pylint
```

### 运行测试

```bash
# 运行所有测试
make test

# 运行单元测试
make test-unit

# 运行集成测试
make test-int

# 快速测试（用于开发）
make test-fast
```

---

## 📁 测试文件组织

```
tests/
├── conftest.py          # pytest 配置和 fixtures
├── test_core.py         # 核心模块测试
├── test_utils.py        # 工具模块测试
└── ...                  # 后续章节测试
```

---

## 📝 编写测试

### 测试模板

```python
"""
模块测试

测试 XXX 模块的功能
"""

import pytest
import numpy as np
from utils.core import XXX


@pytest.mark.unit
class TestXXX:
    """XXX 类测试"""
    
    def test_xxx_init(self):
        """测试初始化"""
        obj = XXX(...)
        assert obj.attr == value
    
    def test_xxx_method(self):
        """测试方法"""
        obj = XXX(...)
        result = obj.method()
        assert result == expected
    
    def test_xxx_invalid_input(self):
        """测试无效输入"""
        with pytest.raises(ValueError):
            XXX(invalid_param=value)
```

### 测试标记

```python
@pytest.mark.unit
def test_something():
    """单元测试"""
    ...

@pytest.mark.integration
def test_something_else():
    """集成测试"""
    ...

@pytest.mark.slow
def test_slow_test():
    """慢速测试（可选运行）"""
    ...
```

### Fixtures

```python
# conftest.py

@pytest.fixture
def seed():
    """随机种子 fixture"""
    return 42

@pytest.fixture
def small_env():
    """小型测试环境"""
    class DummyEnv:
        ...
    return DummyEnv()


# test_xxx.py

def test_with_fixture(small_env, seed):
    """使用 fixture"""
    env = small_env
    np.random.seed(seed)
    ...
```

---

## 🔍 代码检查

### 格式化代码

```bash
# 使用 Black 格式化
make format

# 或手动运行
black utils/ tests/
```

### 代码检查

```bash
# 运行 pylint
make lint

# 或手动运行
pylint utils/ tests/ --rcfile=.pylintrc
```

### 检查格式化

```bash
# 检查代码是否符合格式规范
black --check utils/ tests/
```

---

## 📊 代码覆盖率

```bash
# 生成覆盖率报告
make coverage

# 查看覆盖率
pytest tests/ --cov=utils --cov-report=term-missing
```

**目标覆盖率**: 70%+

---

## 🧪 测试示例

### 单元测试示例

```python
@pytest.mark.unit
class TestQLearning:
    """Q-Learning 算法测试"""
    
    def test_q_learning_init(self):
        """测试初始化"""
        agent = QLearning(state_dim=10, action_dim=4, lr=0.1)
        assert agent.q_values.shape == (10, 4)
    
    def test_get_action(self):
        """测试动作选择"""
        agent = QLearning(state_dim=10, action_dim=4, epsilon=0.1)
        action = agent.get_action(np.zeros(10))
        assert 0 <= action < 4
    
    def test_update(self):
        """测试更新"""
        agent = QLearning(state_dim=10, action_dim=4)
        experience = (np.zeros(10), 0, 1.0, np.zeros(10), False)
        td_error = agent.update(experience)
        assert isinstance(td_error, float)
```

### 集成测试示例

```python
@pytest.mark.integration
def test_training_pipeline():
    """测试完整训练流程"""
    env = gym.make("CartPole-v1")
    agent = DQN(...)
    buffer = ReplayBuffer(capacity=10000)
    
    loop = TrainingLoop(agent, env, buffer=buffer)
    stats = loop.run()
    
    assert len(stats['rewards']) > 0
    assert len(buffer) > 0
```

---

## 🎯 测试最佳实践

### 1. 测试命名

- 测试类：`Test<ClassName>`
- 测试函数：`test_<functionality>_<condition>`

### 2. 测试独立性

- 每个测试应该独立
- 使用 fixtures 提供依赖
- 不依赖测试执行顺序

### 3. 测试覆盖

- 正常情况
- 边界情况
- 异常情况

### 4. 断言清晰

```python
# ❌ 不清晰
assert result

# ✅ 清晰
assert result > 0, "结果应该大于 0"
assert len(items) == expected_count, f"期望{expected_count}个，实际{len(items)}个"
```

---

## 🐛 调试测试

### 运行单个测试

```bash
# 运行单个测试文件
pytest tests/test_core.py -v

# 运行单个测试类
pytest tests/test_core.py::TestMDP -v

# 运行单个测试函数
pytest tests/test_core.py::TestMDP::test_mdp_init -v
```

### 详细输出

```bash
# 详细输出
pytest -v

# 更详细的输出
pytest -vv

# 显示局部变量
pytest -l
```

### 提前停止

```bash
# 第一个失败就停止
pytest -x

# 失败后进入调试器
pytest --pdb
```

---

## 📈 CI/CD 集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=utils
```

---

## 📋 测试清单

在提交代码前：

- [ ] 运行所有测试：`make test`
- [ ] 代码格式化：`make format`
- [ ] 代码检查：`make lint`
- [ ] 新增测试覆盖新功能
- [ ] 更新文档（如有需要）

---

*最后更新：2026-04-22*
