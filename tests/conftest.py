# pytest 配置文件

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line(
        "markers", "unit: 单元测试标记"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试标记"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试标记"
    )


@pytest.fixture
def seed():
    """随机种子 fixture"""
    return 42


@pytest.fixture
def small_env():
    """小型测试环境"""
    class DummyEnv:
        def __init__(self):
            self.state_dim = 4
            self.action_dim = 2
            self._state = None
        
        def reset(self, seed=None):
            if seed is not None:
                np.random.seed(seed)
            self._state = np.zeros(self.state_dim)
            return self._state, {}
        
        def step(self, action):
            self._state = np.random.randn(self.state_dim)
            reward = np.random.randn()
            done = np.random.rand() < 0.1
            return self._state, reward, done, False, {}
        
        def render(self):
            pass
    
    return DummyEnv()
