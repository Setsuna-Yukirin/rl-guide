"""
第 1 章测试：MDP 基础
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.core import TabularMDP


@pytest.fixture
def simple_mdp():
    """简单 MDP fixture"""
    mdp = TabularMDP(n_states=4, n_actions=2, gamma=0.99)
    for s in range(4):
        for a in range(2):
            next_s = min(s + 1, 3) if a == 0 else min(s + 2, 3)
            reward = 1.0 if next_s == 3 else 0.0
            mdp.set_transition(s, a, [(1.0, next_s, reward, next_s == 3)])
    return mdp


@pytest.mark.unit
class TestValueFunction:
    """价值函数测试"""
    
    def test_compute_V_pi(self, simple_mdp):
        """测试 V_π 计算"""
        from chapter_01_mdp_fundamentals import value_function
        
        policy = np.ones((4, 2)) * 0.5
        V = value_function.compute_V_pi(simple_mdp, policy)
        assert V.shape == (4,)
        assert np.all(np.isfinite(V))
    
    def test_policy_iteration(self, simple_mdp):
        """测试策略迭代"""
        from chapter_01_mdp_fundamentals import value_function
        
        V, policy, iters = value_function.policy_iteration(simple_mdp)
        assert V.shape == (4,)
        assert policy.shape == (4, 2)
        assert iters > 0


@pytest.mark.integration
class TestLunchEnv:
    """午餐环境测试"""
    
    def test_lunch_env_step(self):
        """测试环境 step"""
        from chapter_01_mdp_fundamentals.games import lunch_decision
        
        env = lunch_decision.LunchDecisionEnv()
        state, _ = env.reset(seed=42)
        assert state.shape == (3,)
        
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
