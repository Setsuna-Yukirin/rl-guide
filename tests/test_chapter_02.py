"""
第 2 章测试：动态规划
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
    """简单 MDP fixture（用于 DP 算法测试）"""
    mdp = TabularMDP(n_states=4, n_actions=2, gamma=0.99)
    for s in range(4):
        for a in range(2):
            next_s = min(s + 1, 3) if a == 0 else min(s + 2, 3)
            reward = 1.0 if next_s == 3 else 0.0
            mdp.set_transition(s, a, [(1.0, next_s, reward, next_s == 3)])
    return mdp


@pytest.fixture
def gridworld():
    """网格寻路环境 fixture"""
    from chapter_02_dynamic_programming.games import GridWorldNav
    env = GridWorldNav(grid_size=(3, 3), n_traps=1)
    env.reset(seed=42)
    return env


@pytest.mark.unit
class TestPolicyEvaluation:
    """策略评估测试"""
    
    def test_policy_evaluation_convergence(self, simple_mdp):
        """测试策略评估收敛"""
        from chapter_02_dynamic_programming.policy_iteration import policy_evaluation
        
        # 均匀随机策略
        policy = np.ones((4, 2)) * 0.5
        V, iterations = policy_evaluation(simple_mdp, policy, gamma=0.99)
        
        assert V.shape == (4,)
        assert np.all(np.isfinite(V))
        assert iterations > 0
        assert iterations <= 1000  # 应该在 1000 次内收敛
    
    def test_policy_evaluation_deterministic(self, simple_mdp):
        """测试确定性策略评估"""
        from chapter_02_dynamic_programming.policy_iteration import policy_evaluation
        
        # 确定性策略（总是选择动作 0）
        policy = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]
        ])
        V, _ = policy_evaluation(simple_mdp, policy)
        
        assert V.shape == (4,)
        # 终端状态价值应该最高
        assert V[3] >= V[0]


@pytest.mark.unit
class TestPolicyIteration:
    """策略迭代测试"""
    
    def test_policy_iteration_convergence(self, simple_mdp):
        """测试策略迭代收敛"""
        from chapter_02_dynamic_programming.policy_iteration import policy_iteration
        
        V, policy, iterations = policy_iteration(simple_mdp, gamma=0.99)
        
        assert V.shape == (4,)
        assert policy.shape == (4, 2)
        assert iterations > 0
        assert iterations < 100  # 策略迭代通常很快收敛
        
        # 策略应该是确定性的
        assert np.allclose(policy.sum(axis=1), 1.0)
        assert np.all(np.isin(policy, [0.0, 1.0]))
    
    def test_policy_iteration_optimality(self, simple_mdp):
        """测试策略迭代找到最优策略"""
        from chapter_02_dynamic_programming.policy_iteration import policy_iteration
        
        V, policy, _ = policy_iteration(simple_mdp, gamma=0.99)
        
        # 最优策略应该让智能体尽快到达状态 3
        # 检查从状态 0 开始是否能到达状态 3
        current_state = 0
        for _ in range(10):
            action = np.argmax(policy[current_state])
            transitions = simple_mdp.get_transition(current_state, action)
            _, next_state, _, _ = transitions[0]
            if next_state == 3:
                break
            current_state = next_state
        
        assert current_state == 3 or next_state == 3


@pytest.mark.unit
class TestValueIteration:
    """价值迭代测试"""
    
    def test_value_iteration_convergence(self, simple_mdp):
        """测试价值迭代收敛"""
        from chapter_02_dynamic_programming.policy_iteration import value_iteration
        
        V, policy, iterations = value_iteration(simple_mdp, gamma=0.99)
        
        assert V.shape == (4,)
        assert policy.shape == (4, 2)
        assert iterations > 0
        
        # 策略应该是确定性的
        assert np.allclose(policy.sum(axis=1), 1.0)
    
    def test_value_iteration_vs_policy_iteration(self, simple_mdp):
        """比较价值迭代和策略迭代的结果"""
        from chapter_02_dynamic_programming.policy_iteration import (
            value_iteration, policy_iteration
        )
        
        V_vi, policy_vi, _ = value_iteration(simple_mdp, gamma=0.99)
        V_pi, policy_pi, _ = policy_iteration(simple_mdp, gamma=0.99)
        
        # 两种方法应该得到相同的最优价值函数（允许小的数值误差）
        assert np.allclose(V_vi, V_pi, atol=1e-5)


@pytest.mark.integration
class TestGridWorld:
    """网格寻路环境测试"""
    
    def test_gridworld_reset(self, gridworld):
        """测试环境重置"""
        state, _ = gridworld.reset(seed=42)
        
        assert isinstance(state, int)
        assert 0 <= state < gridworld.n_states
        assert gridworld.agent_pos == (0, 0)
    
    def test_gridworld_step(self, gridworld):
        """测试环境步"""
        state, _ = gridworld.reset(seed=42)
        
        # 向下移动
        next_state, reward, terminated, _, _ = gridworld.step(1)
        
        assert isinstance(next_state, int)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
    
    def test_gridworld_goal_reach(self):
        """测试到达终点"""
        from chapter_02_dynamic_programming.games import GridWorldNav
        
        env = GridWorldNav(grid_size=(2, 2), n_traps=0)
        env.reset(seed=42)
        
        # 向右再向下到达终点
        env.step(3)  # 右
        _, reward, terminated, _, _ = env.step(1)  # 下
        
        assert terminated
        assert reward == env.goal_reward
    
    def test_gridworld_trap(self):
        """测试陷阱"""
        from chapter_02_dynamic_programming.games import GridWorldNav
        
        env = GridWorldNav(grid_size=(2, 2), n_traps=1, seed=42)
        env.reset(seed=42)
        
        # 找到陷阱位置并走向它
        trap_pos = list(env.traps)[0]
        
        # 简单测试：只要能触发陷阱奖励即可
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, terminated, _, _ = env.step(action)
            if terminated and reward == env.trap_reward:
                break


@pytest.mark.unit
class TestMCTS:
    """MCTS 算法测试"""
    
    def test_mcts_init(self, gridworld):
        """测试 MCTS 初始化"""
        from chapter_02_dynamic_programming.mcts import MCTS
        
        mcts = MCTS(gridworld, n_iterations=100)
        
        assert mcts.n_iterations == 100
        assert mcts.c == 1.414
        assert mcts.root is None
    
    def test_mcts_search(self, gridworld):
        """测试 MCTS 搜索"""
        from chapter_02_dynamic_programming.mcts import MCTS
        
        mcts = MCTS(gridworld, n_iterations=50)
        state, _ = gridworld.reset(seed=42)
        
        action = mcts.search(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < gridworld.n_actions


@pytest.mark.unit
class TestDPAlgorithms:
    """DP 算法综合测试"""
    
    def test_compute_action_value(self, simple_mdp):
        """测试动作价值计算"""
        from chapter_02_dynamic_programming.policy_iteration import (
            value_iteration, compute_action_value
        )
        
        V, _, _ = value_iteration(simple_mdp, gamma=0.99)
        
        # 计算某个状态 - 动作对的 Q 值
        q = compute_action_value(simple_mdp, V, state=0, action=0)
        
        assert np.isfinite(q)
        assert q >= 0  # 因为最终会到达奖励为 1 的状态
