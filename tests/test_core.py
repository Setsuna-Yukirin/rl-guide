"""
核心模块单元测试

测试 utils/core/ 下的所有类：
- MDP, TabularMDP
- Policy 及其子类
- ValueFunction 及其子类
- ReplayBuffer 及其子类
"""

import pytest
import numpy as np
from utils.core import (
    MDP,
    TabularMDP,
    RandomPolicy,
    EpsilonGreedyPolicy,
    GreedyPolicy,
    TabularVFunction,
    TabularQFunction,
    ReplayBuffer,
    PriorityReplayBuffer,
)


# ============== MDP 测试 ==============

@pytest.mark.unit
class TestMDP:
    """MDP 类测试"""
    
    def test_mdp_init(self):
        """测试 MDP 初始化"""
        mdp = MDP(state_dim=10, action_dim=4, gamma=0.99)
        assert mdp.state_dim == 10
        assert mdp.action_dim == 4
        assert mdp.gamma == 0.99
    
    def test_mdp_invalid_gamma(self):
        """测试无效的 gamma 值"""
        with pytest.raises(ValueError):
            MDP(state_dim=10, action_dim=4, gamma=0)
        with pytest.raises(ValueError):
            MDP(state_dim=10, action_dim=4, gamma=1.5)
    
    def test_mdp_abstract_methods(self):
        """测试抽象方法"""
        mdp = MDP(state_dim=10, action_dim=4)
        with pytest.raises(NotImplementedError):
            mdp.step(0, 0)
        with pytest.raises(NotImplementedError):
            mdp.reset()


@pytest.mark.unit
class TestTabularMDP:
    """TabularMDP 类测试"""
    
    def test_tabular_mdp_init(self):
        """测试表格型 MDP 初始化"""
        mdp = TabularMDP(n_states=4, n_actions=2)
        assert mdp.nS == 4
        assert mdp.nA == 2
    
    def test_set_transition(self):
        """测试设置转移概率"""
        mdp = TabularMDP(n_states=4, n_actions=2)
        mdp.set_transition(0, 0, [
            (0.8, 1, -1.0, False),
            (0.2, 0, -1.0, False),
        ])
        assert 0 in mdp.P
        assert 0 in mdp.P[0]
    
    def test_set_transition_invalid_prob(self):
        """测试无效的转移概率"""
        mdp = TabularMDP(n_states=4, n_actions=2)
        with pytest.raises(ValueError):
            mdp.set_transition(0, 0, [
                (0.5, 1, -1.0, False),  # 概率和不为 1
            ])
    
    def test_step(self):
        """测试 step 方法"""
        mdp = TabularMDP(n_states=4, n_actions=2)
        mdp.set_transition(0, 0, [(1.0, 1, -1.0, False)])
        
        next_state, reward, done, info = mdp.step(0, 0)
        assert next_state == 1
        assert reward == -1.0
        assert done == False
    
    def test_reset(self):
        """测试 reset 方法"""
        mdp = TabularMDP(n_states=4, n_actions=2)
        state, info = mdp.reset(seed=42)
        assert 0 <= state < 4


# ============== Policy 测试 ==============

@pytest.mark.unit
class TestRandomPolicy:
    """RandomPolicy 类测试"""
    
    def test_random_policy_init(self):
        """测试随机策略初始化"""
        policy = RandomPolicy(state_dim=4, action_dim=2)
        assert policy.state_dim == 4
        assert policy.action_dim == 2
    
    def test_get_action(self):
        """测试动作选择"""
        policy = RandomPolicy(state_dim=4, action_dim=2, seed=42)
        action = policy.get_action(np.zeros(4))
        assert action in [0, 1]
    
    def test_get_action_prob(self):
        """测试动作概率"""
        policy = RandomPolicy(state_dim=4, action_dim=4)
        prob = policy.get_action_prob(np.zeros(4), 0)
        assert prob == 0.25


@pytest.mark.unit
class TestEpsilonGreedyPolicy:
    """EpsilonGreedyPolicy 类测试"""
    
    def test_epsilon_greedy_init(self):
        """测试 ε-greedy 策略初始化"""
        policy = EpsilonGreedyPolicy(state_dim=10, action_dim=4, epsilon=0.1)
        assert policy.epsilon == 0.1
    
    def test_invalid_epsilon(self):
        """测试无效的 epsilon"""
        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(state_dim=10, action_dim=4, epsilon=-0.1)
        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(state_dim=10, action_dim=4, epsilon=1.5)
    
    def test_set_q_values(self):
        """测试设置 Q 值"""
        policy = EpsilonGreedyPolicy(state_dim=10, action_dim=4)
        q_values = np.random.randn(10, 4)
        policy.set_q_values(q_values)
        assert np.array_equal(policy.q_values, q_values)
    
    def test_get_action_greedy(self):
        """测试贪心动作选择"""
        policy = EpsilonGreedyPolicy(state_dim=10, action_dim=4, epsilon=0.0)
        q_values = np.random.randn(10, 4)
        policy.set_q_values(q_values)
        
        # 贪心策略应该总是选择最优动作
        for _ in range(10):
            action = policy.get_action(np.array([0]))
            assert action == np.argmax(q_values[0])
    
    def test_decay_epsilon(self):
        """测试 epsilon 衰减"""
        policy = EpsilonGreedyPolicy(state_dim=10, action_dim=4, epsilon=0.5)
        policy.decay_epsilon(decay_rate=0.9, min_epsilon=0.01)
        assert policy.epsilon == 0.45


@pytest.mark.unit
class TestGreedyPolicy:
    """GreedyPolicy 类测试"""
    
    def test_greedy_policy(self):
        """测试贪心策略"""
        policy = GreedyPolicy(state_dim=10, action_dim=4)
        q_values = np.random.randn(10, 4)
        policy.set_q_values(q_values)
        
        # 总是选择最优动作
        for _ in range(10):
            action = policy.get_action(np.array([0]))
            assert action == np.argmax(q_values[0])


# ============== ValueFunction 测试 ==============

@pytest.mark.unit
class TestTabularVFunction:
    """TabularVFunction 类测试"""
    
    def test_v_function_init(self):
        """测试 V 函数初始化"""
        V = TabularVFunction(state_dim=10)
        assert V.state_dim == 10
        assert len(V.values) == 10
    
    def test_get_value(self):
        """测试获取价值"""
        V = TabularVFunction(state_dim=10, init_value=5.0)
        value = V.get_value(0)
        assert value == 5.0
    
    def test_update(self):
        """测试更新"""
        V = TabularVFunction(state_dim=10)
        V.update(state=0, target=10.0, lr=0.1)
        assert V.values[0] == 1.0  # 0 + 0.1 * (10 - 0)


@pytest.mark.unit
class TestTabularQFunction:
    """TabularQFunction 类测试"""
    
    def test_q_function_init(self):
        """测试 Q 函数初始化"""
        Q = TabularQFunction(state_dim=10, action_dim=4)
        assert Q.q_values.shape == (10, 4)
    
    def test_get_value(self):
        """测试获取 Q 值"""
        Q = TabularQFunction(state_dim=10, action_dim=4, init_value=5.0)
        value = Q.get_value(0, 1)
        assert value == 5.0
    
    def test_get_best_action(self):
        """测试获取最优动作"""
        Q = TabularQFunction(state_dim=10, action_dim=4)
        q_values = np.random.randn(10, 4)
        Q.set_q_values(q_values)
        
        best_action = Q.get_best_action(0)
        assert best_action == np.argmax(q_values[0])


# ============== ReplayBuffer 测试 ==============

@pytest.mark.unit
class TestReplayBuffer:
    """ReplayBuffer 类测试"""
    
    def test_buffer_init(self):
        """测试缓冲区初始化"""
        buffer = ReplayBuffer(capacity=1000)
        assert buffer.capacity == 1000
        assert len(buffer) == 0
    
    def test_add(self):
        """测试添加经验"""
        buffer = ReplayBuffer(capacity=1000)
        buffer.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert len(buffer) == 1
    
    def test_sample(self):
        """测试采样"""
        buffer = ReplayBuffer(capacity=1000)
        for i in range(100):
            buffer.add(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
        
        batch = buffer.sample(batch_size=32)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (32, 4)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
    
    def test_sample_insufficient(self):
        """测试经验不足时采样"""
        buffer = ReplayBuffer(capacity=1000)
        buffer.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
        
        with pytest.raises(ValueError):
            buffer.sample(batch_size=10)
    
    def test_clear(self):
        """测试清空"""
        buffer = ReplayBuffer(capacity=1000)
        for i in range(100):
            buffer.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
        
        buffer.clear()
        assert len(buffer) == 0


@pytest.mark.unit
class TestPriorityReplayBuffer:
    """PriorityReplayBuffer 类测试"""
    
    def test_pri_buffer_init(self):
        """测试优先级缓冲区初始化"""
        buffer = PriorityReplayBuffer(capacity=1000)
        assert buffer.capacity == 1000
    
    def test_add_and_sample(self):
        """测试添加和采样"""
        buffer = PriorityReplayBuffer(capacity=1000)
        for i in range(100):
            buffer.add(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
        
        batch, indices, weights = buffer.sample(batch_size=32)
        assert len(indices) == 32
        assert len(weights) == 32
        assert all(w > 0 for w in weights)
    
    def test_update_priorities(self):
        """测试更新优先级"""
        buffer = PriorityReplayBuffer(capacity=1000)
        for i in range(100):
            buffer.add(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
        
        batch, indices, _ = buffer.sample(batch_size=32)
        td_errors = np.abs(np.random.randn(32))
        buffer.update_priorities(indices, td_errors)
        
        # 验证优先级已更新
        for idx, td_error in zip(indices, td_errors):
            expected_priority = (td_error + buffer.epsilon) ** buffer.alpha
            assert abs(buffer.priorities[idx] - expected_priority) < 1e-6
