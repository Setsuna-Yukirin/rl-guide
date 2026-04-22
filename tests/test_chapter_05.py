"""
第 5 章测试：函数近似
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def cartpole_env():
    """CartPole 环境"""
    from chapter_05_function_approximation.games import create_cartpole_env
    env = create_cartpole_env()
    env.reset(seed=42)
    return env


# ==================== Linear Approximation 测试 ====================

@pytest.mark.unit
class TestLinearApproximation:
    """线性逼近测试"""
    
    def test_linear_value_function(self):
        """测试线性价值函数"""
        from chapter_05_function_approximation.linear_approximation import LinearValueFunction
        
        v_func = LinearValueFunction(n_features=10, learning_rate=0.1)
        
        features = np.random.randn(10)
        value = v_func.predict(features)
        
        assert isinstance(value, float)
        
        # 更新
        v_func.update(features, target=5.0)
        new_value = v_func.predict(features)
        
        # 价值应该接近目标
        assert abs(new_value - 5.0) < abs(value - 5.0)
    
    def test_linear_q_function(self):
        """测试线性 Q 函数"""
        from chapter_05_function_approximation.linear_approximation import LinearQFunction
        
        q_func = LinearQFunction(n_features=10, n_actions=4)
        
        features = np.random.randn(10)
        q_values = q_func.predict(features)
        
        assert q_values.shape == (4,)
        
        # 获取最优动作
        best_action = q_func.get_best_action(features)
        assert 0 <= best_action < 4
    
    def test_tile_coding(self):
        """测试平铺编码"""
        from chapter_05_function_approximation.linear_approximation import tile_coding
        
        state = np.array([1.5, 2.3])
        features = tile_coding(state, n_tilings=8, tile_size=1.0)
        
        assert len(features) == 8
        assert np.all(features == 1.0)  # 二值特征


# ==================== Neural Network Q 测试 ====================

@pytest.mark.unit
class TestNeuralNetworkQ:
    """神经网络 Q 函数测试"""
    
    def test_q_network(self):
        """测试 Q 网络"""
        from chapter_05_function_approximation.neural_network_q import QNetwork
        
        net = QNetwork(state_dim=4, n_actions=2, hidden_dims=(64, 64))
        
        state = np.random.randn(4)
        q_values = net.get_action(state)
        
        assert 0 <= q_values < 2
    
    def test_dueling_network(self):
        """测试 Dueling 网络"""
        from chapter_05_function_approximation.neural_network_q import DuelingQNetwork
        import torch
        
        net = DuelingQNetwork(state_dim=4, n_actions=2)
        
        state = torch.randn(1, 4)
        q_values = net(state)
        
        assert q_values.shape == (1, 2)
    
    def test_neural_q_function(self):
        """测试神经网络 Q 函数"""
        from chapter_05_function_approximation.neural_network_q import NeuralQFunction
        
        q_func = NeuralQFunction(state_dim=4, n_actions=2)
        
        # 训练一步
        states = np.random.randn(32, 4)
        actions = np.random.randint(0, 2, 32)
        rewards = np.random.randn(32)
        next_states = np.random.randn(32, 4)
        dones = np.random.randint(0, 2, 32).astype(float)
        
        loss = q_func.train_step(states, actions, rewards, next_states, dones)
        
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0


# ==================== DQN 测试 ====================

@pytest.mark.unit
class TestDQN:
    """DQN 测试"""
    
    def test_replay_buffer(self):
        """测试回放缓冲区"""
        from chapter_05_function_approximation.dqn import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(50):
            buffer.push(
                np.random.randn(4),
                np.random.randint(0, 2),
                np.random.randn(),
                np.random.randn(4),
                np.random.randint(0, 2)
            )
        
        assert len(buffer) == 50
        
        states, actions, rewards, next_states, dones = buffer.sample(10)
        
        assert states.shape == (10, 4)
        assert actions.shape == (10,)
    
    def test_dqn_agent(self, cartpole_env):
        """测试 DQN 智能体"""
        from chapter_05_function_approximation.dqn import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            n_actions=2,
            buffer_size=100,
            batch_size=32
        )
        
        # 训练几个 episode
        rewards = agent.train(cartpole_env, n_episodes=5, verbose=False)
        
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)
    
    def test_double_dqn(self, cartpole_env):
        """测试 Double DQN"""
        from chapter_05_function_approximation.dqn import DoubleDQNAgent
        
        agent = DoubleDQNAgent(
            state_dim=4,
            n_actions=2,
            buffer_size=100,
            batch_size=32
        )
        
        rewards = agent.train(cartpole_env, n_episodes=5, verbose=False)
        
        assert len(rewards) == 5


# ==================== Environment 测试 ====================

@pytest.mark.unit
class TestEnvironments:
    """环境测试"""
    
    def test_cartpole(self):
        """测试 CartPole"""
        from chapter_05_function_approximation.games import create_cartpole_env
        
        env = create_cartpole_env()
        state, _ = env.reset(seed=42)
        
        assert state.shape == (4,)
        
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        assert next_state.shape == (4,)
        assert isinstance(reward, float)
    
    def test_lunar_lander(self):
        """测试 Lunar Lander"""
        from chapter_05_function_approximation.games import create_lunar_lander_env
        
        try:
            env = create_lunar_lander_env()
            state, _ = env.reset(seed=42)
            
            assert state.shape == (8,)
        except Exception as e:
            # Box2D 可能需要额外依赖
            pytest.skip(f"Lunar Lander not available: {e}")
    
    def test_breakout(self):
        """测试 Breakout"""
        from chapter_05_function_approximation.games import create_breakout_env
        
        try:
            env = create_breakout_env()
            state, _ = env.reset(seed=42)
            
            # 帧堆叠后应该是 4 帧
            assert len(state.shape) == 3
        except Exception as e:
            # Atari 环境可能需要额外依赖
            pytest.skip(f"Atari environment not available: {e}")
