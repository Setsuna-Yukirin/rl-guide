"""
第 6 章测试：策略梯度
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def cartpole_env():
    """CartPole 环境（离散动作）"""
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    env.reset(seed=42)
    return env


@pytest.fixture
def pendulum_env():
    """Pendulum 环境（连续动作）"""
    import gymnasium as gym
    env = gym.make('Pendulum-v1')
    env.reset(seed=42)
    return env


# ==================== REINFORCE 测试 ====================

@pytest.mark.unit
class TestReinforce:
    """REINFORCE 测试"""
    
    def test_policy_network(self):
        """测试策略网络"""
        from chapter_06_policy_gradient.reinforce import PolicyNetwork
        
        net = PolicyNetwork(state_dim=4, n_actions=2)
        
        import torch
        state = torch.randn(1, 4)
        probs = net(state)
        
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]))
    
    def test_reinforce_agent(self, cartpole_env):
        """测试 REINFORCE 智能体"""
        from chapter_06_policy_gradient.reinforce import REINFORCEAgent
        
        agent = REINFORCEAgent(state_dim=4, n_actions=2)
        
        # 训练几个 episode
        rewards = agent.train(cartpole_env, n_episodes=5, verbose=False)
        
        assert len(rewards) == 5
        assert all(isinstance(r, (float, np.floating)) for r in rewards)
    
    def test_gaussian_policy(self):
        """测试高斯策略（连续动作）"""
        from chapter_06_policy_gradient.reinforce import GaussianPolicyNetwork
        
        net = GaussianPolicyNetwork(state_dim=3, action_dim=1)
        
        import torch
        state = torch.randn(1, 3)
        mean, std = net(state)
        
        assert mean.shape == (1, 1)
        assert std.shape == (1, 1)


# ==================== Actor-Critic 测试 ====================

@pytest.mark.unit
class TestActorCritic:
    """Actor-Critic 测试"""
    
    def test_actor_critic_network(self):
        """测试 Actor-Critic 网络"""
        from chapter_06_policy_gradient.actor_critic import ActorCriticNetwork
        
        net = ActorCriticNetwork(state_dim=4, n_actions=2)
        
        import torch
        state = torch.randn(1, 4)
        probs, value = net(state)
        
        assert probs.shape == (1, 2)
        assert value.shape == (1, 1)
    
    def test_a2c_agent(self, cartpole_env):
        """测试 A2C 智能体"""
        from chapter_06_policy_gradient.actor_critic import A2CAgent
        
        agent = A2CAgent(state_dim=4, n_actions=2, n_steps=5)
        
        rewards = agent.train(cartpole_env, n_episodes=5, verbose=False)
        
        assert len(rewards) == 5


# ==================== DDPG/TD3 测试 ====================

@pytest.mark.unit
class TestDDPG:
    """DDPG 测试"""
    
    def test_ddpg_agent(self, pendulum_env):
        """测试 DDPG 智能体"""
        from chapter_06_policy_gradient.ddpg_td3 import DDPGAgent
        
        agent = DDPGAgent(
            state_dim=3,
            action_dim=1,
            action_bounds=(-2.0, 2.0),
            buffer_size=100,
            batch_size=32
        )
        
        # 训练几个 episode
        rewards = []
        for _ in range(5):
            total_reward, steps = agent.train_episode(pendulum_env)
            rewards.append(total_reward)
        
        assert len(rewards) == 5
    
    def test_td3_agent(self, pendulum_env):
        """测试 TD3 智能体"""
        from chapter_06_policy_gradient.ddpg_td3 import TD3Agent
        
        agent = TD3Agent(
            state_dim=3,
            action_dim=1,
            action_bounds=(-2.0, 2.0),
            buffer_size=100,
            batch_size=32
        )
        
        rewards = []
        for _ in range(5):
            total_reward, steps = agent.train_episode(pendulum_env)
            rewards.append(total_reward)
        
        assert len(rewards) == 5


# ==================== Environment 测试 ====================

@pytest.mark.unit
class TestEnvironments:
    """环境测试"""
    
    def test_pendulum(self):
        """测试 Pendulum"""
        from chapter_06_policy_gradient.games import create_pendulum_env
        
        env = create_pendulum_env()
        state, _ = env.reset(seed=42)
        
        assert state.shape == (3,)
        
        # 连续动作
        action = env.action_space.sample()
        assert isinstance(action, np.ndarray)
    
    def test_lunar_lander(self):
        """测试 Lunar Lander"""
        from chapter_06_policy_gradient.games import create_lunar_lander_env
        
        try:
            env = create_lunar_lander_env(continuous=True)
            state, _ = env.reset(seed=42)
            
            assert state.shape == (8,)
        except Exception as e:
            pytest.skip(f"Lunar Lander not available: {e}")
    
    def test_car_racing(self):
        """测试 CarRacing"""
        from chapter_06_policy_gradient.games import create_car_racing_env
        
        try:
            env = create_car_racing_env()
            state, _ = env.reset(seed=42)
            
            # 图像输入
            assert len(state.shape) == 3
        except Exception as e:
            pytest.skip(f"CarRacing not available: {e}")
