"""
第 4 章测试：时序差分学习（重点章节）
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def cliff_walking():
    """Cliff Walking 环境 fixture"""
    from chapter_04_temporal_difference.games import CliffWalkingEnv
    env = CliffWalkingEnv()
    env.reset(seed=42)
    return env


@pytest.fixture
def windy_gridworld():
    """Windy Gridworld 环境 fixture"""
    from chapter_04_temporal_difference.games import WindyGridworldEnv
    env = WindyGridworldEnv()
    env.reset(seed=42)
    return env


@pytest.fixture
def simple_policy():
    """简单随机策略"""
    def policy(state):
        return np.random.randint(4)
    return policy


@pytest.mark.unit
class TestTDPrediction:
    """TD 预测测试"""
    
    def test_td_0_prediction(self, cliff_walking, simple_policy):
        """测试 TD(0) 预测"""
        from chapter_04_temporal_difference.td_prediction import td_0_prediction
        
        V = td_0_prediction(
            cliff_walking, simple_policy, n_episodes=100, alpha=0.1, gamma=0.99
        )
        
        assert len(V) > 0
        assert all(np.isfinite(v) for v in V.values())
    
    def test_td_predictor_update(self):
        """测试 TDPredictor 更新"""
        from chapter_04_temporal_difference.td_prediction import TDPredictor
        
        predictor = TDPredictor(alpha=0.1, gamma=0.99)
        
        # 初始价值为 0
        assert predictor.V[0] == 0.0
        
        # 更新
        predictor.update(state=0, reward=-1.0, next_state=1, terminated=False)
        
        # TD 误差 = -1 + 0.99*0 - 0 = -1
        # V[0] = 0 + 0.1*(-1) = -0.1
        assert np.isclose(predictor.V[0], -0.1)
    
    def test_td_predictor_train(self, cliff_walking, simple_policy):
        """测试 TDPredictor 训练"""
        from chapter_04_temporal_difference.td_prediction import TDPredictor
        
        predictor = TDPredictor(alpha=0.1, gamma=0.99)
        predictor.train(cliff_walking, simple_policy, n_episodes=50)
        
        assert len(predictor.V) > 0
        assert all(n > 0 for n in predictor.N.values())
    
    def test_td_lambda_prediction(self, cliff_walking, simple_policy):
        """测试 TD(λ) 预测"""
        from chapter_04_temporal_difference.td_prediction import td_lambda_prediction
        
        V = td_lambda_prediction(
            cliff_walking, simple_policy, n_episodes=100,
            alpha=0.1, gamma=0.99, lambda_=0.5
        )
        
        assert len(V) > 0


@pytest.mark.unit
class TestQLearning:
    """Q-Learning 测试"""
    
    def test_q_learning_agent_init(self, cliff_walking):
        """测试 Q-Learning 智能体初始化"""
        from chapter_04_temporal_difference.q_learning import QLearningAgent
        
        agent = QLearningAgent(
            n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1
        )
        
        assert agent.n_actions == 4
        assert agent.alpha == 0.1
        assert agent.epsilon == 0.1
    
    def test_q_learning_action(self):
        """测试 Q-Learning 动作选择"""
        from chapter_04_temporal_difference.q_learning import QLearningAgent
        
        agent = QLearningAgent(n_actions=2, epsilon=0.0)  # 完全贪婪
        agent.Q[0] = np.array([1.0, 2.0])
        
        # 应该总是选择动作 1
        for _ in range(10):
            action = agent.get_action(0, explore=False)
            assert action == 1
    
    def test_q_learning_update(self):
        """测试 Q-Learning 更新"""
        from chapter_04_temporal_difference.q_learning import QLearningAgent
        
        agent = QLearningAgent(n_actions=2, alpha=0.1, gamma=0.99)
        
        # 初始 Q 值为 0
        assert agent.Q[0][0] == 0.0
        
        # 更新：Q(0,0) <- 0 + 0.1 * (-1 + 0.99*0 - 0) = -0.1
        agent.update(state=0, action=0, reward=-1.0, next_state=1, terminated=False)
        
        assert np.isclose(agent.Q[0][0], -0.1)
    
    def test_q_learning_train_episode(self, cliff_walking):
        """测试 Q-Learning 单集训练"""
        from chapter_04_temporal_difference.q_learning import QLearningAgent
        
        agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        reward = agent.train_episode(cliff_walking)
        
        assert isinstance(reward, float)
        assert reward < 0  # Cliff Walking 中奖励总是负的
    
    def test_q_learning_train(self, cliff_walking):
        """测试 Q-Learning 训练"""
        from chapter_04_temporal_difference.q_learning import QLearningAgent
        
        agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        rewards = agent.train(cliff_walking, n_episodes=50)
        
        assert len(rewards) == 50
        assert all(isinstance(r, float) for r in rewards)
    
    def test_q_learning_function(self, cliff_walking):
        """测试 q_learning 函数"""
        from chapter_04_temporal_difference.q_learning import q_learning
        
        Q, policy, rewards = q_learning(
            cliff_walking, n_episodes=50, alpha=0.1, gamma=0.99, epsilon=0.1
        )
        
        assert len(Q) > 0
        assert len(policy) > 0
        assert len(rewards) == 50
    
    def test_double_q_learning(self, cliff_walking):
        """测试 Double Q-Learning"""
        from chapter_04_temporal_difference.q_learning import DoubleQLearningAgent
        
        agent = DoubleQLearningAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        rewards = agent.train(cliff_walking, n_episodes=50)
        
        assert len(rewards) == 50
        assert len(agent.Q1) > 0 or len(agent.Q2) > 0


@pytest.mark.unit
class TestSARSA:
    """SARSA 测试"""
    
    def test_sarsa_agent_init(self, cliff_walking):
        """测试 SARSA 智能体初始化"""
        from chapter_04_temporal_difference.sarsa import SarsaAgent
        
        agent = SarsaAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        assert agent.n_actions == 4
        assert agent.alpha == 0.1
    
    def test_sarsa_update(self):
        """测试 SARSA 更新"""
        from chapter_04_temporal_difference.sarsa import SarsaAgent
        
        agent = SarsaAgent(n_actions=2, alpha=0.1, gamma=0.99)
        
        # 更新：需要 next_action
        agent.update(
            state=0, action=0, reward=-1.0,
            next_state=1, next_action=1, terminated=False
        )
        
        # TD 误差 = -1 + 0.99*0 - 0 = -1
        # Q[0][0] = 0 + 0.1*(-1) = -0.1
        assert np.isclose(agent.Q[0][0], -0.1)
    
    def test_sarsa_train(self, cliff_walking):
        """测试 SARSA 训练"""
        from chapter_04_temporal_difference.sarsa import SarsaAgent
        
        agent = SarsaAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        rewards = agent.train(cliff_walking, n_episodes=50)
        
        assert len(rewards) == 50
    
    def test_sarsa_function(self, cliff_walking):
        """测试 sarsa 函数"""
        from chapter_04_temporal_difference.sarsa import sarsa
        
        Q, policy, rewards = sarsa(
            cliff_walking, n_episodes=50, alpha=0.1, gamma=0.99, epsilon=0.1
        )
        
        assert len(Q) > 0
        assert len(rewards) == 50
    
    def test_n_step_sarsa(self, windy_gridworld):
        """测试 n 步 SARSA"""
        from chapter_04_temporal_difference.sarsa import NStepSarsaAgent
        
        agent = NStepSarsaAgent(
            n_actions=4, n_steps=5, alpha=0.1, gamma=0.99, epsilon=0.1
        )
        
        rewards = agent.train(windy_gridworld, n_episodes=20)
        
        assert len(rewards) == 20


@pytest.mark.unit
class TestExpectedSARSA:
    """Expected SARSA 测试"""
    
    def test_expected_sarsa_agent_init(self, cliff_walking):
        """测试 Expected SARSA 智能体初始化"""
        from chapter_04_temporal_difference.expected_sarsa import ExpectedSarsaAgent
        
        agent = ExpectedSarsaAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        assert agent.n_actions == 4
        assert agent.epsilon == 0.1
    
    def test_expected_q_computation(self):
        """测试期望 Q 值计算"""
        from chapter_04_temporal_difference.expected_sarsa import ExpectedSarsaAgent
        
        agent = ExpectedSarsaAgent(n_actions=2, epsilon=0.0)  # 完全贪婪
        agent.Q[0] = np.array([1.0, 2.0])
        
        # 期望 Q 值应该等于最大值（因为 ε=0）
        expected_q = agent.compute_expected_q(0)
        assert np.isclose(expected_q, 2.0)
    
    def test_expected_sarsa_with_exploration(self):
        """测试带探索的期望 Q 值"""
        from chapter_04_temporal_difference.expected_sarsa import ExpectedSarsaAgent
        
        agent = ExpectedSarsaAgent(n_actions=2, epsilon=0.5)
        agent.Q[0] = np.array([1.0, 3.0])
        
        # E[Q] = (1-0.5)*3 + 0.5*(1+3)/2 = 1.5 + 1.0 = 2.5
        expected_q = agent.compute_expected_q(0)
        assert np.isclose(expected_q, 2.5)
    
    def test_expected_sarsa_train(self, cliff_walking):
        """测试 Expected SARSA 训练"""
        from chapter_04_temporal_difference.expected_sarsa import ExpectedSarsaAgent
        
        agent = ExpectedSarsaAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        rewards = agent.train(cliff_walking, n_episodes=50)
        
        assert len(rewards) == 50
    
    def test_expected_sarsa_function(self, cliff_walking):
        """测试 expected_sarsa 函数"""
        from chapter_04_temporal_difference.expected_sarsa import expected_sarsa
        
        Q, policy, rewards = expected_sarsa(
            cliff_walking, n_episodes=50, alpha=0.1, gamma=0.99, epsilon=0.1
        )
        
        assert len(Q) > 0
        assert len(rewards) == 50


@pytest.mark.integration
class TestCliffWalking:
    """Cliff Walking 环境测试"""
    
    def test_cliff_walking_reset(self, cliff_walking):
        """测试重置"""
        state, _ = cliff_walking.reset(seed=42)
        
        # 起点状态
        assert state == 36  # (3, 0) -> 3*12 + 0
    
    def test_cliff_walking_step(self, cliff_walking):
        """测试步"""
        state, _ = cliff_walking.reset(seed=42)
        
        # 向右移动（不朝悬崖方向）
        # 起点是 (3, 0)，向上移动
        next_state, reward, terminated, _, _ = cliff_walking.step(0)
        
        assert reward == -1.0
        assert not terminated
    
    def test_cliff_walking_fall(self):
        """测试掉下悬崖"""
        from chapter_04_temporal_difference.games import CliffWalkingEnv
        
        env = CliffWalkingEnv()
        env.reset(seed=42)
        
        # 移动到悬崖边缘 (3, 1)
        env.agent_pos = (3, 0)
        
        # 向右移动会掉下悬崖
        next_state, reward, terminated, _, _ = env.step(3)
        
        assert reward == -100.0
        assert terminated
        assert env.agent_pos == (3, 0)  # 回到起点
    
    def test_cliff_walking_goal(self):
        """测试到达终点"""
        from chapter_04_temporal_difference.games import CliffWalkingEnv
        
        env = CliffWalkingEnv()
        env.reset(seed=42)
        env.agent_pos = (3, 10)  # 终点左边
        
        # 向右移动到终点
        next_state, reward, terminated, _, _ = env.step(3)
        
        assert terminated
        assert reward == -1.0


@pytest.mark.integration
class TestWindyGridworld:
    """Windy Gridworld 环境测试"""
    
    def test_windy_gridworld_reset(self, windy_gridworld):
        """测试重置"""
        state, _ = windy_gridworld.reset(seed=42)
        
        # 起点状态
        assert state == 30  # (3, 0) -> 3*10 + 0
    
    def test_windy_gridworld_wind_effect(self):
        """测试风力效果"""
        from chapter_04_temporal_difference.games import WindyGridworldEnv
        
        env = WindyGridworldEnv()
        env.reset(seed=42)
        
        # 在有风的列（列 3-5）
        env.agent_pos = (3, 4)  # 列 4 有风
        
        # 向右移动，但会被风吹向上
        next_state, reward, terminated, _, _ = env.step(3)
        
        # 应该被向上吹
        new_pos = env.agent_pos
        assert new_pos[0] < 3  # 行号变小（向上）
    
    def test_windy_gridworld_king_moves(self):
        """测试 8 方向移动"""
        from chapter_04_temporal_difference.games import WindyGridworldEnv
        
        env = WindyGridworldEnv(king_moves=True)
        env.reset(seed=42)
        
        assert env.n_actions == 8
        
        # 对角移动
        next_state, reward, terminated, _, _ = env.step(1)  # 右上
        
        assert env.agent_pos[0] < 3  # 向上
        assert env.agent_pos[1] > 0  # 向右


@pytest.mark.integration
class TestAlgorithmComparison:
    """算法比较测试"""
    
    def test_sarsa_vs_qlearning(self, cliff_walking):
        """比较 SARSA 和 Q-Learning"""
        from chapter_04_temporal_difference.sarsa import SarsaAgent
        from chapter_04_temporal_difference.q_learning import QLearningAgent
        
        # SARSA
        sarsa_agent = SarsaAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        sarsa_rewards = sarsa_agent.train(cliff_walking, n_episodes=100)
        
        # Q-Learning
        q_agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1)
        q_rewards = q_agent.train(cliff_walking, n_episodes=100)
        
        # 两种算法都应该学习（奖励应该改善）
        assert np.mean(sarsa_rewards[-20:]) > np.mean(sarsa_rewards[:20])
        assert np.mean(q_rewards[-20:]) > np.mean(q_rewards[:20])
    
    def test_expected_sarsa_performance(self, cliff_walking):
        """测试 Expected SARSA 性能"""
        from chapter_04_temporal_difference.expected_sarsa import ExpectedSarsaAgent
        from chapter_04_temporal_difference.sarsa import SarsaAgent
        from chapter_04_temporal_difference.q_learning import QLearningAgent
        
        # 三种算法
        agents = {
            'SARSA': SarsaAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1),
            'Q-Learning': QLearningAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1),
            'Expected SARSA': ExpectedSarsaAgent(n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1),
        }
        
        results = {}
        for name, agent in agents.items():
            rewards = agent.train(cliff_walking, n_episodes=100)
            results[name] = np.mean(rewards[-20:])  # 最后 20 集的平均
        
        # 所有算法都应该有合理的性能
        for name, avg_reward in results.items():
            assert avg_reward > -200  # 不应该太差
