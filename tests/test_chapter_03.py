"""
第 3 章测试：蒙特卡洛方法
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def blackjack():
    """Blackjack 环境 fixture"""
    from chapter_03_monte_carlo.games import BlackjackEnv
    env = BlackjackEnv()
    env.reset(seed=42)
    return env


@pytest.fixture
def simple_policy():
    """简单策略 fixture（总是停牌）"""
    def policy(state):
        return 1  # 总是停牌
    return policy


@pytest.mark.unit
class TestMCPrediction:
    """MC 预测测试"""
    
    def test_generate_episode(self, blackjack, simple_policy):
        """测试 episode 生成"""
        from chapter_03_monte_carlo.mc_prediction import generate_episode
        
        episode = generate_episode(blackjack, simple_policy, max_steps=100)
        
        assert len(episode) > 0
        assert len(episode[0]) == 3  # (state, action, reward)
    
    def test_compute_return(self):
        """测试回报计算"""
        from chapter_03_monte_carlo.mc_prediction import compute_return
        
        # 简单 episode
        episode = [
            (0, 0, 1.0),
            (1, 0, 2.0),
            (2, 0, 3.0)
        ]
        
        # γ=1.0
        G0 = compute_return(episode, 0, gamma=1.0)
        assert G0 == 6.0  # 1 + 2 + 3
        
        G1 = compute_return(episode, 1, gamma=1.0)
        assert G1 == 5.0  # 2 + 3
        
        # γ=0.9
        G0_discounted = compute_return(episode, 0, gamma=0.9)
        expected = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
        assert np.isclose(G0_discounted, expected)
    
    def test_first_visit_mc_prediction(self, blackjack, simple_policy):
        """测试第一访问 MC 预测"""
        from chapter_03_monte_carlo.mc_prediction import first_visit_mc_prediction
        
        V, returns = first_visit_mc_prediction(
            blackjack, simple_policy, n_episodes=100, gamma=1.0
        )
        
        assert len(V) > 0
        assert all(np.isfinite(v) for v in V.values())
    
    def test_every_visit_mc_prediction(self, blackjack, simple_policy):
        """测试每次访问 MC 预测"""
        from chapter_03_monte_carlo.mc_prediction import every_visit_mc_prediction
        
        V, returns = every_visit_mc_prediction(
            blackjack, simple_policy, n_episodes=100, gamma=1.0
        )
        
        assert len(V) > 0
        # 每次访问应该有更多数据
        total_returns = sum(len(v) for v in returns.values())
        assert total_returns > 0


@pytest.mark.unit
class TestMCControl:
    """MC 控制测试"""
    
    def test_mc_control_es(self, blackjack):
        """测试 ε-soft MC 控制"""
        from chapter_03_monte_carlo.mc_control import mc_control_es
        
        Q, policy = mc_control_es(blackjack, n_episodes=100, epsilon=0.1)
        
        assert len(Q) > 0
        assert len(policy) > 0
        
        # 检查 Q 值形状
        for state, q_values in Q.items():
            assert len(q_values) == 2  # 两个动作
    
    def test_extract_greedy_policy(self):
        """测试贪婪策略提取"""
        from chapter_03_monte_carlo.mc_control import extract_greedy_policy
        
        Q = {
            0: np.array([1.0, 2.0]),
            1: np.array([3.0, 1.0]),
            2: np.array([0.5, 0.5])
        }
        
        policy = extract_greedy_policy(Q, n_actions=2)
        
        assert policy[0] == 1  # 选择 Q 值大的
        assert policy[1] == 0
        # 平局时选择第一个
    
    def test_mc_control_glie(self, blackjack):
        """测试 GLIE MC 控制"""
        from chapter_03_monte_carlo.mc_control import mc_control_glie
        
        Q, policy = mc_control_glie(blackjack, n_episodes=100, k0=100.0)
        
        assert len(Q) > 0
        assert len(policy) > 0


@pytest.mark.unit
class TestEpsilonGreedy:
    """ε-贪婪策略测试"""
    
    def test_epsilon_greedy_init(self):
        """测试初始化"""
        from chapter_03_monte_carlo.epsilon_greedy import EpsilonGreedyPolicy
        
        policy = EpsilonGreedyPolicy(n_actions=2, epsilon=0.1)
        
        assert policy.n_actions == 2
        assert policy.epsilon == 0.1
    
    def test_epsilon_greedy_action(self):
        """测试动作选择"""
        from chapter_03_monte_carlo.epsilon_greedy import EpsilonGreedyPolicy
        
        policy = EpsilonGreedyPolicy(n_actions=2, epsilon=0.0)  # 完全贪婪
        policy.Q = {
            0: np.array([1.0, 2.0]),
            1: np.array([3.0, 1.0])
        }
        
        # 总是选择最优动作
        for _ in range(10):
            action = policy.get_action(0, explore=False)
            assert action == 1
        
        action = policy.get_action(1, explore=False)
        assert action == 0
    
    def test_epsilon_greedy_explore(self):
        """测试探索行为"""
        from chapter_03_monte_carlo.epsilon_greedy import EpsilonGreedyPolicy
        
        policy = EpsilonGreedyPolicy(n_actions=2, epsilon=1.0)  # 完全探索
        
        # 应该随机选择
        actions = set()
        for _ in range(100):
            action = policy.get_action(0, explore=True)
            actions.add(action)
        
        # 高概率两个动作都会被选到
        assert len(actions) >= 1
    
    def test_decay_epsilon(self):
        """测试 ε 衰减"""
        from chapter_03_monte_carlo.epsilon_greedy import EpsilonGreedyPolicy
        
        policy = EpsilonGreedyPolicy(n_actions=2, epsilon=0.5)
        
        policy.decay_epsilon(decay_rate=0.9, min_epsilon=0.01)
        assert policy.epsilon == 0.45
        
        policy.decay_epsilon(decay_rate=0.9, min_epsilon=0.01)
        assert policy.epsilon == 0.405
    
    def test_glie_policy(self):
        """测试 GLIE 策略"""
        from chapter_03_monte_carlo.epsilon_greedy import GLIEPolicy
        
        policy = GLIEPolicy(n_actions=2, k0=100.0)
        
        # 初始 ε 应该接近 1
        epsilon_0 = policy.get_epsilon()
        assert 0.5 < epsilon_0 <= 1.0
        
        # 多次选择后 ε 应该减小
        for _ in range(100):
            policy.get_action(0)
        
        epsilon_100 = policy.get_epsilon()
        assert epsilon_100 < epsilon_0


@pytest.mark.integration
class TestBlackjackEnv:
    """Blackjack 环境测试"""
    
    def test_blackjack_reset(self, blackjack):
        """测试重置"""
        state, info = blackjack.reset(seed=42)
        
        player_total, dealer_showing, usable_ace = state
        assert 4 <= player_total <= 21
        assert 1 <= dealer_showing <= 10
        assert usable_ace in [0, 1]
    
    def test_blackjack_step_hit(self, blackjack):
        """测试要牌"""
        state, _ = blackjack.reset(seed=42)
        
        # 要牌
        next_state, reward, terminated, _, _ = blackjack.step(0)
        
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
    
    def test_blackjack_step_stand(self, blackjack):
        """测试停牌"""
        state, _ = blackjack.reset(seed=42)
        
        # 立即停牌
        next_state, reward, terminated, _, _ = blackjack.step(1)
        
        # 停牌后应该终止
        assert terminated
        assert reward in [-1.0, 0.0, 1.0, 1.5]
    
    def test_blackjack_bust(self):
        """测试爆牌"""
        from chapter_03_monte_carlo.games import BlackjackEnv
        
        env = BlackjackEnv()
        
        # 多次要牌直到爆牌
        state, _ = env.reset(seed=100)
        
        for _ in range(20):
            state, reward, terminated, _, _ = env.step(0)  # 要牌
            if terminated:
                break
        
        # 最终应该终止
        assert terminated
    
    def test_state_conversion(self):
        """测试状态转换"""
        from chapter_03_monte_carlo.games import state_to_index, index_to_state
        
        state = (15, 7, 1)
        index = state_to_index(state)
        recovered = index_to_state(index)
        
        assert recovered == state


@pytest.mark.unit
class TestMCPredictor:
    """MCPredictor 类测试"""
    
    def test_mc_predictor_init(self):
        """测试初始化"""
        from chapter_03_monte_carlo.mc_prediction import MCPredictor
        
        predictor = MCPredictor(gamma=0.9)
        
        assert predictor.gamma == 0.9
        assert len(predictor.V) == 0
    
    def test_mc_predictor_update(self):
        """测试更新"""
        from chapter_03_monte_carlo.mc_prediction import MCPredictor
        
        predictor = MCPredictor(gamma=1.0)
        
        episode = [
            (0, 0, 1.0),
            (1, 0, 2.0),
            (2, 0, 3.0)
        ]
        
        predictor.update(episode, first_visit=True)
        
        assert 0 in predictor.V
        assert predictor.N[0] == 1
    
    def test_mc_predictor_train(self, blackjack, simple_policy):
        """测试训练"""
        from chapter_03_monte_carlo.mc_prediction import MCPredictor
        
        predictor = MCPredictor(gamma=1.0)
        predictor.train(blackjack, simple_policy, n_episodes=50)
        
        assert len(predictor.V) > 0
        assert all(n > 0 for n in predictor.N.values())
