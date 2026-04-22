"""
工具模块单元测试

测试 utils/ 下的工具函数：
- visualization
- training_loop
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from utils.visualization import (
    plot_learning_curve,
    plot_value_heatmap,
    plot_policy_arrows,
    plot_training_stats,
)
from utils.training_loop import TrainingLoop
from utils.core import EpsilonGreedyPolicy, TabularQFunction, ReplayBuffer


# ============== 可视化测试 ==============

@pytest.mark.unit
class TestVisualization:
    """可视化函数测试"""
    
    def test_plot_learning_curve(self, tmp_path):
        """测试学习曲线绘制"""
        rewards = [np.random.randn() for _ in range(50)]
        save_path = tmp_path / "test_curve.png"
        
        fig, ax = plot_learning_curve(
            rewards,
            window_size=10,
            save_path=str(save_path),
            show=False,
        )
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_value_heatmap(self, tmp_path):
        """测试热力图绘制"""
        values = np.random.randn(10, 10)
        save_path = tmp_path / "test_heatmap.png"
        
        fig, ax, im = plot_value_heatmap(
            values,
            annotate=True,
            save_path=str(save_path),
            show=False,
        )
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_policy_arrows(self, tmp_path):
        """测试策略箭头图"""
        policy = np.random.randn(10, 10, 2) * 0.5
        value = np.random.randn(10, 10)
        save_path = tmp_path / "test_policy.png"
        
        fig, ax = plot_policy_arrows(
            policy,
            value,
            save_path=str(save_path),
            show=False,
        )
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_training_stats(self, tmp_path):
        """测试训练统计图"""
        stats = {
            'rewards': [np.random.randn() for _ in range(100)],
            'losses': [np.random.rand() for _ in range(100)],
        }
        save_path = tmp_path / "test_stats.png"
        
        fig, axes = plot_training_stats(
            stats,
            save_path=str(save_path),
            show=False,
        )
        
        assert save_path.exists()
        plt.close(fig)


# ============== 训练循环测试 ==============

@pytest.mark.unit
class TestTrainingLoop:
    """TrainingLoop 类测试"""
    
    def test_training_loop_init(self, small_env):
        """测试训练循环初始化"""
        Q = TabularQFunction(state_dim=4, action_dim=2)
        policy = EpsilonGreedyPolicy(state_dim=4, action_dim=2)
        policy.set_q_values(Q.q_values)
        
        loop = TrainingLoop(
            agent=policy,
            env=small_env,
            num_episodes=10,
        )
        
        assert loop.num_episodes == 10
        assert loop.verbose == True
    
    def test_training_loop_run(self, small_env):
        """测试训练循环运行"""
        Q = TabularQFunction(state_dim=4, action_dim=2)
        policy = EpsilonGreedyPolicy(state_dim=4, action_dim=2, epsilon=0.1)
        policy.set_q_values(Q.q_values)
        
        loop = TrainingLoop(
            agent=policy,
            env=small_env,
            num_episodes=5,
            verbose=False,
        )
        
        stats = loop.run()
        
        assert 'rewards' in stats
        assert 'steps' in stats
        assert len(stats['rewards']) == 5
    
    def test_training_loop_with_buffer(self, small_env):
        """测试带缓冲区的训练循环"""
        Q = TabularQFunction(state_dim=4, action_dim=2)
        policy = EpsilonGreedyPolicy(state_dim=4, action_dim=2)
        policy.set_q_values(Q.q_values)
        
        buffer = ReplayBuffer(capacity=1000)
        
        loop = TrainingLoop(
            agent=policy,
            env=small_env,
            num_episodes=5,
            buffer=buffer,
            batch_size=32,
            verbose=False,
        )
        
        stats = loop.run()
        
        assert len(stats['rewards']) == 5
        assert len(buffer) > 0


# ============== 回归测试 ==============

@pytest.mark.integration
class TestRegression:
    """回归测试 - 确保现有功能正常工作"""
    
    def test_core_imports(self):
        """测试核心模块导入"""
        from utils.core import (
            MDP, TabularMDP,
            Policy, RandomPolicy, EpsilonGreedyPolicy, GreedyPolicy,
            TabularVFunction, TabularQFunction,
            ReplayBuffer, PriorityReplayBuffer,
        )
        # 如果能导入则测试通过
    
    def test_utils_imports(self):
        """测试工具模块导入"""
        from utils import (
            plot_learning_curve,
            plot_value_heatmap,
            TrainingLoop,
        )
        # 如果能导入则测试通过
    
    def test_policy_consistency(self):
        """测试策略一致性"""
        from utils.core import RandomPolicy, EpsilonGreedyPolicy, TabularQFunction
        
        # 随机策略应该返回有效动作
        policy = RandomPolicy(state_dim=4, action_dim=2, seed=42)
        actions = [policy.get_action(np.zeros(4)) for _ in range(100)]
        assert all(a in [0, 1] for a in actions)
        
        # ε-greedy 策略应该返回有效动作
        q_func = TabularQFunction(state_dim=4, action_dim=2)
        policy = EpsilonGreedyPolicy(state_dim=4, action_dim=2, epsilon=0.1)
        policy.set_q_values(q_func.q_values)
        actions = [policy.get_action(np.array([0])) for _ in range(100)]
        assert all(a in [0, 1] for a in actions)
    
    def test_buffer_fifo(self):
        """测试缓冲区 FIFO 行为"""
        buffer = ReplayBuffer(capacity=10)
        
        # 添加 15 条经验（超过容量）
        for i in range(15):
            buffer.add(np.array([i]), 0, 1.0, np.array([i]), False)
        
        # 缓冲区应该只保留最后 10 条
        assert len(buffer) == 10
        
        # 采样应该只包含最近的经验
        batch = buffer.sample(batch_size=5)
        states = batch[0]
        assert all(s[0] >= 5 for s in states)  # 最早的状态应该是 5
