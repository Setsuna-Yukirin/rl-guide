"""
rl-guide 工具模块

提供强化学习的工具函数和类：
- 核心抽象：MDP, Policy, ValueFunction, ReplayBuffer
- 可视化：学习曲线、热力图、策略图
- 训练循环：TrainingLoop, ParallelTrainingLoop
"""

from utils.core.mdp import MDP, TabularMDP
from utils.core.policy import (
    Policy,
    RandomPolicy,
    EpsilonGreedyPolicy,
    GreedyPolicy,
)
from utils.core.value_function import (
    ValueFunction,
    TabularVFunction,
    TabularQFunction,
    NeuralVFunction,
)
from utils.core.replay_buffer import ReplayBuffer, PriorityReplayBuffer

from utils.visualization import (
    plot_learning_curve,
    plot_value_heatmap,
    plot_policy_arrows,
    plot_training_stats,
    create_training_animation,
)

from utils.training_loop import TrainingLoop, ParallelTrainingLoop

__all__ = [
    # Core
    "MDP",
    "TabularMDP",
    "Policy",
    "RandomPolicy",
    "EpsilonGreedyPolicy",
    "GreedyPolicy",
    "ValueFunction",
    "TabularVFunction",
    "TabularQFunction",
    "NeuralVFunction",
    "ReplayBuffer",
    "PriorityReplayBuffer",
    
    # Visualization
    "plot_learning_curve",
    "plot_value_heatmap",
    "plot_policy_arrows",
    "plot_training_stats",
    "create_training_animation",
    
    # Training
    "TrainingLoop",
    "ParallelTrainingLoop",
]
