"""
第 5 章：函数近似
"""

from chapter_05_function_approximation.linear_approximation import (
    LinearValueFunction,
    LinearQFunction,
    tile_coding,
    TileCodedValueFunction
)

from chapter_05_function_approximation.neural_network_q import (
    QNetwork,
    DuelingQNetwork,
    CNNQNetwork,
    NeuralQFunction
)

from chapter_05_function_approximation.dqn import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent
)

__all__ = [
    # Linear Approximation
    'LinearValueFunction',
    'LinearQFunction',
    'tile_coding',
    'TileCodedValueFunction',
    
    # Neural Network Q
    'QNetwork',
    'DuelingQNetwork',
    'CNNQNetwork',
    'NeuralQFunction',
    
    # DQN
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'DQNAgent',
    'DoubleDQNAgent',
    'DuelingDQNAgent',
]
