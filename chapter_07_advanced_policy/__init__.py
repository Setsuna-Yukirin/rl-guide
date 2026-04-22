"""
第 7 章：高级策略优化

本章实现：
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- Offline RL (BC, CQL, IQL)
"""

from chapter_07_advanced_policy.ppo import PPOAgent, PPOConfig
from chapter_07_advanced_policy.sac import SACAgent, SACConfig
from chapter_07_advanced_policy.dpo import DPOTrainer, DPOConfig, LLMPreferenceDataset
from chapter_07_advanced_policy.grpo import GRPOTrainer, GRPOConfig, MathRewardFunction, CodeRewardFunction
from chapter_07_advanced_policy.offline_rl import (
    OfflineDataset,
    BehaviorCloningAgent,
    CQLAgent,
    IQLAgent,
    OfflineRLConfig,
)

__all__ = [
    # PPO
    'PPOAgent',
    'PPOConfig',
    # SAC
    'SACAgent',
    'SACConfig',
    # DPO
    'DPOTrainer',
    'DPOConfig',
    'LLMPreferenceDataset',
    # GRPO
    'GRPOTrainer',
    'GRPOConfig',
    'MathRewardFunction',
    'CodeRewardFunction',
    # Offline RL
    'OfflineDataset',
    'BehaviorCloningAgent',
    'CQLAgent',
    'IQLAgent',
    'OfflineRLConfig',
]
