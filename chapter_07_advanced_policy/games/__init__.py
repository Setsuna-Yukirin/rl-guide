"""
第 7 章游戏环境

提供连续控制和 LLM 模拟环境的创建函数
"""

import gymnasium as gym
from typing import Optional, Tuple


def create_pendulum_env() -> gym.Env:
    """
    创建 Pendulum 环境（连续控制）
    
    State: [cos(theta), sin(theta), theta_dot]
    Action: 扭矩 [-2, 2]
    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*action^2)
    """
    return gym.make('Pendulum-v1')


def create_half_cheetah_env() -> gym.Env:
    """
    创建 HalfCheetah 环境（连续控制）
    
    State: 17 维（位置、速度）
    Action: 6 维扭矩 [-1, 1]
    Reward: 向前跑的速度
    """
    return gym.make('HalfCheetah-v4')


def create_hopper_env() -> gym.Env:
    """
    创建 Hopper 环境（连续控制）
    
    State: 11 维
    Action: 3 维扭矩 [-1, 1]
    Reward: 向前跳的速度
    """
    return gym.make('Hopper-v4')


def create_walker_env() -> gym.Env:
    """
    创建 Walker2d 环境（连续控制）
    
    State: 17 维
    Action: 6 维扭矩 [-1, 1]
    Reward: 向前走的速度
    """
    return gym.make('Walker2d-v4')


def create_ant_env() -> gym.Env:
    """
    创建 Ant 环境（连续控制）
    
    State: 27 维
    Action: 8 维扭矩 [-1, 1]
    Reward: 向前爬的速度
    """
    return gym.make('Ant-v4')


def create_humanoid_env() -> gym.Env:
    """
    创建 Humanoid 环境（连续控制）
    
    State: 376 维
    Action: 17 维扭矩 [-1, 1]
    Reward: 向前行走的速度 - 控制成本
    """
    return gym.make('Humanoid-v4')


def create_lunar_lander_env(continuous: bool = True) -> gym.Env:
    """
    创建 Lunar Lander 环境
    
    Args:
        continuous: 是否使用连续动作空间
    """
    if continuous:
        return gym.make('LunarLanderContinuous-v2')
    else:
        return gym.make('LunarLander-v2')


def create_bipedal_walker_env() -> gym.Env:
    """
    创建 BipedalWalker 环境（连续控制）
    
    State: 24 维
    Action: 4 维扭矩 [-1, 1]
    Reward: 向前走的距离 - 成本
    """
    return gym.make('BipedalWalker-v3')


def create_car_racing_env() -> gym.Env:
    """
    创建 CarRacing 环境（图像输入）
    
    State: 96x96 RGB 图像
    Action: 3 维 [方向盘，油门，刹车]
    Reward: 赛道覆盖度 - 时间成本
    """
    return gym.make('CarRacing-v2')


def create_mujoco_env(env_name: str) -> gym.Env:
    """
    创建 MuJoCo 环境
    
    Args:
        env_name: 环境名称（如 'HalfCheetah-v4', 'Hopper-v4' 等）
    """
    return gym.make(env_name)


# 环境注册表
ENV_REGISTRY = {
    'pendulum': create_pendulum_env,
    'half_cheetah': create_half_cheetah_env,
    'hopper': create_hopper_env,
    'walker': create_walker_env,
    'ant': create_ant_env,
    'humanoid': create_humanoid_env,
    'lunar_lander': create_lunar_lander_env,
    'bipedal_walker': create_bipedal_walker_env,
    'car_racing': create_car_racing_env,
}


def get_env(env_name: str, **kwargs) -> gym.Env:
    """
    根据名称获取环境
    
    Args:
        env_name: 环境名称
        **kwargs: 传递给创建函数的参数
    
    Returns:
        gym.Env: 环境实例
    """
    if env_name in ENV_REGISTRY:
        return ENV_REGISTRY[env_name](**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}. "
                        f"Available: {list(ENV_REGISTRY.keys())}")
