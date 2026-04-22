"""
第 6 章游戏模块

连续控制环境
"""


def create_car_racing_env(**kwargs):
    """
    创建 CarRacing 环境
    
    Args:
        **kwargs: 传递给 gymnasium.make 的参数
    
    Returns:
        CarRacing 环境（连续控制版本）
    """
    import gymnasium as gym
    return gym.make('CarRacing-v3', continuous=True, **kwargs)


def create_lunar_lander_env(continuous: bool = True, **kwargs):
    """
    创建 Lunar Lander 环境
    
    Args:
        continuous: 是否连续动作空间
        **kwargs: 其他参数
    
    Returns:
        Lunar Lander 环境
    """
    import gymnasium as gym
    env_name = 'LunarLanderContinuous-v3' if continuous else 'LunarLander-v3'
    return gym.make(env_name, **kwargs)


def create_pendulum_env(**kwargs):
    """
    创建 Pendulum 环境（机械臂简化版）
    
    Args:
        **kwargs: 传递给 gymnasium.make 的参数
    
    Returns:
        Pendulum 环境
    """
    import gymnasium as gym
    return gym.make('Pendulum-v1', **kwargs)


def create_bipedal_walker_env(**kwargs):
    """
    创建 BipedalWalker 环境（双足机器人）
    
    Args:
        **kwargs: 传递给 gymnasium.make 的参数
    
    Returns:
        BipedalWalker 环境
    """
    import gymnasium as gym
    return gym.make('BipedalWalker-v3', **kwargs)


def create_pong_env(**kwargs):
    """
    创建 Pong 环境
    
    Args:
        **kwargs: 传递给 gymnasium.make 的参数
    
    Returns:
        Pong 环境
    """
    import gymnasium as gym
    return gym.make('ALE/Pong-v5', **kwargs)
