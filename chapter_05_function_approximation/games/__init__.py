"""
第 5 章游戏模块
"""

# CartPole - 使用 gymnasium 原生环境
# Breakout - 使用 gymnasium 的 Atari 环境
# Lunar Lander - 使用 gymnasium 原生环境

__all__ = [
    'create_cartpole_env',
    'create_breakout_env',
    'create_lunar_lander_env',
]


def create_cartpole_env(**kwargs):
    """
    创建 CartPole 环境
    
    Args:
        **kwargs: 传递给 gymnasium.make 的参数
    
    Returns:
        CartPole 环境
    
    Example:
        >>> env = create_cartpole_env()
        >>> state, _ = env.reset()
    """
    import gymnasium as gym
    return gym.make('CartPole-v1', **kwargs)


def create_breakout_env(
    frameskip: int = 4,
    frame_size: int = 84,
    grayscale: bool = True,
    **kwargs
):
    """
    创建 Breakout 环境（带预处理）
    
    Args:
        frameskip: 帧跳过（每 n 帧执行一次动作）
        frame_size: 帧大小
        grayscale: 是否灰度化
        **kwargs: 其他参数
    
    Returns:
        Breakout 环境（预处理后）
    """
    import gymnasium as gym
    
    env = gym.make('ALE/Breakout-v5', **kwargs)
    
    # 应用包装器
    if grayscale:
        env = gym.wrappers.GrayscaleObservation(env)
    
    env = gym.wrappers.ResizeObservation(env, (frame_size, frame_size))
    env = gym.wrappers.FrameStackObservation(env, frameskip)
    
    return env


def create_lunar_lander_env(continuous: bool = False, **kwargs):
    """
    创建 Lunar Lander 环境
    
    Args:
        continuous: 是否连续动作空间
        **kwargs: 传递给 gymnasium.make 的参数
    
    Returns:
        Lunar Lander 环境
    """
    import gymnasium as gym
    
    env_name = 'LunarLander-v3' if not continuous else 'LunarLanderContinuous-v3'
    return gym.make(env_name, **kwargs)
