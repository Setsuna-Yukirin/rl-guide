"""
可视化工具

提供强化学习训练和结果的可视化功能：
- 学习曲线
- 价值函数热力图
- 策略箭头图
- 训练动画
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import seaborn as sns


def plot_learning_curve(
    rewards: Union[List[float], np.ndarray],
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    window_size: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    style: str = "seaborn-v0_8",
):
    """
    绘制学习曲线
    
    Args:
        rewards: 每集的奖励列表
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        window_size: 滑动窗口大小（用于平滑）
        save_path: 保存路径（可选）
        show: 是否显示图表
        figsize: 图表大小
        style: matplotlib 风格
    
    Example:
        >>> rewards = [np.random.randn() for _ in range(100)]
        >>> plot_learning_curve(rewards, window_size=10)
    """
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)
    
    episodes = np.arange(len(rewards))
    
    # 绘制原始曲线
    ax.plot(episodes, rewards, alpha=0.3, label="Raw", linewidth=1)
    
    # 绘制平滑曲线
    if window_size is not None and window_size > 1:
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        ax.plot(smoothed_episodes, smoothed, label=f"Smoothed (window={window_size})", 
                linewidth=2, color='red')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def plot_value_heatmap(
    values: np.ndarray,
    title: str = "Value Function Heatmap",
    xlabel: str = "X Position",
    ylabel: str = "Y Position",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    annotate: bool = False,
):
    """
    绘制价值函数热力图
    
    Args:
        values: 价值数组，形状为 (height, width)
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        cmap: 颜色映射
        save_path: 保存路径（可选）
        show: 是否显示图表
        figsize: 图表大小
        annotate: 是否标注数值
    
    Example:
        >>> values = np.random.randn(10, 10)
        >>> plot_value_heatmap(values, annotate=True)
    """
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    im = ax.imshow(values, cmap=cmap, aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value", fontsize=12)
    
    # 标注数值
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                text = ax.text(j, i, f"{values[i, j]:.2f}",
                              ha="center", va="center", color="white",
                              fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax, im


def plot_policy_arrows(
    policy: np.ndarray,
    value: Optional[np.ndarray] = None,
    title: str = "Policy",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    绘制策略箭头图
    
    箭头方向表示策略选择的动作，颜色表示价值。
    
    Args:
        policy: 策略数组，形状为 (height, width, 2)，每个元素是 (dx, dy)
        value: 价值数组，形状为 (height, width)（可选）
        title: 图表标题
        save_path: 保存路径（可选）
        show: 是否显示图表
        figsize: 图表大小
    
    Example:
        >>> policy = np.random.randn(10, 10, 2)
        >>> value = np.random.randn(10, 10)
        >>> plot_policy_arrows(policy, value)
    """
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)
    
    height, width = policy.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    
    # 绘制价值背景
    if value is not None:
        im = ax.imshow(value, cmap='viridis', aspect='auto', alpha=0.5)
        plt.colorbar(im, ax=ax, label="Value")
    
    # 绘制箭头
    ax.quiver(x, y, policy[:, :, 0], policy[:, :, 1], 
              angles='xy', scale_units='xy', scale=1, color='red')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def plot_training_stats(
    stats: Dict[str, List[float]],
    titles: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 10),
    ncols: int = 2,
):
    """
    绘制训练统计图表
    
    Args:
        stats: 统计字典 {metric_name: [values]}
        titles: 自定义标题字典 {metric_name: title}
        save_path: 保存路径（可选）
        show: 是否显示图表
        figsize: 图表大小
        ncols: 子图列数
    
    Example:
        >>> stats = {
        ...     'rewards': [np.random.randn() for _ in range(100)],
        ...     'losses': [np.random.rand() for _ in range(100)],
        ... }
        >>> plot_training_stats(stats)
    """
    plt.style.use("seaborn-v0_8")
    
    n_metrics = len(stats)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(stats.items()):
        ax = axes[idx]
        title = titles.get(metric_name, metric_name.replace('_', ' ').title()) if titles else metric_name
        
        ax.plot(values, alpha=0.7, linewidth=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Step" if "step" in metric_name.lower() else "Episode")
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


def create_training_animation(
    frames: List[np.ndarray],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    fps: int = 10,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    创建训练过程动画
    
    Args:
        frames: 帧列表，每帧是一个 RGB 数组
        title: 动画标题
        save_path: 保存路径（可选，.gif 或 .mp4）
        fps: 帧率
        figsize: 图表大小
    
    Example:
        >>> frames = [np.random.rand(100, 100, 3) for _ in range(50)]
        >>> create_training_animation(frames, save_path="training.gif")
    """
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 创建动画
    def update(frame):
        ax.imshow(frame)
        return [ax]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), 
                                   interval=1000/fps, blit=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            # 默认保存为 gif
            ani.save(save_path + '.gif', writer='pillow', fps=fps)
        print(f"动画已保存到：{save_path}")
    
    plt.close()
    
    return ani


if __name__ == "__main__":
    # 简单测试
    print("测试可视化工具...\n")
    
    # 测试学习曲线
    print("1. 测试 plot_learning_curve")
    rewards = [np.random.randn() + i * 0.1 for i in range(100)]
    fig, ax = plot_learning_curve(rewards, window_size=10, show=False)
    print("   ✓ plot_learning_curve 测试通过\n")
    
    # 测试热力图
    print("2. 测试 plot_value_heatmap")
    values = np.random.randn(10, 10)
    fig, ax, im = plot_value_heatmap(values, annotate=True, show=False)
    print("   ✓ plot_value_heatmap 测试通过\n")
    
    # 测试策略箭头
    print("3. 测试 plot_policy_arrows")
    policy = np.random.randn(10, 10, 2) * 0.5
    value = np.random.randn(10, 10)
    fig, ax = plot_policy_arrows(policy, value, show=False)
    print("   ✓ plot_policy_arrows 测试通过\n")
    
    # 测试训练统计
    print("4. 测试 plot_training_stats")
    stats = {
        'rewards': [np.random.randn() for _ in range(100)],
        'losses': [np.random.rand() for _ in range(100)],
        'epsilon': [0.9 ** i for i in range(100)],
    }
    fig, axes = plot_training_stats(stats, show=False)
    print("   ✓ plot_training_stats 测试通过\n")
    
    print("✅ 所有可视化工具测试通过！")
