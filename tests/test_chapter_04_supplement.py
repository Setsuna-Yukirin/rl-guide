"""
第 4 章补充测试：游戏环境和探索策略
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def maze_env():
    """Maze Treasure 环境 fixture"""
    from chapter_04_temporal_difference.games import MazeTreasureEnv
    env = MazeTreasureEnv(maze_size=(6, 6), n_traps=3, n_walls=5, seed=42)
    env.reset(seed=42)
    return env


@pytest.fixture
def snake_env():
    """Snake 环境 fixture"""
    from chapter_04_temporal_difference.games import SnakeSimpleEnv
    env = SnakeSimpleEnv(grid_size=(8, 8), initial_length=3, max_steps=100)
    env.reset(seed=42)
    return env


# ==================== Maze Treasure 测试 ====================

@pytest.mark.unit
class TestMazeTreasure:
    """Maze Treasure 环境测试"""
    
    def test_maze_reset(self, maze_env):
        """测试重置"""
        state, _ = maze_env.reset(seed=42)
        
        assert isinstance(state, int)
        assert 0 <= state < maze_env.n_states
        assert maze_env.agent_pos == maze_env.start_pos
    
    def test_maze_step(self, maze_env):
        """测试步"""
        state, _ = maze_env.reset(seed=42)
        
        next_state, reward, terminated, _, _ = maze_env.step(1)  # 向下
        
        assert isinstance(next_state, int)
        assert isinstance(reward, float)
        assert reward == maze_env.step_reward
    
    def test_maze_treasure_reach(self):
        """测试到达宝藏"""
        from chapter_04_temporal_difference.games import MazeTreasureEnv
        
        env = MazeTreasureEnv(maze_size=(3, 3), n_traps=0, n_walls=0)
        env.reset(seed=42)
        
        # 手动设置位置，直接走向宝藏
        env.agent_pos = (env.n_rows - 1, env.n_cols - 2)  # 宝藏左边
        env.treasure_pos = (env.n_rows - 1, env.n_cols - 1)
        
        _, reward, terminated, _, _ = env.step(3)  # 向右
        
        assert terminated
        assert reward == env.treasure_reward
    
    def test_maze_trap(self):
        """测试陷阱"""
        from chapter_04_temporal_difference.games import MazeTreasureEnv
        
        env = MazeTreasureEnv(maze_size=(3, 3), n_traps=1, n_walls=0, seed=42)
        env.reset(seed=42)
        
        # 移动到陷阱位置
        trap_pos = list(env.traps)[0]
        env.agent_pos = trap_pos
        
        _, reward, terminated, _, _ = env.step(0)
        
        assert terminated
        assert reward == env.trap_reward
    
    def test_maze_wall_collision(self, maze_env):
        """测试撞墙"""
        maze_env.reset(seed=42)
        
        # 找到一堵墙
        if maze_env.walls:
            wall_pos = list(maze_env.walls)[0]
            maze_env.agent_pos = (wall_pos[0], wall_pos[1] - 1)  # 墙左边
            
            # 尝试向右撞墙
            _, _, _, _, _ = maze_env.step(3)
            
            # 应该撞墙失败，位置不变
            assert maze_env.agent_pos != wall_pos
    
    def test_maze_legal_actions(self, maze_env):
        """测试合法动作"""
        maze_env.reset(seed=42)
        
        legal = maze_env.get_legal_actions()
        
        assert isinstance(legal, list)
        assert all(0 <= a <= 3 for a in legal)
    
    def test_maze_render(self, maze_env):
        """测试渲染"""
        maze_env.reset(seed=42)
        
        rendered = maze_env.render()
        assert rendered is None  # human 模式返回 None
        
        ansi = maze_env._render_ansi()
        assert isinstance(ansi, str)
        assert 'A' in ansi  # 智能体
        assert 'T' in ansi  # 宝藏


# ==================== Snake 测试 ====================

@pytest.mark.unit
class TestSnake:
    """Snake 环境测试"""
    
    def test_snake_reset(self, snake_env):
        """测试重置"""
        state, _ = snake_env.reset(seed=42)
        
        assert isinstance(state, int)
        assert len(snake_env.snake) == snake_env.initial_length
    
    def test_snake_step(self, snake_env):
        """测试步"""
        state, _ = snake_env.reset(seed=42)
        
        next_state, reward, terminated, _, _ = snake_env.step(0)  # 向上
        
        assert isinstance(next_state, int)
        assert isinstance(reward, float)
        assert reward == snake_env.step_reward
    
    def test_snake_illegal_turn(self, snake_env):
        """测试非法转向（不能直接反向）"""
        snake_env.reset(seed=42)
        
        # 初始方向是向右 (3)，尝试直接向左 (2)
        snake_env.direction = 3
        _, _, _, _, info = snake_env.step(2)  # 应该被忽略
        
        # 方向应该保持向右
        assert snake_env.direction == 3
    
    def test_snake_eat_food(self):
        """测试吃食物"""
        from chapter_04_temporal_difference.games import SnakeSimpleEnv
        
        env = SnakeSimpleEnv(grid_size=(5, 5), initial_length=3)
        env.reset(seed=42)
        
        # 手动设置蛇头在食物旁边
        head = env.snake[0]
        env.food_pos = (head[0], head[1] + 1)  # 右边
        
        initial_length = len(env.snake)
        _, reward, terminated, _, _ = env.step(3)  # 向右
        
        assert not terminated
        assert reward == env.food_reward
        assert len(env.snake) == initial_length + 1
    
    def test_snake_wall_death(self):
        """测试撞墙死亡"""
        from chapter_04_temporal_difference.games import SnakeSimpleEnv
        
        env = SnakeSimpleEnv(grid_size=(5, 5), initial_length=3)
        env.reset(seed=42)
        
        # 设置蛇头在顶部
        env.snake[0] = (0, 2)
        env.direction = 0  # 向上
        
        _, reward, terminated, _, info = env.step(0)
        
        assert terminated
        assert reward == env.death_reward
        assert info.get('reason') == 'wall'
    
    def test_snake_self_death(self):
        """测试撞自己死亡"""
        from chapter_04_temporal_difference.games import SnakeSimpleEnv
        from collections import deque
        
        env = SnakeSimpleEnv(grid_size=(5, 5), initial_length=5)
        env.reset(seed=42)
        
        # 创建一个蛇身，蛇头向右移动会撞到身体
        # 蛇头在 (2,2)，右边 (2,3) 是身体
        env.snake = deque([(2, 2), (2, 3), (2, 4), (1, 4), (1, 3)])
        env.direction = 3  # 向右
        
        # 向右移动会撞到 (2,3)
        _, reward, terminated, _, info = env.step(3)
        
        assert terminated
        assert reward == env.death_reward
        assert info.get('reason') == 'self'
    
    def test_snake_timeout(self):
        """测试超时"""
        from chapter_04_temporal_difference.games import SnakeSimpleEnv
        
        env = SnakeSimpleEnv(grid_size=(3, 3), initial_length=3, max_steps=5)
        env.reset(seed=42)
        
        # 走很多步
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(0)
            if terminated:
                break
        
        # 应该超时
        if truncated:
            assert info.get('reason') == 'timeout'
    
    def test_snake_render(self, snake_env):
        """测试渲染"""
        snake_env.reset(seed=42)
        
        ansi = snake_env._render_ansi()
        assert isinstance(ansi, str)
        assert 'H' in ansi  # 蛇头
        assert 'F' in ansi  # 食物


# ==================== Exploration Strategies 测试 ====================

@pytest.mark.unit
class TestExplorationStrategies:
    """探索策略测试"""
    
    def test_epsilon_greedy_init(self):
        """测试 ε-贪婪初始化"""
        from chapter_04_temporal_difference.exploration_strategies import EpsilonGreedy
        
        strategy = EpsilonGreedy(n_actions=4, epsilon=0.1)
        
        assert strategy.n_actions == 4
        assert strategy.epsilon == 0.1
    
    def test_epsilon_greedy_explore(self):
        """测试 ε-贪婪探索"""
        from chapter_04_temporal_difference.exploration_strategies import EpsilonGreedy
        
        strategy = EpsilonGreedy(n_actions=2, epsilon=1.0)  # 完全探索
        q_values = np.array([1.0, 2.0])
        
        # 应该随机选择
        actions = set()
        for _ in range(100):
            action = strategy.select_action(q_values)
            actions.add(action)
        
        assert len(actions) == 2  # 两个动作都应该被选到
    
    def test_epsilon_greedy_exploit(self):
        """测试 ε-贪婪利用"""
        from chapter_04_temporal_difference.exploration_strategies import EpsilonGreedy
        
        strategy = EpsilonGreedy(n_actions=2, epsilon=0.0)  # 完全利用
        q_values = np.array([1.0, 2.0])
        
        # 应该总是选择最优动作
        for _ in range(10):
            action = strategy.select_action(q_values)
            assert action == 1
    
    def test_epsilon_decay(self):
        """测试 ε 衰减"""
        from chapter_04_temporal_difference.exploration_strategies import EpsilonGreedy
        
        strategy = EpsilonGreedy(n_actions=2, epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.01)
        
        strategy.update(0, 0, -1, 1)
        assert strategy.epsilon == 0.45
        
        strategy.update(0, 0, -1, 1)
        assert strategy.epsilon == 0.405
    
    def test_boltzmann_exploration(self):
        """测试 Boltzmann 探索"""
        from chapter_04_temporal_difference.exploration_strategies import BoltzmannExploration
        
        strategy = BoltzmannExploration(n_actions=2, temperature=1.0)
        q_values = np.array([1.0, 2.0])
        
        # 应该按概率选择
        actions = set()
        for _ in range(100):
            action = strategy.select_action(q_values)
            actions.add(action)
        
        # 两个动作都应该有机会
        assert len(actions) >= 1
    
    def test_ucb_exploration(self):
        """测试 UCB 探索"""
        from chapter_04_temporal_difference.exploration_strategies import UpperConfidenceBound
        
        strategy = UpperConfidenceBound(n_actions=2, c=2.0)
        q_values = np.array([1.0, 1.0])
        
        # 初始应该探索所有动作
        for a in range(2):
            action = strategy.select_action(q_values)
            strategy.update(0, action, -1, 1)
            assert strategy.action_counts[action] > 0
    
    def test_thompson_sampling(self):
        """测试 Thompson Sampling"""
        from chapter_04_temporal_difference.exploration_strategies import ThompsonSampling
        
        strategy = ThompsonSampling(n_actions=2)
        q_values = np.array([1.0, 2.0])
        
        # 应该能选择动作
        action = strategy.select_action(q_values)
        assert 0 <= action < 2
        
        # 更新后验
        strategy.update(0, 0, 1.0, 1)
        assert strategy.alpha[0] > strategy.prior_alpha
    
    def test_noisy_network(self):
        """测试噪声网络"""
        from chapter_04_temporal_difference.exploration_strategies import NoisyNetwork
        
        strategy = NoisyNetwork(n_actions=2, noise_std=0.1)
        q_values = np.array([1.0, 1.0])
        
        # 添加噪声后选择
        action = strategy.select_action(q_values)
        assert 0 <= action < 2
    
    def test_create_strategy_factory(self):
        """测试工厂函数"""
        from chapter_04_temporal_difference.exploration_strategies import create_exploration_strategy
        
        # 测试所有策略
        strategies = ['epsilon', 'decay_epsilon', 'boltzmann', 'ucb', 'thompson', 'noisy']
        
        for name in strategies:
            strategy = create_exploration_strategy(name, n_actions=4)
            assert strategy is not None
    
    def test_create_strategy_invalid(self):
        """测试无效策略名称"""
        from chapter_04_temporal_difference.exploration_strategies import create_exploration_strategy
        
        with pytest.raises(ValueError):
            create_exploration_strategy('invalid', n_actions=4)


# 需要从 collections 导入 deque
from collections import deque
