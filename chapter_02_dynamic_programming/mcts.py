"""
蒙特卡洛树搜索 (MCTS) 实现

MCTS 是一种基于采样的规划算法，适用于：
- 状态空间过大无法枚举
- 环境模型已知或可模拟
- 需要在线决策的场景
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class MCTSNode:
    """MCTS 树节点"""
    state: any                    # 状态
    parent: Optional['MCTSNode']  # 父节点
    children: dict                # 子节点 {action: MCTSNode}
    visits: int                   # 访问次数 N(s)
    value: float                  # 累计价值 ΣQ(s,a)
    action: Optional[int]         # 导致此节点的动作
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def q_value(self) -> float:
        """平均动作价值 Q(s,a)"""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits


class MCTS:
    """
    蒙特卡洛树搜索
    
    核心流程：
    1. Selection: 用 UCT 公式选择子节点
    2. Expansion: 扩展新节点
    3. Simulation: 随机模拟到终止
    4. Backpropagation: 回溯更新
    """
    
    def __init__(
        self,
        env,
        n_iterations: int = 1000,
        exploration_constant: float = 1.414,  # √2
        max_depth: int = 100,
        gamma: float = 0.99
    ):
        """
        Args:
            env: 环境（需要有 reset, step, action_space, get_legal_actions 方法）
            n_iterations: MCTS 迭代次数
            exploration_constant: UCT 探索常数 c
            max_depth: 最大模拟深度
            gamma: 折扣因子
        """
        self.env = env
        self.n_iterations = n_iterations
        self.c = exploration_constant
        self.max_depth = max_depth
        self.gamma = gamma
        self.root = None
    
    def uct_score(self, node: MCTSNode, action: int) -> float:
        """
        计算 UCT 分数
        
        UCT(s,a) = Q(s,a) + c * sqrt(ln N(s) / N(s,a))
        """
        if action not in node.children:
            return float('inf')  # 未探索的动作优先
        
        child = node.children[action]
        if child.visits == 0:
            return float('inf')
        
        exploitation = child.q_value
        exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
        
        return exploitation + exploration
    
    def select(self, node: MCTSNode) -> Tuple[MCTSNode, int]:
        """
        选择阶段：用 UCT 公式选择动作和子节点
        """
        legal_actions = self._get_legal_actions(node.state)
        
        if not legal_actions:
            return node, None
        
        # 选择 UCT 分数最高的动作
        best_action = max(legal_actions, key=lambda a: self.uct_score(node, a))
        return node.children[best_action], best_action
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        扩展阶段：为未完全扩展的节点添加子节点
        """
        legal_actions = self._get_legal_actions(node.state)
        
        # 找到未探索的动作
        unexplored = [a for a in legal_actions if a not in node.children]
        
        if not unexplored:
            return node
        
        # 随机选择一个未探索的动作
        action = np.random.choice(unexplored)
        
        # 执行动作获取新状态
        next_state = self._simulate_step(node.state, action)
        
        # 创建新节点
        child = MCTSNode(
            state=next_state,
            parent=node,
            children={},
            visits=0,
            value=0.0,
            action=action
        )
        node.children[action] = child
        
        return child
    
    def simulate(self, state: any) -> float:
        """
        模拟阶段：从给定状态随机模拟到终止
        
        Returns:
            累计折扣奖励
        """
        total_reward = 0.0
        discount = 1.0
        current_state = state
        
        for depth in range(self.max_depth):
            legal_actions = self._get_legal_actions(current_state)
            
            if not legal_actions:
                break
            
            # 随机选择动作
            action = np.random.choice(legal_actions)
            
            # 执行动作
            next_state, reward, terminated, _, _ = self._env_step(current_state, action)
            
            total_reward += discount * reward
            discount *= self.gamma
            current_state = next_state
            
            if terminated:
                break
        
        return total_reward
    
    def backpropagate(self, node: MCTSNode, value: float):
        """
        回溯阶段：将模拟结果回传更新路径上的所有节点
        """
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def search(self, state: any) -> int:
        """
        执行 MCTS 搜索
        
        Args:
            state: 当前状态
        
        Returns:
            最佳动作
        """
        # 创建根节点
        self.root = MCTSNode(
            state=state,
            parent=None,
            children={},
            visits=0,
            value=0.0,
            action=None
        )
        
        for _ in range(self.n_iterations):
            # 1. Selection
            node = self.root
            while node in [n for n in self.root.traverse()] and all(a in node.children for a in self._get_legal_actions(node.state)):
                if not node.children:
                    break
                node, _ = self.select(node)
            
            # 2. Expansion
            if self._get_legal_actions(node.state):
                node = self.expand(node)
            
            # 3. Simulation
            reward = self.simulate(node.state)
            
            # 4. Backpropagation
            self.backpropagate(node, reward)
        
        # 选择访问次数最多的动作
        return self._get_best_action(self.root)
    
    def _get_best_action(self, node: MCTSNode) -> int:
        """选择访问次数最多的动作"""
        if not node.children:
            legal_actions = self._get_legal_actions(node.state)
            return np.random.choice(legal_actions) if legal_actions else 0
        
        return max(node.children.keys(), key=lambda a: node.children[a].visits)
    
    def _get_legal_actions(self, state: any) -> list:
        """获取合法动作列表"""
        if hasattr(self.env, 'get_legal_actions'):
            return self.env.get_legal_actions(state)
        elif hasattr(self.env, 'action_space'):
            return list(range(self.env.action_space.n))
        return []
    
    def _simulate_step(self, state: any, action: int) -> any:
        """模拟环境步（不改变真实环境）"""
        # 保存状态
        if hasattr(self.env, 'clone_state'):
            saved_state = self.env.clone_state()
        else:
            saved_state = None
        
        # 执行动作
        next_state, _, _, _, _ = self._env_step(state, action)
        
        # 恢复状态
        if saved_state is not None and hasattr(self.env, 'set_state'):
            self.env.set_state(saved_state)
        
        return next_state
    
    def _env_step(self, state: any, action: int) -> Tuple:
        """执行环境步"""
        if hasattr(self.env, 'step_from_state'):
            return self.env.step_from_state(state, action)
        else:
            # 临时设置状态并执行
            if hasattr(self.env, 'set_state'):
                self.env.set_state(state)
            return self.env.step(action)


# 添加遍历方法到 MCTSNode
def _traverse(node: MCTSNode):
    """遍历所有可达节点"""
    yield node
    for child in node.children.values():
        yield from _traverse(child)

MCTSNode.traverse = _traverse
