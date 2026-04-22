"""
GRPO (Group Relative Policy Optimization) 算法实现

GRPO 是一种针对 LLM 推理任务优化的算法，通过组内比较自动产生监督信号，
无需显式奖励模型。特别适用于数学、代码等有明确正确答案的任务。

核心思想：
- 对同一 prompt 生成 G 个回答
- 通过执行结果/规则评分计算奖励
- 组内标准化优势：A_i = (r_i - mean(r)) / std(r)
- 使用 PPO 风格更新

数学公式：
对于每个 prompt x，生成 G 个回答 {y_1, ..., y_G}
计算每个回答的奖励 {r_1, ..., r_G}
优势估计：A_i = (r_i - mean(r)) / std(r)

L_GRPO(θ) = E[Σ_i min(r_i(θ) * A_i, clip(r_i(θ), 1-ε, 1+ε) * A_i)]

参考文献：
Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
https://arxiv.org/abs/2402.03300
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GRPOConfig:
    """GRPO 配置"""
    lr: float = 1e-5
    gamma: float = 1.0  # 通常为 1.0（单步决策）
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    group_size: int = 8  # 每组生成的回答数量
    ppo_epochs: int = 2
    batch_size: int = 32
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GRPOValueNetwork(nn.Module):
    """
    价值网络（用于基线估计）
    
    输入：prompt 编码
    输出：状态价值估计
    """
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        embeds = self.embedding(input_ids)  # [B, L, D]
        pooled = embeds.mean(dim=1)  # [B, D]
        return self.network(pooled)  # [B, 1]


class GRPOPolicyNetwork(nn.Module):
    """
    GRPO 策略网络（简化版 LLM）
    
    实际应用中应使用预训练 LLM
    """
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """编码输入"""
        embeds = self.embedding(input_ids)
        pooled = embeds.mean(dim=1)
        return self.encoder(pooled)
    
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输出 logits"""
        hidden = self.forward(input_ids)
        return self.decoder(hidden)
    
    def get_log_probs(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """计算目标序列的对数概率"""
        hidden = self.forward(input_ids)
        logits = self.decoder(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return target_log_probs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入 token IDs [B, L]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            do_sample: 是否采样（False 则 greedy decoding）
        
        Returns:
            generated_ids: 生成的 token IDs [B, max_new_tokens]
        """
        self.eval()
        generated = []
        
        with torch.no_grad():
            curr_ids = input_ids
            
            for _ in range(max_new_tokens):
                logits = self.get_logits(curr_ids)  # [B, V]
                
                if do_sample and temperature > 0:
                    # 采样
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)  # [B, 1]
                else:
                    # Greedy
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated.append(next_token)
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)
        
        return torch.cat(generated, dim=-1)  # [B, max_new_tokens]


class MathRewardFunction:
    """
    数学问题奖励函数
    
    通过执行结果自动评分：
    - 答案正确：+1
    - 答案错误：0
    - 格式错误：-0.5
    """
    
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0
    
    def __call__(self, prompt: str, response: str, ground_truth: Optional[str] = None) -> float:
        """
        计算奖励
        
        Args:
            prompt: 问题
            response: 模型回答
            ground_truth: 正确答案（如果有）
        
        Returns:
            reward: 奖励值
        """
        self.total_count += 1
        
        # 尝试提取答案
        answer = self._extract_answer(response)
        
        if ground_truth is not None:
            # 有标准答案：直接比较
            if self._is_correct(answer, ground_truth):
                self.correct_count += 1
                return 1.0
            else:
                return 0.0
        else:
            # 无标准答案：检查格式
            if answer is not None:
                return 0.5
            else:
                return 0.0  # 改为 0.0，表示中性
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """从回答中提取答案"""
        import re
        
        # 尝试匹配 "答案是 X" 或 "### X" 或 "\\boxed{X}" 等格式
        patterns = [
            r'答案是 [:：]?\s*([^\n。]+)',
            r'###\s*([^\n]+)',
            r'\\boxed\{([^\}]+)\}',
            r'\$\$?\s*([^\$]+)\s*\$\$?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 如果没有明确格式，尝试提取最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]
        
        return None
    
    def _is_correct(self, answer: Optional[str], ground_truth: str) -> bool:
        """检查答案是否正确"""
        if answer is None:
            return False
        
        # 数值比较（允许小误差）
        try:
            ans_num = float(answer)
            gt_num = float(ground_truth)
            return abs(ans_num - gt_num) < 1e-6
        except ValueError:
            # 字符串比较
            return answer.strip().lower() == ground_truth.strip().lower()
    
    def get_accuracy(self) -> float:
        """获取准确率"""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
    def reset(self):
        """重置统计"""
        self.correct_count = 0
        self.total_count = 0


class CodeRewardFunction:
    """
    代码问题奖励函数
    
    通过代码执行结果评分：
    - 通过所有测试用例：+1
    - 通过部分测试用例：+0.5
    - 编译/运行错误：0
    """
    
    def __init__(self, test_cases: Optional[List[Dict]] = None):
        self.test_cases = test_cases or []
        self.pass_count = 0
        self.total_count = 0
    
    def __call__(self, prompt: str, response: str, test_cases: Optional[List[Dict]] = None) -> float:
        """
        计算奖励
        
        Args:
            prompt: 问题
            response: 模型生成的代码
            test_cases: 测试用例列表
        
        Returns:
            reward: 奖励值
        """
        self.total_count += 1
        test_cases = test_cases or self.test_cases
        
        if not test_cases:
            # 没有测试用例：检查代码格式
            if self._is_valid_code(response):
                return 0.5
            else:
                return 0.0
        
        # 执行测试
        passed = 0
        for tc in test_cases:
            if self._run_test(response, tc):
                passed += 1
        
        self.pass_count += 1 if passed == len(test_cases) else 0
        
        return passed / len(test_cases)
    
    def _is_valid_code(self, code: str) -> bool:
        """检查代码是否有效（简化版）"""
        # 检查基本语法
        import re
        
        # 提取代码块
        match = re.search(r'```(?:python)?\n(.+?)```', code, re.DOTALL)
        if match:
            code = match.group(1)
        
        # 检查是否有 def/class 等关键字
        keywords = ['def ', 'class ', 'import ', 'from ', 'return ']
        return any(kw in code for kw in keywords)
    
    def _run_test(self, code: str, test_case: Dict) -> bool:
        """运行单个测试用例（简化版，实际应使用沙箱）"""
        # 注意：实际应用中需要使用安全的代码执行沙箱
        # 这里只是演示
        try:
            # 提取代码
            import re
            match = re.search(r'```(?:python)?\n(.+?)```', code, re.DOTALL)
            if match:
                code = match.group(1)
            
            # 执行代码（危险！仅用于演示）
            # 实际应用中应使用 subprocess + 沙箱
            exec_globals = {}
            exec(code, exec_globals)
            
            # 运行测试
            input_val = test_case.get('input')
            expected = test_case.get('expected')
            test_func = test_case.get('test_func')
            
            if test_func:
                result = exec_globals.get(test_func)(input_val)
                return result == expected
            
            return True
        except Exception:
            return False
    
    def get_pass_rate(self) -> float:
        """获取通过率"""
        if self.total_count == 0:
            return 0.0
        return self.pass_count / self.total_count


class GRPOTrainer:
    """
    GRPO 训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        reward_fn: Optional[Callable] = None,
        config: Optional[GRPOConfig] = None,
        tokenizer: Optional[Callable] = None,
        group_size: Optional[int] = None,
    ):
        self.config = config or GRPOConfig()
        if group_size is not None:
            self.config.group_size = group_size
        self.model = model.to(self.config.device)
        
        # 参考模型
        if ref_model is not None:
            self.ref_model = ref_model.to(self.config.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = type(model)()
            self.ref_model.load_state_dict(model.state_dict())
            self.ref_model.to(self.config.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # 价值网络
        self.value_network = GRPOValueNetwork().to(self.config.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.lr)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        
        # 奖励函数
        self.reward_fn = reward_fn or MathRewardFunction()
        
        # 分词器
        self.tokenizer = tokenizer
        
        # 训练统计
        self.total_updates = 0
        self.training_history = []
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """文本分词"""
        if self.tokenizer is not None:
            return self.tokenizer(text)
        
        # 简单分词
        vocab_size = 1000
        ids = [hash(c) % (vocab_size - 100) + 100 for c in text[:self.config.max_length]]
        ids = ids + [0] * (self.config.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)
    
    def _generate_group(
        self,
        prompt: str,
        group_size: int,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        为一组 prompt 生成多个回答
        
        Returns:
            responses: [(response_text, input_ids), ...]
        """
        input_ids = self._tokenize(prompt).unsqueeze(0).to(self.config.device)
        
        responses = []
        for _ in range(group_size):
            # 生成
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
            )
            
            # 解码（简化：直接转字符串）
            response_text = f"generated_{generated_ids.sum().item()}"
            responses.append((response_text, generated_ids))
        
        return responses
    
    def _compute_rewards(
        self,
        prompt: str,
        responses: List[Tuple[str, torch.Tensor]],
        ground_truth: Optional[str] = None,
    ) -> torch.Tensor:
        """
        计算组内每个回答的奖励
        
        Returns:
            rewards: [G]
        """
        rewards = []
        for response_text, _ in responses:
            reward = self.reward_fn(prompt, response_text, ground_truth)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算组内标准化优势
        
        A_i = (r_i - mean(r)) / (std(r) + eps)
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages
    
    def train_step(
        self,
        prompts: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            prompts: prompt 列表
            ground_truths: 标准答案列表（可选）
        
        Returns:
            stats: 训练统计
        """
        self.model.train()
        self.value_network.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_reward = 0.0
        n_samples = 0
        
        for i, prompt in enumerate(prompts):
            ground_truth = ground_truths[i] if ground_truths else None
            
            # 1. 生成一组回答
            responses = self._generate_group(prompt, self.config.group_size)
            
            # 2. 计算奖励
            rewards = self._compute_rewards(prompt, responses, ground_truth)
            total_reward += rewards.mean().item()
            
            # 3. 计算优势
            advantages = self._compute_advantages(rewards)
            
            # 4. 计算价值函数损失
            input_ids = self._tokenize(prompt).unsqueeze(0).to(self.config.device)
            values = self.value_network(input_ids).squeeze()
            
            value_loss = ((values - rewards.mean()) ** 2).mean()
            
            # 5. PPO 风格更新
            for _ in range(self.config.ppo_epochs):
                # 重新计算价值损失（避免梯度图重用）
                values = self.value_network(input_ids).squeeze()
                value_loss = ((values - rewards.mean()) ** 2).mean()
                
                # 计算策略比率（简化：使用生成时的对数概率）
                # 实际应用中需要重新计算
                log_ratio = torch.randn(self.config.group_size, requires_grad=True) * 0.1  # 模拟
                
                ratio = torch.exp(log_ratio)
                
                # 截断策略损失
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 熵奖励（简化）
                entropy = torch.tensor(0.1)
                
                # 总损失
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # 优化
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_samples += 1
        
        stats = {
            'policy_loss': total_policy_loss / max(1, n_samples),
            'value_loss': total_value_loss / max(1, n_samples),
            'entropy': total_entropy / max(1, n_samples),
            'mean_reward': total_reward / max(1, len(prompts)),
        }
        
        self.total_updates += 1
        self.training_history.append(stats)
        
        return stats
    
    def train(
        self,
        data: List[Tuple[str, Optional[str]]],
        n_epochs: int = 10,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        训练 GRPO 模型
        
        Args:
            data: 数据列表 [(prompt, ground_truth), ...]
            n_epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
        
        Returns:
            history: 训练历史
        """
        batch_size = batch_size or self.config.batch_size
        history = []
        
        for epoch in range(n_epochs):
            np.random.shuffle(data)
            epoch_stats = []
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                prompts = [item[0] for item in batch]
                ground_truths = [item[1] for item in batch if len(item) > 1]
                ground_truths = ground_truths if len(ground_truths) == len(prompts) else None
                
                stats = self.train_step(prompts, ground_truths)
                epoch_stats.append(stats)
            
            avg_stats = {
                key: np.mean([s[key] for s in epoch_stats])
                for key in epoch_stats[0]
            }
            history.append(avg_stats)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} | "
                      f"Policy Loss: {avg_stats['policy_loss']:.4f} | "
                      f"Value Loss: {avg_stats['value_loss']:.4f} | "
                      f"Mean Reward: {avg_stats['mean_reward']:.4f}")
        
        return history
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'value_network': self.value_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'total_updates': self.total_updates,
            'training_history': self.training_history,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.total_updates = checkpoint['total_updates']
        self.training_history = checkpoint['training_history']
