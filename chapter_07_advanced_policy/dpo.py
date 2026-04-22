"""
DPO (Direct Preference Optimization) 算法实现

DPO 是一种直接从人类偏好数据优化策略的算法，无需显式奖励模型。
这是 LLM 后训练（对齐）的关键技术。

核心思想：
- 传统 RLHF：SFT → 奖励模型 → PPO 优化（复杂、不稳定）
- DPO：直接从偏好数据优化（简单、稳定）

数学公式：
L_DPO(π_θ; π_ref) = -E[log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) 
                              - β * log(π_θ(y_l|x)/π_ref(y_l|x)))]

其中：
- (x, y_w, y_l) 是偏好三元组（prompt, 优选回答，劣选回答）
- π_ref 是参考策略（通常是 SFT 模型）
- β 是温度参数

参考文献：
Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
https://arxiv.org/abs/2305.18290
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DPOConfig:
    """DPO 配置"""
    lr: float = 1e-5
    beta: float = 0.1  # 温度参数
    batch_size: int = 32
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleTextEncoder(nn.Module):
    """
    简化文本编码器（用于演示）
    
    实际应用中应使用预训练 LLM（如 LLaMA、GPT 等）
    """
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """编码输入"""
        embeds = self.embedding(input_ids)  # [B, L, D]
        pooled = embeds.mean(dim=1)  # [B, D]
        return self.encoder(pooled)  # [B, H]
    
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输出 logits"""
        hidden = self.forward(input_ids)
        return self.decoder(hidden)  # [B, V]
    
    def get_log_probs(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        计算目标序列的对数概率
        
        Args:
            input_ids: 输入 token IDs [B, L_in]
            target_ids: 目标 token IDs [B] (简化：只考虑单个 token)
        
        Returns:
            log_probs: 对数概率 [B]
        """
        hidden = self.forward(input_ids)  # [B, H]
        logits = self.decoder(hidden)  # [B, V]
        
        # 计算目标 token 的对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 确保 target_ids 是正确的形状
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(-1)  # [B, 1]
        
        target_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)
        
        return target_log_probs


class DPOTrainer:
    """
    DPO 训练器
    
    直接从偏好数据优化策略，无需奖励模型
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        config: Optional[DPOConfig] = None,
        tokenizer: Optional[Callable] = None,
        beta: Optional[float] = None,
    ):
        self.config = config or DPOConfig()
        if beta is not None:
            self.config.beta = beta
        self.model = model.to(self.config.device)
        
        # 参考模型（通常是 SFT 模型，冻结参数）
        if ref_model is not None:
            self.ref_model = ref_model.to(self.config.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            # 如果没有提供参考模型，创建一个副本并冻结
            self.ref_model = type(model)()
            self.ref_model.load_state_dict(model.state_dict())
            self.ref_model.to(self.config.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        
        # 分词器（可选）
        self.tokenizer = tokenizer
        
        # 训练统计
        self.total_updates = 0
        self.training_history = []
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """
        文本分词
        
        如果提供了 tokenizer，使用它；否则使用简单的字符级分词
        """
        if self.tokenizer is not None:
            return self.tokenizer(text)
        
        # 简单的字符级分词（用于演示）
        # 实际应用中应使用 BPE/WordPiece 等
        vocab_size = 1000  # 与 SimpleTextEncoder 匹配
        # 哈希到 vocab 范围
        ids = [hash(c) % (vocab_size - 100) + 100 for c in text[:self.config.max_length]]
        ids = ids + [0] * (self.config.max_length - len(ids))  # padding
        return torch.tensor(ids, dtype=torch.long)
    
    def _compute_log_probs(
        self,
        model: nn.Module,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        计算模型对响应的对数概率
        
        Args:
            model: 模型
            prompts: prompt 列表
            responses: 响应列表
        
        Returns:
            log_probs: 对数概率 [B]
        """
        log_probs = []
        
        for prompt, response in zip(prompts, responses):
            if self.tokenizer is not None:
                # 使用真实 tokenizer
                input_ids = self.tokenizer(prompt)
                target_ids = self.tokenizer(response)
            else:
                # 简单分词
                input_ids = self._tokenize(prompt)
                target_ids = self._tokenize(response)
            
            input_ids = input_ids.unsqueeze(0).to(self.config.device)
            target_ids = target_ids.unsqueeze(0).to(self.config.device)
            
            with torch.no_grad() if model == self.ref_model else torch.enable_grad():
                log_prob = model.get_log_probs(input_ids, target_ids)
            
            log_probs.append(log_prob[0])
        
        return torch.stack(log_probs)
    
    def compute_dpo_loss(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 DPO 损失
        
        Args:
            prompts: prompt 列表
            chosen_responses: 优选响应列表
            rejected_responses: 劣选响应列表
        
        Returns:
            loss: DPO 损失
            stats: 训练统计
        """
        batch_size = len(prompts)
        
        # 计算模型的对数概率
        log_pi_chosen = self._compute_log_probs(self.model, prompts, chosen_responses)
        log_pi_rejected = self._compute_log_probs(self.model, prompts, rejected_responses)
        
        # 计算参考模型的对数概率
        with torch.no_grad():
            log_ref_chosen = self._compute_log_probs(self.ref_model, prompts, chosen_responses)
            log_ref_rejected = self._compute_log_probs(self.ref_model, prompts, rejected_responses)
        
        # 计算对数概率比
        log_ratio_chosen = log_pi_chosen - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected
        
        # DPO 损失
        # L = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
        logits = self.config.beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()
        
        # 统计信息
        stats = {
            'loss': loss.item(),
            'log_ratio_chosen': log_ratio_chosen.mean().item(),
            'log_ratio_rejected': log_ratio_rejected.mean().item(),
            'margin': (log_ratio_chosen - log_ratio_rejected).mean().item(),
            'accuracy': (logits > 0).float().mean().item(),
        }
        
        return loss, stats
    
    def train_step(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            prompts: prompt 列表
            chosen_responses: 优选响应列表
            rejected_responses: 劣选响应列表
        
        Returns:
            stats: 训练统计
        """
        self.model.train()
        
        # 计算损失
        loss, stats = self.compute_dpo_loss(prompts, chosen_responses, rejected_responses)
        
        # 反向传播
        loss.backward()
        
        # 更新
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.total_updates += 1
        self.training_history.append(stats)
        
        return stats
    
    def train(
        self,
        preference_data: List[Tuple[str, str, str]],
        n_epochs: int = 10,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        训练 DPO 模型
        
        Args:
            preference_data: 偏好数据列表 [(prompt, chosen, rejected), ...]
            n_epochs: 训练轮数
            batch_size: 批次大小（None 则使用配置值）
            verbose: 是否打印训练信息
        
        Returns:
            history: 训练历史
        """
        batch_size = batch_size or self.config.batch_size
        history = []
        
        for epoch in range(n_epochs):
            # 打乱数据
            np.random.shuffle(preference_data)
            
            epoch_stats = []
            
            for i in range(0, len(preference_data), batch_size):
                batch = preference_data[i:i + batch_size]
                prompts = [item[0] for item in batch]
                chosen = [item[1] for item in batch]
                rejected = [item[2] for item in batch]
                
                stats = self.train_step(prompts, chosen, rejected)
                epoch_stats.append(stats)
            
            # 平均统计
            avg_stats = {
                key: np.mean([s[key] for s in epoch_stats])
                for key in epoch_stats[0]
            }
            history.append(avg_stats)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs} | "
                      f"Loss: {avg_stats['loss']:.4f} | "
                      f"Accuracy: {avg_stats['accuracy']:.2%} | "
                      f"Margin: {avg_stats['margin']:.4f}")
        
        return history
    
    def evaluate(
        self,
        test_data: List[Tuple[str, str, str]],
        batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_data: 测试数据 [(prompt, chosen, rejected), ...]
            batch_size: 批次大小
        
        Returns:
            metrics: 评估指标
        """
        self.model.eval()
        batch_size = batch_size or self.config.batch_size
        
        all_accuracies = []
        all_margins = []
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i + batch_size]
                prompts = [item[0] for item in batch]
                chosen = [item[1] for item in batch]
                rejected = [item[2] for item in batch]
                
                _, stats = self.compute_dpo_loss(prompts, chosen, rejected)
                all_accuracies.append(stats['accuracy'])
                all_margins.append(stats['margin'])
        
        return {
            'accuracy': np.mean(all_accuracies),
            'margin': np.mean(all_margins),
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'training_history': self.training_history,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_updates = checkpoint['total_updates']
        self.training_history = checkpoint['training_history']


class LLMPreferenceDataset:
    """
    LLM 偏好数据集
    
    用于加载和管理偏好数据
    """
    
    def __init__(self):
        self.data = []
    
    def add(self, prompt: str, chosen: str, rejected: str, source: str = "human"):
        """
        添加偏好样本
        
        Args:
            prompt: 输入 prompt
            chosen: 优选响应
            rejected: 劣选响应
            source: 数据来源（human/ai/synthetic）
        """
        self.data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'source': source,
        })
    
    def to_list(self) -> List[Tuple[str, str, str]]:
        """转换为训练格式"""
        return [(item['prompt'], item['chosen'], item['rejected']) for item in self.data]
    
    def __len__(self) -> int:
        return len(self.data)
    
    @classmethod
    def from_jsonl(cls, path: str) -> 'LLMPreferenceDataset':
        """从 JSONL 文件加载"""
        import json
        
        dataset = cls()
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                dataset.add(
                    prompt=item.get('prompt', item.get('input', '')),
                    chosen=item.get('chosen', item.get('preferred', '')),
                    rejected=item.get('rejected', item.get('dispreferred', '')),
                    source=item.get('source', 'unknown'),
                )
        return dataset
    
    def save_jsonl(self, path: str):
        """保存为 JSONL 文件"""
        import json
        
        with open(path, 'w') as f:
            for item in self.data:
                f.write(json.dumps(item) + '\n')
