import json
import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig


class EvalDataset(Dataset):
    """
    用于评估的数据集类
    从JSON文件加载评测数据，包含question和ground_truth_answer
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        question_key: str = "question",
        answer_key: str = "ground_truth_answer",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: JSON文件路径
            tokenizer: tokenizer实例（用于tokenize prompt）
            question_key: 问题字段的key
            answer_key: 答案字段的key
            max_samples: 最大样本数（用于快速测试，None表示使用全部数据）
        """
        self.tokenizer = tokenizer
        self.question_key = question_key
        self.answer_key = answer_key
        
        # 加载数据
        self.data = self._load_data(data_path, max_samples)
        
    def _load_data(self, data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """从JSON文件加载数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        返回一个样本
        
        Returns:
            {
                "question": str,
                "ground_truth_answer": str,
                "index": int,  # 样本索引，用于追踪
            }
        """
        sample = self.data[idx]
        
        question = sample.get(self.question_key, "")
        answer = sample.get(self.answer_key, "")
        
        return {
            "question": question,
            "ground_truth_answer": answer,
            "index": idx,
        }


def get_eval_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    question_key: str = "question",
    answer_key: str = "ground_truth_answer",
    batch_size: int = 1,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
) -> DataLoader:
    """
    创建用于评估的DataLoader
    
    Args:
        data_path: JSON文件路径
        tokenizer: tokenizer实例
        question_key: 问题字段的key
        answer_key: 答案字段的key
        batch_size: batch大小（评估时通常为1）
        num_workers: DataLoader的worker数量
        max_samples: 最大样本数（用于快速测试）
        shuffle: 是否shuffle
        
    Returns:
        eval_dataloader: 评估DataLoader
    """
    dataset = EvalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        question_key=question_key,
        answer_key=answer_key,
        max_samples=max_samples,
    )
    
    def collate_fn(batch):
        """简单的collate函数，直接返回batch"""
        return batch
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    
    return dataloader

