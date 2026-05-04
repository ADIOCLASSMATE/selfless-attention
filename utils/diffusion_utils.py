import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer
)
from typing import List, Union
import sys


class DiffusionLanguage():
    def __init__(self, mask_token_id, config) -> None:
        self.mask_token_id = mask_token_id # 126336 is used for [MASK] token
        self.task = config.model.attention_task
        self.config = config
        # self.ar_ratio = config.training.ar_ratio
    
    @torch.no_grad()
    def forward_process(self, text_ids, attention_mask=None, eps=1e-3, t_in=None, step_ratio=None, prompt_length=None):
        if self.task == "random":
            b, l = text_ids.shape
            t = torch.rand(b, device=text_ids.device) if t_in is None else t_in

            t_sample = (1 - eps) * t + eps
            t_sample = t_sample[:, None].repeat(1, l) # (B, 1)
            
            # if torch.rand(1) < self.ar_ratio:
            #     # 退化为autoregressive
            #     random_start = ((1 - t_sample[:, 0]) * l).long()  # (B,)
            #     arange = torch.arange(l, device=text_ids.device).unsqueeze(0).expand(b, l)  # (B, L)
                
            #     normalized_index = arange.float() / (random_start[:, None].float() + 1e-6)
            #     decayed_value = 1.1 - normalized_index.clamp(max=1.0) * 1.0 
            #     v_sample = torch.where(arange < random_start[:, None], decayed_value, 0.0)
            # else:
            v_sample = torch.rand(b, l, device=text_ids.device) # (B, L)
            # 确保第一个 token 不被 mask
            v_sample[:, 0] = 2

            if prompt_length is not None:
                # 确保 prompt_length 是 (B,) 形状
                if prompt_length.dim() > 1:
                    prompt_length = prompt_length.squeeze(-1) 
 
                v_sample = self.prompt_process(text_ids, v_sample, prompt_length, pad_ids=None)
            
            masked_indices = v_sample < t_sample
              
            noisy_ids = torch.where(masked_indices, self.mask_token_id, text_ids) # (B, L)

            return noisy_ids, masked_indices, t_sample, v_sample
        
        else:
            raise ValueError(f"Wrong task: {self.task}")

    @torch.no_grad()
    def sample_v(self, text_ids, attention_mask=None, eps=1e-3, pad_ids=None, t_in=None, step_ratio=None, prompt_lengths=None):
        if self.task == "random":
            b, l = text_ids.shape
            t = torch.rand(b, device=text_ids.device) if t_in is None else t_in

            t_sample = (1 - eps) * t + eps
            t_sample = t_sample[:, None].repeat(1, l) # (B, 1)
            
            v_sample = torch.rand(b, l, device=text_ids.device) # (B, L)
            # 确保第一个 token 不被 mask
            v_sample[:, 0] = 2
        elif self.task == "ar":
            # ar: autoregressive, 对v_sample进行自回归处理
            # 构建严格降序的v_sample，用于自回归建模
            # 前面的token值大（先处理），后面的token值小（后处理）
            b, l = text_ids.shape
            
            # t_sample直接置零
            t_sample = torch.zeros(b, l, device=text_ids.device)  # (B, L)
            
            # 创建位置索引 [0, 1, 2, ..., l-1]
            pos_idx = torch.arange(l, device=text_ids.device, dtype=torch.float32).unsqueeze(0)  # (1, L)
            # 构建严格降序的v_sample: 从接近1的值递减到接近0的值（都在0到1之间）
            # 第一个token的v值最大（接近1），最后一个token的v值最小（接近0）
            # 使用线性插值：从 1-eps 递减到 eps
            if l > 1:
                # 从 1-eps 线性递减到 eps
                v_sample = 1 - eps - (1 - 2 * eps) * pos_idx / (l - 1)  # (1, L)
            else:
                # 当l=1时，只有一个token，设为1-eps
                v_sample = torch.ones(1, 1, device=text_ids.device) * (1 - eps)
            v_sample = v_sample.expand(b, l)  # (B, L)

        else:
            raise ValueError(f"Wrong task: {self.task}")
        
        if prompt_lengths is not None:
            # 确保 prompt_length 是 (B,) 形状
            if prompt_lengths.dim() > 1:
                prompt_lengths = prompt_lengths.squeeze(-1)
            
            v_sample = self.prompt_process(text_ids, v_sample, prompt_lengths, pad_ids)
            
            
        return t_sample, v_sample
        
    @torch.no_grad()
    def prompt_process(self, text_ids, v_sample, prompt_lengths, pad_ids):
        prompt_task = getattr(self.config.training, 'prompt_task', None)
        if prompt_task == "ar":
            # prompt部分的v_sample值从大到小递减，最小值为2，意思为prompt部分自回归生成
            # 
            # 示例：假设 prompt_length=10, solution_length=5, L=15
            # 序列结构：[prompt(0-9) | solution(10-14) | padding(15+)]
            # 对于prompt部分，我们希望v_sample = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]（降序，最小为2）
            # solution部分保持随机采样值[0,1]
            # padding部分设为-1
            #
            B, L = text_ids.shape
            
            # 创建位置索引：序列中每个位置的绝对索引 [0, 1, 2, ..., L-1]
            pos_idx = torch.arange(L, device=text_ids.device, dtype=torch.long).unsqueeze(0)  # [1, L]
            # prompt_mask: 标识哪些位置属于prompt部分（位置 < prompt_length）
            prompt_mask = pos_idx < prompt_lengths.unsqueeze(1)  # [B, L]
            # 计算prompt部分每个位置相对于prompt结束位置的偏移（从后往前）
            # 对于prompt_length=10，位置[0,1,2,...,9]的offset=[9,8,7,...,0]
            prompt_offset = pos_idx  # [1, L]，在prompt内是[0, 1, 2, ..., prompt_length-1]
            # 计算降序索引：从prompt_length-1递减到0
            # 对于prompt_length=10，reverse_idx = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            prompt_reverse_idx = prompt_lengths.unsqueeze(1) - 1 - prompt_offset  # [B, L]
            # 计算最终的v_sample值：2 + reverse_idx，最小值为2（当offset=prompt_length-1时）
            # 对于prompt_length=10，values = 2 + [9,8,7,6,5,4,3,2,1,0] = [11,10,9,8,7,6,5,4,3,2]
            prompt_values = 2 + prompt_reverse_idx  # [B, L]
            # 只在prompt部分应用新值，solution部分保持原样（随机采样值[0,1]）
            v_sample = torch.where(prompt_mask, prompt_values, v_sample)
            # 将padding部分设为-1
            if pad_ids is not None:
                v_sample.masked_fill_(text_ids == pad_ids, -1)
        elif prompt_task == "random":
            # prompt部分的v_sample值随机采样，最小值为2，意思为prompt部分采用随机顺序生成，按照非自回归方式生成
            B, L = text_ids.shape
            
            pos_idx = torch.arange(L, device=text_ids.device, dtype=torch.long).unsqueeze(0)  # [1, L]
            prompt_mask = pos_idx < prompt_lengths.unsqueeze(1)  # [B, L]
            # 对prompt部分的v_sample值加2，使其范围从[0,1]变为[2,3]
            v_sample = torch.where(prompt_mask, v_sample + 2, v_sample)
            # 将padding部分设为-1
            if pad_ids is not None:
                v_sample.masked_fill_(text_ids == pad_ids, -1)
            
        else:
            raise ValueError(f"UNKNOWN PROMPT TASK, get {prompt_task}")
        
        return v_sample