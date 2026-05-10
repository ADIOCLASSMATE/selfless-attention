import os
import accelerate
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from omegaconf import OmegaConf
from utils.utils import get_full_attention_mask, load_model_tokenizer
from utils.diffusion_utils import DiffusionLanguage
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("dllm")
class DLLMEvalHarness(LM):
    def __init__(
        self,
        config_path,
        batch_size=1,
        device="cuda",
    ):
        '''
        Args:
            config_path: Path to the model config.
            device: 'cuda' or 'cpu'
        '''
        super().__init__()
        self.eval_config = OmegaConf.load(config_path)

        # 初始化 Accelerator
        self.accelerator = accelerate.Accelerator()
        
        # 加载模型和 Tokenizer
        self.model, self.tokenizer = load_model_tokenizer(config=self.eval_config)
        self.model.eval()
        self.diff_lm = DiffusionLanguage(mask_token_id=self.model.config.mask_token_id, config=self.eval_config)

        self.device = self.accelerator.device
        self.model = self.accelerator.prepare(self.model)
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self.mc_num = self.eval_config.mc_num
        self.batch_size = self.eval_config.batch_size
        assert self.mc_num % self.batch_size == 0

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @torch.no_grad()
    def get_logits(self, input_ids, prompt_length):
        input_ids_masked, masked_indices, t_sample, _ = self.diff_lm.forward_process(input_ids, prompt_length=prompt_length)
        t_sample = t_sample.to(self.accelerator.device)
        B, L = input_ids_masked.shape
        
        attention_mask = get_full_attention_mask(L-1)
        logits = self.model(input_ids=input_ids_masked[:, :-1], attention_mask=attention_mask).logits  # shape: [B, L, V]
        
        return logits, masked_indices[:, 1:], t_sample[:, 1:]

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def loss_function(self, logits, labels, ignore_index, reduction='mean'):
        return F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    
    @torch.no_grad()
    def get_loglikelihood(self, input_ids, labels, prompt_length):
        logits, masked_indices, t_sample = self.get_logits(input_ids=input_ids, prompt_length=prompt_length)

        loss = self.loss_function(
            logits[masked_indices],
            labels[masked_indices],
            ignore_index=-100,
            reduction='none',
        )

        loss = loss / t_sample[masked_indices]
        loss = loss.sum() / input_ids.shape[0]
        
        return -loss.item()

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        raise NotImplementedError

    def _encode(self, context):
        context_enc = self.tokenizer(context, add_special_tokens=False)["input_ids"]
                
        return context_enc
    
    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
        context_enc = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        
        context_enc_len = len(context_enc)

        continuation_enc = whole_enc[context_enc_len:]
        
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            # if self.accelerator.is_main_process:
            #     print(f"prefix: {e['prefix']}")
            #     print(f"target: {e['target']}")
            prefix_ids, target_ids = self._encode_pair(e["prefix"], e["target"])
            input_ids = prefix_ids + target_ids
            labels = [-100] * len(prefix_ids) + target_ids
            p_len = len(prefix_ids) 

            if len(input_ids) > self.eval_config.max_len:
                input_ids = input_ids[-self.eval_config.max_len:]
                labels = labels[-self.eval_config.max_len:]
                p_len = min(p_len, self.eval_config.max_len)

            input_ids = torch.tensor(input_ids, dtype=torch.long)[None, :].repeat((self.batch_size, 1))
            labels = torch.tensor(labels, dtype=torch.long)[None, :].repeat((self.batch_size, 1))
            prompt_length = torch.tensor([p_len], dtype=torch.long).repeat(self.batch_size) # (B)
            return {
                "input_ids": input_ids,
                "labels": labels[:, 1:].contiguous(),
                "prompt_length": prompt_length
            }

        raw_data = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        
        ds = Dataset.from_list(raw_data)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        my_results = []
        
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                input_ids = elem["input_ids"].to(self.device)
                labels = elem["labels"].to(self.device)
                prompt_length = elem["prompt_length"].to(self.device)
                ll_list = []
                for _ in range(self.mc_num // self.batch_size):
                    ll = self.get_loglikelihood(input_ids, labels, prompt_length)
                    ll_list.append(ll)

                ll_mean = np.mean(ll_list)
                my_results.append((ll_mean, 0.0))
                
        return my_results

    def loglikelihood_rolling(self, requests):
        def _tokenize(e):
            input_ids = self._encode(e["text"])
            input_ids = torch.tensor(input_ids, dtype=torch.long)[None, :].repeat((self.batch_size, 1))
            return {
                "input_ids": input_ids,
            }

        raw_data = [{"text": req.args[0]} for req in requests]
        
        ds = Dataset.from_list(raw_data)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        my_results = []
        
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing rolling likelihood..."):
                input_ids = elem["input_ids"]
                seq_len = input_ids.size(-1)
                max_len = self.eval_config.max_len
                stride = max_len // 2
                
                total_ll = 0.0
                
                for start_loc in range(0, seq_len, stride):
                    end_loc = min(start_loc + max_len, seq_len)
                    
                    input_ids_window = input_ids[:, start_loc:end_loc].to(self.device)
                    
                    labels_window = input_ids_window.clone()
                    
                    if start_loc > 0:
                        context_len = stride
                        
                        if context_len < labels_window.size(1):
                            labels_window[:, :context_len] = -100
                            prompt_length = torch.tensor([context_len], dtype=torch.long).repeat(self.batch_size).to(self.device)
                        else:
                            continue
                          
                    else:
                        prompt_length = torch.tensor([0], dtype=torch.long).repeat(self.batch_size).to(self.device)

                    labels_window = labels_window[:, 1:].contiguous()
                    ll_list = []
                    for _ in range(self.mc_num // self.batch_size):
                        ll = self.get_loglikelihood(input_ids_window, labels_window, prompt_length)
                        ll_list.append(ll)
                        
                    ll_mean = np.mean(ll_list)
                    
                    total_ll += ll_mean

                my_results.append(total_ll)
                
        return my_results

    def generate_until(self, requests: list[Instance]):
        raise NotImplementedError

if __name__ == "__main__":
    set_seed(42)
    cli_evaluate()