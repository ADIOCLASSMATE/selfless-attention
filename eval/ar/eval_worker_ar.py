import os
import accelerate
import torch
import sys
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from omegaconf import OmegaConf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import load_model_tokenizer
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
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.eval_config.model.model_path,
            dtype=torch.float16, 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.eval_config.model.model_path,
        )
        self.model.eval()

        self.device = self.accelerator.device
        self.model = self.accelerator.prepare(self.model)
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def loss_function(self, logits, labels, ignore_index, reduction='mean'):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index, reduction=reduction)
    
    @torch.no_grad()
    def get_loglikelihood(self, input_ids, labels):
        # attention_mask = (input_ids != self.tokenizer.pad_token_id)
        logits = self.model(input_ids).logits

        # loss = F.cross_entropy(
        #     logits[..., :-1, :].view(-1, logits.size(-1)),
        #     labels[..., 1:].view(-1),
        #     reduction="sum", 
        #     ignore_index=-100
        # )
        loss = self.loss_function(logits=logits[..., :-1, :], labels=labels[..., 1:], ignore_index=-100, reduction='sum')
        
        loss = loss / input_ids.size(0)
        
        return -loss.item()

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        raise NotImplementedError

    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
        context_enc = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        
        context_enc_len = len(context_enc)

        continuation_enc = whole_enc[context_enc_len:]
        
        return context_enc, continuation_enc
    
    def _encode(self, context):
        context_enc = self.tokenizer(context, add_special_tokens=False)["input_ids"]
                
        return context_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix_ids, target_ids = self._encode_pair(e["prefix"], e["target"])
            input_ids = prefix_ids + target_ids
            labels = [-100] * len(prefix_ids) + target_ids
            if len(input_ids) > self.eval_config.max_len:
                input_ids = input_ids[-self.eval_config.max_len:]
                labels = labels[-self.eval_config.max_len:]
            return {
                "input_ids": input_ids,
                "labels": labels,
            }

        raw_data = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        
        ds = Dataset.from_list(raw_data)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        my_results = []
        
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                input_ids = elem["input_ids"].unsqueeze(0).to(self.device)
                labels = elem["labels"].unsqueeze(0).to(self.device)

                ll = self.get_loglikelihood(input_ids, labels)

                my_results.append((ll, 0.0))
                
        return my_results

    def loglikelihood_rolling(self, requests):
        def _tokenize(e):
            input_ids = self._encode(e["text"])
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
                seq_len = input_ids.size(0)
                max_len = self.eval_config.max_len
                stride = max_len // 2
                
                total_ll = 0.0
                total_tokens = 0
                
                for start_loc in range(0, seq_len, stride):
                    end_loc = min(start_loc + max_len, seq_len)
                    trg_len = end_loc - start_loc
                    
                    input_ids_window = input_ids[start_loc:end_loc].unsqueeze(0).to(self.device)
                    
                    labels_window = input_ids_window.clone()
                    
                    if start_loc > 0:
                        context_len = stride
                        
                        if context_len < labels_window.size(1):
                            labels_window[:, :context_len] = -100
                        else:
                            labels_window[:, :] = -100

                    ll = self.get_loglikelihood(input_ids_window, labels_window)
                    
                    if start_loc == 0:
                        n_tokens = input_ids_window.size(1) - 1
                    else:
                        n_tokens = input_ids_window.size(1) - stride
                    
                    total_ll += ll
                    total_tokens += n_tokens

                my_results.append(total_ll)
                
        return my_results

    def generate_until(self, requests: list[Instance]):
        raise NotImplementedError

if __name__ == "__main__":
    set_seed(42)
    cli_evaluate()