import sys
import os
import json
import multiprocessing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_DATASETS_OFFLINE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "true"
os.environ["HF_OFFLINE"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "true"
import math
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.modeling_model.modeling_dream import Qwen3ForCausalLM as DreamLM
from models.modeling_model.modeling_mad import Qwen3ForCausalLM as MADLM
from models.modeling_model.modeling_llada import Qwen3ForCausalLM as LLaDA
from utils.diffusion_utils import DiffusionLanguage
from utils.utils import get_AR_attention_mask, get_diffusion_attention_mask, get_full_attention_mask


class BatchRepeatCollator:
    """将单个样本重复N次形成batch，用于对Diffusion Loss进行蒙特卡洛估计"""
    def __init__(self, repeat_times=128):
        self.repeat_times = repeat_times
    
    def __call__(self, batch):
        # 强制要求 dataloader 的 batch_size=1
        if len(batch) != 1:
            raise ValueError(f"Collator expects batch size of 1, but got {len(batch)}")
        
        sample = batch[0]
        # 将单个样本重复 repeat_times 次
        repeated_batch = {
            k: [v] * self.repeat_times for k, v in sample.items()
        }
        
        # 转换为 torch tensor
        result = {}
        for k, v in repeated_batch.items():
            if isinstance(v[0], list):
                result[k] = torch.tensor(v, dtype=torch.long)
            else:
                result[k] = torch.tensor(v)
        
        return result


def build_packed_dataset(
    tokenizer,
    config,
    split="train",
    num_proc=None,
    cache_dir=None,
    text_column_name="text"
):
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count() - 2)
        
    data_path = config.dataset.params.data_path

    print(f"Loading dataset: {data_path}...")
    if "wikitext" in data_path.lower():
        raw_datasets = load_dataset(
            "wikitext",
            name=data_path,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    elif "openwebtext" in data_path.lower():
        raw_datasets = load_dataset(
            data_path,
            name=config.dataset.params.get("data_name", None),
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        raw_datasets = raw_datasets.take(10000)
    else:
        raw_datasets = load_dataset(
            data_path,
            name=config.dataset.params.get("data_name", None),
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples[text_column_name], 
            truncation=True,
            max_length=2049,
            padding=False,
            return_special_tokens_mask=False
        )
        
        eos_id = tokenizer.eos_token_id
            
        input_ids_list = [ids + [eos_id] for ids in tokenized_output["input_ids"]]
        attention_mask_list = [mask + [1] for mask in tokenized_output["attention_mask"]]
        
        labels_list = [ids[1:] for ids in input_ids_list]
        
        # 更新字典
        tokenized_output["input_ids"] = input_ids_list
        tokenized_output["attention_mask"] = attention_mask_list
        tokenized_output["labels"] = labels_list
        
        return tokenized_output

    print("Tokenizing data...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer",
    )
    
    return tokenized_datasets


if __name__ == "__main__":
    accelerator = Accelerator()
    
    evaluation_results = {}
    
    model_path_dict = {
        # "llada": "output/llada-fwb-edu/hf_model-final",
        # "qwen_base": "public/models/Qwen/Qwen3-0.6B-Base",
        # "qwen": "public/models/Qwen/Qwen3-0.6B",
        # "dream_eos": "output/dream-fwb-edu-eos/hf_model-final",
        # "dream_base": "output/dream-fwb-edu-base/hf_model-final",
        # "dream_scratch": "output/dream-fwb-edu-from_scratch/hf_model-final",
        "mad_onM": "output_final/mad-onM/hf_model-final",
        # "mad_eos": "output/mad-fwb-edu-eos/hf_model-final",
        # "mad_base": "output/mad-fwb-edu-base/hf_model-final",
        # "mad_scratch": "output/mad-fwb-edu-from_scratch/hf_model-final",
        # "mad_scratch_old": "output/mad-fwb-edu-from_scratch_old/hf_model-final",
    }
    
    dataset_list = ["EleutherAI/lambada_openai", "wikitext-103-raw-v1", "wikitext-2-raw-v1", "openwebtext",]
        
    for model_name, model_path in model_path_dict.items():
        if "dream" in model_name.lower():
            modelclass = DreamLM
        elif "mad" in model_name.lower():
            modelclass = MADLM
        elif "llada" in model_name.lower():
            modelclass = LLaDA
        elif "qwen" in model_name.lower():
            modelclass = AutoModelForCausalLM
        else:
            continue

        # Load Model & Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
            model = modelclass.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
        model.config.use_flex_attention = False
            # Load Dataset
            # 注意：cache_dir 最好指定绝对路径或确保存在
        for dataset_name in dataset_list:
            if dataset_name not in evaluation_results:
                evaluation_results[dataset_name] = {}

            # 【修复1】添加 training 配置，并设置 batch_size 为 1
            config = OmegaConf.create(
                {
                    "dataset": {
                        "params": {"data_path": dataset_name},
                    },
                    "model": {"attention_task": "random"},
                    "training": {"batch_size": 1} # 必须为1，因为 Collator 会做 expansion
                }
            )
            
            dataset = build_packed_dataset(tokenizer=tokenizer, config=config, split="test", cache_dir="public/.cache/huggingface/datasets")
            
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=1,
                collate_fn=BatchRepeatCollator(repeat_times=64), # 在这里扩充 batch
                num_workers=8,
                pin_memory=True,
            )
            
            # 【修复2】Prepare dataloader
            model, dataloader = accelerator.prepare(model, dataloader)
            model.eval()
            unwrapped_model = accelerator.unwrap_model(model)
            if "qwen" not in model_name.lower():
                diff_lm = DiffusionLanguage(mask_token_id=unwrapped_model.config.mask_token_id, config=config)
            local_total_loss_diff = torch.tensor(0.0, device=accelerator.device)
            local_total_loss_ar = torch.tensor(0.0, device=accelerator.device)
            local_total_count = torch.tensor(0.0, device=accelerator.device)
            
            accelerator.print(f"Evaluating Dataset: {dataset_name} | Model: {model_name}")
            
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process):
                with torch.no_grad():
                    text_ids = batch["input_ids"] 
                    if text_ids.shape[-1] <= 32:
                        continue
                    label_ids = batch["labels"]
                    
                    current_batch_size = text_ids.size(0)

                    # Diffusion 前向加噪过程
                    if "qwen" not in model_name.lower():
                        input_ids_masked, masked_indices, t_sample, v_sample = diff_lm.forward_process(text_ids)
                    
                        v_sample = torch.where(masked_indices, 0, v_sample).to(accelerator.device)
                        input_ids_masked = input_ids_masked.to(accelerator.device)
                        masked_indices = masked_indices.to(accelerator.device)
                        t_sample = t_sample.to(accelerator.device)
                        
                        B, L = input_ids_masked.shape

                    # --- Diffusion Loss ---
                    if "mad" in model_name.lower():
                        diffusion_attention_mask = get_diffusion_attention_mask(v_sample=v_sample, seq_len=L-1, device=accelerator.device)
                        
                        loss_diff = unwrapped_model.forward_process_mask(
                            input_ids=input_ids_masked[:, :-1],
                            labels=label_ids,
                            attention_mask=diffusion_attention_mask,
                            p_mask=t_sample[:, 1:],
                            masked_indices=masked_indices[:, 1:],
                        )
                        
                        # --- AR Loss (Standard PPL) ---
                        ar_attention_mask = get_AR_attention_mask(seq_len=L-1, device=accelerator.device)
        
                        loss_ar = unwrapped_model.forward_process(
                            input_ids=text_ids[:, :-1],
                            labels=label_ids,
                            attention_mask=ar_attention_mask,
                        )
                    elif "dream" in model_name.lower():
                        diffusion_attention_mask = get_full_attention_mask(seq_len=L-1, device=accelerator.device)
                        
                        loss_diff = unwrapped_model.forward_process(
                            input_ids=input_ids_masked[:, :-1],
                            labels=label_ids,
                            attention_mask=diffusion_attention_mask,
                            p_mask=t_sample[:, 1:],
                            masked_indices=masked_indices[:, 1:],
                        )
                        # --- AR Loss (Standard PPL) ---
                        ar_attention_mask = get_AR_attention_mask(seq_len=L-1, device=accelerator.device)
        
                        loss_ar = unwrapped_model.forward_process_ar(
                            input_ids=text_ids[:, :-1],
                            labels=label_ids,
                            attention_mask=ar_attention_mask,
                        )
                    elif "llada" in model_name.lower():
                        diffusion_attention_mask = get_full_attention_mask(seq_len=L-1, device=accelerator.device)
                        
                        loss_diff = unwrapped_model.forward_process(
                            input_ids=input_ids_masked[:, :-1],
                            labels=text_ids[:, :-1],
                            attention_mask=diffusion_attention_mask,
                            p_mask=t_sample[:, :-1],
                            masked_indices=masked_indices[:, :-1],
                        )
        
                        loss_ar = torch.tensor(2)
                    elif "qwen" in model_name.lower():
                        loss_diff = torch.tensor(2)
        
                        loss_ar = unwrapped_model(text_ids[:, :-1], labels=label_ids).loss
                        
                    else:
                        raise ValueError
                    
                    local_total_loss_diff += loss_diff.detach() * current_batch_size
                    local_total_loss_ar += loss_ar.detach() * current_batch_size
                    local_total_count += current_batch_size

            # Gather results
            accelerator.wait_for_everyone()
            
            global_count = accelerator.reduce(local_total_count, reduction="sum")
            global_loss_diff = accelerator.reduce(local_total_loss_diff, reduction="sum")
            global_loss_ar = accelerator.reduce(local_total_loss_ar, reduction="sum")
            
            # 只在主进程计算和打印
            if accelerator.is_main_process:
                avg_loss_diff = (global_loss_diff / global_count).item()
                ppl_diff = math.exp(avg_loss_diff)
                
                avg_loss_ar = (global_loss_ar / global_count).item()
                ppl_ar = math.exp(avg_loss_ar)
                
                print(f">>> Result {dataset_name} | {model_name}: PPL_Diff={ppl_diff:.4f}, PPL_AR={ppl_ar:.4f}")
                
                evaluation_results[dataset_name] = {
                    "loss_diff": avg_loss_diff,
                    "ppl_diff": ppl_diff,
                    "loss_ar": avg_loss_ar,
                    "ppl_ar": ppl_ar
                }
        if accelerator.is_main_process:
            with open(f"output_eval_final/PPL/{model_name}_evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, indent=4)
            
            torch.cuda.empty_cache()