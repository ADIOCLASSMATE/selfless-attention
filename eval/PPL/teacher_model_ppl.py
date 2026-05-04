import json
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 假设你的生成函数保存在这些路径下
# 请确保生成函数的签名大致为: generate(model, prompt_ids, gen_length, num_response, ...)
from models.modeling_model.modeling_dream import Qwen3ForCausalLM as DreamLM
from models.modeling_model.modeling_mad import Qwen3ForCausalLM as MADLM
from datasets import load_dataset

def calculate_batch_ppl(teacher_model, seqs, prompt_len, tokenizer):
    """
    计算 Batch PPL
    seqs: [batch_size (4), total_len (256)]
    prompt_len: int, prompt 的长度，用于 mask
    """
    # 1. 创建 Labels
    labels = seqs.clone()
    
    # 2. Mask Prompt 部分 (我们只关心生成的 PPL)
    labels[:, :prompt_len] = -100
    
    # 3. Mask EOS 及其之后的部分
    # 遍历 batch 中的每一行
    for i in range(labels.shape[0]):
        # 找到该序列中所有的 EOS token 位置
        eos_indices = (seqs[i] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            # 找到第一个 EOS 的位置
            first_eos_idx = eos_indices[0].item()
            # 根据需求：将 EOS token 及其后面的 token 设置为 -100
            labels[i, first_eos_idx:] = -100
    
    # 4. 计算 Loss (Reduction=None 以获得每个样本的 loss)
    with torch.no_grad():
        # Teacher model forward
        outputs = teacher_model(seqs, labels=labels)
        
        # 获取 logits 并进行 shift
        # logits: [B, L, V] -> [B, L-1, V]
        logits = outputs.logits[:, :-1, :].contiguous()
        # labels: [B, L] -> [B, L-1]
        shift_labels = labels[:, 1:].contiguous()
        
        # 手动计算 CrossEntropy 以处理 reduction='none' 和 mask
        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
        
        # Flatten 用于计算 loss
        flat_logits = logits.view(-1, teacher_model.config.vocab_size)
        flat_labels = shift_labels.view(-1)
        
        loss = loss_fct(flat_logits, flat_labels)
        
        # Reshape 回 [B, L-1]
        loss = loss.view(seqs.shape[0], -1)
        
        # 计算每个样本的平均 loss (只计算非 -100 的位置)
        # mask: [B, L-1], True where label != -100
        active_mask = (shift_labels != -100).float()
        
        # 每个样本的有效 token 数量
        num_active_tokens = active_mask.sum(dim=1)
        
        # 每个样本的总 loss
        sum_loss = (loss * active_mask).sum(dim=1)
        
        # 避免除以 0 (如果生成全是 -100, loss 设为 nan 或 0)
        mean_loss = sum_loss / num_active_tokens.clamp(min=1e-9)
        
        # PPL = exp(mean_loss)
        ppls = torch.exp(mean_loss)
        
        # 如果没有有效 token，PPL 设为 None 或 -1
        ppls = torch.where(num_active_tokens > 0, ppls, torch.tensor(float('nan'), device=ppls.device))
        
    return ppls.tolist()

if __name__ == "__main__":
    # 配置
    PROMPT_LEN = 64
    GEN_LEN = 64
    NUM_RESPONSE = 16
    NUM_SAMPLES = 100
    BATCH_SIZE = NUM_RESPONSE # 这里指一个 prompt 生成多少个 response
    
    model_path_dict = {
        "mad": "output/mad-fwb-edu-base/hf_model-final",
        "dream": "output/dream-fwb-edu-base/hf_model-final",
    }
    
    print("Loading Teacher Model...")
    # 注意：这里假设 Student 和 Teacher 共享相同的 Tokenizer/Vocabulary
    # 如果不一样，需要将 seqs decode 成 text 再用 teacher tokenizer encode
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "public/models/Qwen/Qwen3-32B", 
        trust_remote_code=True, 
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" # 建议开启加速
    ).to("cuda")
    teacher_model.eval()
    
    print("Loading Dataset...")
    # 使用 streaming 模式避免下载整个数据集，或者取 split
    dataset = load_dataset(
                "cimec/lambada",
                split="validation",
                cache_dir="public/.cache/huggingface/datasets",
                streaming=False
            )
    print("Sampling fixed evaluation prompts...")
    
    tokenizer = AutoTokenizer.from_pretrained("output/mad-fwb-edu-base/hf_model-final", trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    fixed_prompts = []
    dataset_iter = iter(dataset)
    while len(fixed_prompts) < NUM_SAMPLES:
        try:
            sample = next(dataset_iter)
            text = sample["text"]

            input_ids = tokenizer(
                text,
                return_tensors="pt",
                truncation=False
            ).input_ids

            if input_ids.shape[1] < PROMPT_LEN:
                continue

            prompt_ids = input_ids[:, :PROMPT_LEN]
            prompt_text = tokenizer.decode(
                prompt_ids[0],
                skip_special_tokens=True
            )

            fixed_prompts.append({
                "prompt_ids": prompt_ids,   # CPU tensor
                "prompt_text": prompt_text
            })

        except StopIteration:
            break

    print(f"Collected {len(fixed_prompts)} fixed prompts.")

    
    results = []

    for model_name, model_path in model_path_dict.items():
        print(f"\nEvaluating Model: {model_name}")
        
        # 1. Load Student Model
        if model_name == "dream":
            modelclass = DreamLM
            # 假设 genfunc 签名: genfunc(model, prompt_ids, gen_length, num_response, ...)
        elif model_name == "mad":
            modelclass = MADLM
        else:
            raise ValueError
            
        model = modelclass.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
        try:
            model.config.use_flex_attention = True
        except:
            pass
        model.eval()
        
            
        # 2. Iterate Dataset
        dataset_iter = iter(dataset)
        data_count = 0
        
        pbar = tqdm(total=len(fixed_prompts), desc=f"{model_name} processing")

        for data_count, sample in enumerate(fixed_prompts):
            try:
                prompt_ids = sample["prompt_ids"].to("cuda")
                prompt_text = sample["prompt_text"]

                with torch.no_grad():
                    generated_ids = model.generate(
                        prompt_ids=prompt_ids,
                        gen_length=GEN_LEN,
                        num_response=NUM_RESPONSE,
                        block_size=8,
                        temperature=1.0,
                        ratio=1.0,
                    )

                ppls = calculate_batch_ppl(
                    teacher_model,
                    generated_ids,
                    PROMPT_LEN,
                    tokenizer
                )

                valid_ppls = [p for p in ppls if not pd.isna(p)]
                avg_ppl = sum(valid_ppls) / len(valid_ppls) if len(valid_ppls) > 0 else None

                responses_text = tokenizer.batch_decode(
                    generated_ids[:, PROMPT_LEN:],
                    skip_special_tokens=True
                )

                record = {
                    "model": model_name,
                    "sample_id": data_count,
                    "prompt": prompt_text,
                    "avg_ppl": avg_ppl,
                }

                for idx, (resp, ppl) in enumerate(zip(responses_text, ppls)):
                    record[f"response_{idx}"] = resp
                    record[f"ppl_{idx}"] = ppl

                results.append(record)
                pbar.update(1)

            except Exception as e:
                print(f"[{model_name}] Error on sample {data_count}: {e}")
                continue

        pbar.close()
        
        torch.cuda.empty_cache()

    # 6. Save to JSON
    output_file = "model_evaluation_ppl.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {output_file}")

    # 打印简要统计
    if len(results) > 0:
        df = pd.DataFrame(results)
        print("\nSummary Statistics (Average PPL per model):")
        print(df.groupby("model")["avg_ppl"].mean())