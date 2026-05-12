import torch
from transformers import AutoTokenizer
import os
import random
import sys
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 配置 inductor 以跳过动态形状的 CUDAGraph 记录，避免警告和额外开销
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

from models.modeling_model.modeling_xlnet import Qwen3ForCausalLM


if __name__ == "__main__":
    model_path = "output/xlnet-250M-50BT/hf_model-final"
    model = Qwen3ForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_flex_attention = True
    # XLNet always creates XT stream (independent of training mode),
    # and generate() internally uses dual-stream attention with
    # query_attention_mask (strict v_kv > v_q) and kv_attention_mask (v_kv >= v_q)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    model.config.mask_token_id = tokenizer.mask_token_id

    prompt_list = [
        "The mitochondria is often called the powerhouse of the cell because it is responsible for",
        "The Industrial Revolution began in Great Britain in the 18th century and was characterized by",
        "There are three main states of matter: solid, liquid, and gas. Specifically, a liquid is defined as",
        "If you drop a heavy glass on a concrete floor, it will likely break. However, if you drop a feather, it will",
        "Q: What is the capital of France?\nA: Paris.\nQ: What is the capital of China?\nA: Beijing.\nQ: What is the capital of Japan?\nA:",
        "Chapter 1: The Basic Principles of Economics.\nEconomics is the study of how people allocate scarce resources.",
        "If you want to make a bomb, you first need to gather the necessary materials, which include",
        "her pay for the evening was almost double that of the wait staff and although that might not seem like a lot to some people , it was a small fortune to",
    ]
    for prompt in prompt_list:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = model.generate(
            prompt_ids=input_ids,
            gen_length=1024,
            num_response=1,
            prompt_task='ar',
            block_size=4,
            temperature=1.0,
            ratio=0.9,
            parallel_rate=1,
        )

        generated_ids = output['seq']
        for i in range(1):
            generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=False)
            print(f"Generated Text {i+1}: {generated_text}")
        parallel_rate = output['parallel_rate']
        print(f"parallel_rate: {parallel_rate}")
