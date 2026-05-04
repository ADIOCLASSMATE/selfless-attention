"""
PPL evaluation for XLNet (Selfish baseline) model.
Adapted from test_PPL_selfless.py — uses get_xlnet_mask instead of get_selfless_mask.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import torch
import math
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.modeling_model.modeling_xlnet import Qwen3ForCausalLM as XLNetLM
from utils.diffusion_utils import DiffusionLanguage
from utils.utils import get_xlnet_mask, get_xlnet_mask_ar, get_AR_attention_mask, get_diffusion_attention_mask, get_full_attention_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to XLNet checkpoint")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--mc_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_path", type=str, default="./output_eval_final/xlnet-all")
    args = parser.parse_args()

    accelerator = Accelerator()
    
    # Load model
    model = XLNetLM.from_pretrained(args.model_path, trust_remote_code=True, dtype=torch.bfloat16)
    model.config.use_flex_attention = False
    model = model.to(accelerator.device)
    model.eval()
    
    diff_lm = DiffusionLanguage(mask_token_id=model.config.mask_token_id, config=None)
    
    from utils.dataset_eval import get_eval_dataloader
    dataloader = get_eval_dataloader(args.dataset, args.batch_size, args.max_seq_length)
    
    total_loss_diff = 0.0
    total_loss_ar = 0.0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Evaluating XLNet"):
        text_ids = batch["input_ids"]
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)
        text_ids = text_ids.to(accelerator.device)
        B, L = text_ids.shape
        
        # Diffusion (PLM) loss with XLNet mask
        t_sample, v_sample = diff_lm.sample_v(text_ids)
        v_sample = v_sample.to(accelerator.device)
        query_mask, kv_mask = get_xlnet_mask(v_sample=v_sample, seq_len=L, device=accelerator.device)
        
        with torch.no_grad():
            loss_diff = model.forward_process(
                X0_input_ids=text_ids,
                labels=text_ids,
                query_attention_mask=query_mask,
                kv_attention_mask=kv_mask,
            )
        total_loss_diff += loss_diff.item() * B
        
        # AR loss with XLNet AR mask
        query_mask_ar, kv_mask_ar = get_xlnet_mask_ar(seq_len=L, device=accelerator.device)
        with torch.no_grad():
            loss_ar = model.forward_process(
                X0_input_ids=text_ids,
                labels=text_ids,
                query_attention_mask=query_mask_ar,
                kv_attention_mask=kv_mask_ar,
            )
        total_loss_ar += loss_ar.item() * B
        total_tokens += B
    
    avg_loss_diff = total_loss_diff / total_tokens
    avg_loss_ar = total_loss_ar / total_tokens
    ppl_diff = math.exp(avg_loss_diff)
    ppl_ar = math.exp(avg_loss_ar)
    
    print(f"XLNet Diffusion PPL: {ppl_diff:.4f} (Loss: {avg_loss_diff:.4f})")
    print(f"XLNet AR PPL: {ppl_ar:.4f} (Loss: {avg_loss_ar:.4f})")
    
    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    results = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "ppl_diffusion": ppl_diff,
        "ppl_ar": ppl_ar,
        "loss_diffusion": avg_loss_diff,
        "loss_ar": avg_loss_ar,
    }
    import json
    with open(os.path.join(args.output_path, "xlnet_ppl_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
