import torch
from transformers import AutoTokenizer
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from models.modeling_model.modeling_dream import Qwen3ForCausalLM


if __name__ == "__main__":
    model_path = "output_final/dream-preload-80BT/hf_model-final"
    model = Qwen3ForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    model.config.mask_token_id = tokenizer.mask_token_id
    prompt_list = [
        "About Grand Slam Fishing Charters\nAs a family owned business we know how important it is that your trip becomes the best memory of your vacation, we are proud of our islands, our waters and our crew and we are desperate show you the best possible time during your stay. We can not guarantee fish every time but we can guarantee you a great time! The biggest perk of our job is seeing so many of our customers become close friends"
                   ]
    # prompt = "Once upon a time"
    for prompt in prompt_list:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = model.generate(
            prompt_ids=input_ids,
            gen_length=1024,
            num_response=8,
            block_size=4,
            temperature=1.0,
            ratio=0.9,
        )
        
        generated_ids = output['seq']
        for i in range(8):
            generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=False)
            print(f"Generated Text {i+1}: {generated_text}")
        parallel_rate = output['parallel_rate']
        print(f"parallel_rate: {parallel_rate}")