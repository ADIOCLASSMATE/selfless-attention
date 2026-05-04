import multiprocessing
import itertools
import os
from datasets import load_dataset, DatasetDict
from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_packed_dataset(
    tokenizer,
    config,
    split="train",
    num_proc=None,
    cache_dir=None,
    text_column_name="text"
):
    """
    直接从 Hugging Face 加载数据集，分词并进行 Packing。
    """
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count() - 2)
        
    data_path = config.dataset.params.data_path
    max_length = config.dataset.preprocessing.max_seq_length

    # 1. 从 Hugging Face 加载数据集
    print(f"Loading dataset from Hugging Face: {data_path}...")
    raw_datasets = load_dataset(
        data_path,
        name=config.dataset.params.get("data_name", None), # 兼容有子集的数据集
        split=split,
        cache_dir=cache_dir,
        streaming=False
    )

    # 2. tokenize 并添加 EOS
    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples[text_column_name], 
            truncation=False, 
            padding=False,
            return_special_tokens_mask=False
        )
        
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            # 如果没有 EOS，尝试使用 pad 或者手动指定，但通常 LLM tokenizer 都有
            eos_id = tokenizer.add_special_tokens({'eos_token': '[EOS]'})
            
        # 给每个样本末尾添加 EOS token
        tokenized_output["input_ids"] = [ids + [eos_id] for ids in tokenized_output["input_ids"]]
        tokenized_output["attention_mask"] = [mask + [1] for mask in tokenized_output["attention_mask"]]
        
        return tokenized_output

    print("Tokenizing data...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer on dataset",
    )

    # 3. Packing
    def group_texts(examples):
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        
        total_length = len(concatenated_examples["input_ids"])
        
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=5000,
        num_proc=num_proc,
        desc=f"Grouping texts in chunks of {max_length}",
    )
    
    lm_datasets.set_format("torch")
    
    print(f"Dataset processed. Total samples: {len(lm_datasets)}")
    return lm_datasets

def get_dataloaders(
    config, 
    tokenizer,
    val_split_ratio=0.001, 
):
    """
    适配 accelerate 的 DataLoader 配置
    """
    dataset = build_packed_dataset(tokenizer=tokenizer, config=config, cache_dir="public/.cache/huggingface/datasets")
    num_workers = config.training.dataloader_workers
    split_dataset = dataset.train_test_split(test_size=val_split_ratio, seed=config.training.seed)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 为 DDP 训练设置 drop_last=True，这是适配 accelerate 的关键
    train_dataloader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=config.training.batch_size,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, 
    )

    val_dataloader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=config.training.batch_size,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    config = OmegaConf.load("configs/MAD/owt.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    
    
    train_dataloader, val_dataloader = get_dataloaders(
        config=config,
        tokenizer=tokenizer
    )
    for batch in tqdm(train_dataloader):
        
        print(f"ids: {batch['input_ids'][0]}")
        break