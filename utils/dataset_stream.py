import torch
import glob
import random
import os
import traceback
from itertools import cycle, chain, islice
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

class StreamDataset(IterableDataset):
    def __init__(self,
                 config,
                 data_path,
                 tokenizer,
                 infinite=True, # True=训练模式(无限), False=验证模式(跑完即停)
                 ):
        super().__init__()
        self.config = config
        self.data_class = config.dataset.params.get("data_class", "parquet")
        self.max_length = config.dataset.preprocessing.max_seq_length
        self.buffer_size = config.dataset.params.shuffle_buffer_size
        self.seed = config.training.seed
        self.tokenizer = tokenizer
        self.infinite = infinite 

        # 1. 获取文件列表
        if isinstance(data_path, (list, tuple)):
            expanded_files = []
            for p in data_path:
                expanded_files.extend(glob.glob(p))
            self.all_files = sorted(expanded_files)
        else:
            self.all_files = sorted(glob.glob(data_path))

        if len(self.all_files) == 0:
            raise ValueError(f"No parquet files found in {data_path}")

        # 2. 获取分布式信息
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # 3. 混合分片策略 (Hybrid Sharding)
        # 如果文件数量少于 GPU 数量，或者只有一个文件 -> 开启样本级分片
        if len(self.all_files) < self.world_size:
            if self.rank == 0:
                print(f"[StreamDataset] Files ({len(self.all_files)}) < World Size ({self.world_size}). Switching to Sample-Level Sharding.")
            self.my_files = self.all_files # 所有卡都持有所有文件
            self.shard_by_sample = True    # 标记：需要进行样本过滤
        else:
            # 文件足够多 -> 维持文件级分片（IO效率更高）
            self.my_files = self.all_files[self.rank::self.world_size]
            self.shard_by_sample = False
        
        if len(self.my_files) == 0 and not self.shard_by_sample:
             print(f"[Warning] Rank {self.rank} has no files to process!")

    def _tokenize_text(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.tokenizer.eos_token_id)
        return ids

    def _file_generator(self):
        """
        生成器：遍历文件并输出文本内容
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank) 
        
        while True:
            # 1. 确定本轮文件的读取顺序
            if self.infinite:
                # 训练模式：打乱
                indices = torch.randperm(len(self.my_files), generator=g).tolist()
                current_files = [self.my_files[i] for i in indices]
            else:
                # 验证模式：顺序
                current_files = self.my_files

            # 2. 遍历文件
            for file_path in current_files:
                try:
                    # 建立流式读取
                    dataset = load_dataset(self.data_class, data_files=file_path, split="train", streaming=True)
                    if self.shard_by_sample:
                        dataset = islice(dataset, self.rank, None, self.world_size)
                    text_column = None
                    
                    for item in dataset:
                        # 第一次拿到数据时，确定 text 列名
                        if text_column is None:
                            cols = list(item.keys())
                            text_column = "text" if "text" in cols else cols[0]
                        
                        yield item[text_column]
                        
                except Exception as e:
                    print(f"[Error] Rank {self.rank} failed reading {file_path}: {e}")
                    # traceback.print_exc()
                    continue
            
            # 验证集跑完一轮退出
            if not self.infinite:
                break

    def __iter__(self):
        buffer = []
        token_stream = []
        
        text_iter = self._file_generator()

        for text in text_iter:
            # 1. Tokenize
            tokenized_ids = self._tokenize_text(text)
            if not tokenized_ids:
                continue

            # 2. 填充 Buffer
            if len(buffer) < self.buffer_size:
                buffer.append(tokenized_ids)
                continue
            
            # 3. Buffer 满了，随机取样
            idx = random.randint(0, len(buffer) - 1)
            sample = buffer[idx]
            buffer[idx] = tokenized_ids 
            
            token_stream.extend(sample)

            # 4. 切割 Block
            while len(token_stream) >= self.max_length:
                input_ids = token_stream[:self.max_length]
                token_stream = token_stream[self.max_length:]
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                }
        
        if buffer:
            random.shuffle(buffer)
            for sample in buffer:
                token_stream.extend(sample)
                while len(token_stream) >= self.max_length:
                    input_ids = token_stream[:self.max_length]
                    token_stream = token_stream[self.max_length:]
                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    }

    def get_dataloader(self):
        def collate_fn(batch):
            return {
                "input_ids": torch.stack([x["input_ids"] for x in batch]),
            }
        
        return DataLoader(
            self,
            batch_size=self.config.training.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )

def create_train_val_dataloaders(config, tokenizer):
    data_path = config.dataset.params.data_path
    # 1. 获取文件列表
    if isinstance(data_path, (list, tuple)):
        expanded_files = []
        for p in data_path:
            expanded_files.extend(glob.glob(p))
        all_files = sorted(expanded_files)
    else:
        all_files = sorted(glob.glob(data_path))

    if len(all_files) == 0:
        raise ValueError(f"No parquet files found in {data_path}")
    seed = config.training.get("seed", 42)
    rng = random.Random(seed) 
    rng.shuffle(all_files)
    train_data_path = all_files[:-1]
    val_data_path = all_files[-1:]
    
    train_dataset = StreamDataset(config=config, data_path=train_data_path, tokenizer=tokenizer, infinite=True)
    val_dataset = StreamDataset(config=config, data_path=val_data_path, tokenizer=tokenizer, infinite=False)
    train_dataloader = train_dataset.get_dataloader()
    val_dataloader = val_dataset.get_dataloader()
    
    return train_dataloader, val_dataloader
    
    
# ----------------------------------------------------
# 完整的测试代码
# ----------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import sys
    import numpy as np

    # ==========================================
    # 0. 配置与初始化
    # ==========================================
    # 模拟 Config
    config = OmegaConf.create({
        "model": {"model_path": "public/models/Qwen/Qwen3-0.6B"},
        "dataset": {
            "params": {
                # 请确保这里指向真实的 Parquet 文件路径
                "data_path": "public/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/data/*.parquet",
                "data_class": "parquet",
                "shuffle_buffer_size": 100 # 测试时设小一点，方便数据快速流出
            },
            "preprocessing": {
                "max_seq_length": 256 # 测试时设短一点
            }
        },
        "training": {
            "batch_size": 2,
            "seed": 42
        }
    })

    print(">>> Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    except:
        print("[Warning] Model path invalid, falling back to 'gpt2' for demonstration.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # 检查是否有文件
    files = glob.glob(config.dataset.params.data_path)
    if not files:
        print(f"[Error] No files found at {config.dataset.params.data_path}")
        print("Please modify 'config.dataset.params.data_path' to a valid location.")
        sys.exit(1)
    
    # 强制只使用第一个文件进行测试，以保证分布式测试的确定性
    test_file_path = [files[0]]
    print(f">>> Using file for testing: {test_file_path[0]}")

    # ==========================================
    # 测试 1: EOS Token 检查 & 文本解码质量
    # ==========================================
    print("\n" + "="*60)
    print(">>> Test 1: EOS Token Presence & Decoding Quality (Rank 0)")
    print("="*60)
    
    # 模拟单卡环境
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    dataset = StreamDataset(
        config=config,
        data_path=test_file_path,
        tokenizer=tokenizer,
        infinite=True
    )
    dataloader = dataset.get_dataloader()
    
    #以此变量存储获取到的 batch，用于后续分析
    first_batch = None

    for i, batch in enumerate(dataloader):
        first_batch = batch
        break # 只取一个 batch

    input_ids = first_batch["input_ids"] # shape: [batch_size, max_seq_length]
    print(f"[Info] Input Shape: {input_ids.shape}")

    # --- 1.1 EOS 检查 ---
    eos_id = tokenizer.eos_token_id
    has_eos = (input_ids == eos_id).any().item()
    
    if has_eos:
        print(f"✅ [PASS] EOS Token (ID: {eos_id}) found in the batch.")
        # 统计数量
        count = (input_ids == eos_id).sum().item()
        print(f"       -> Count of EOS tokens in this batch: {count}")
    else:
        print(f"⚠️ [WARNING] EOS Token not found in this batch. (This might be normal if max_length is small and sentences are long, but check logic)")

    # --- 1.2 文本解码检查 ---
    print("\n[Info] Decoding first sample in batch:")
    sample_ids = input_ids[0].tolist()
    text = tokenizer.decode(sample_ids)
    
    print("-" * 40)
    print(f"{text[:300]} ... [truncated]")
    print("-" * 40)
    
    if len(text) > 10 and " " in text:
        print("✅ [PASS] Decoded text looks like valid string.")
    else:
        print("❌ [FAIL] Decoded text looks suspicious (empty or garbage).")


    # ==========================================
    # 测试 2: 分布式数据互斥性 (Rank 0 vs Rank 1)
    # 目标：证明 Rank 0 和 Rank 1 在读取同一个文件时，获取的数据完全不重叠
    # ==========================================
    print("\n" + "="*60)
    print(">>> Test 2: Distributed Data Exclusivity (Rank 0 vs Rank 1)")
    print("="*60)

    # 定义一个辅助函数来收集数据哈希
    def collect_data_fingerprints(rank, world_size, num_batches=20):
        # 模拟环境变量
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        # 实例化 Dataset (必须使用 infinite=False 且 shard_by_sample=True 的场景来测试 islice)
        # 注意：这里我们使用相同的 seed
        ds = StreamDataset(
            config=config,
            data_path=test_file_path, # 同一个文件
            tokenizer=tokenizer,
            infinite=False # 验证模式，确保顺序性
        )
        
        # 收集数据特征 (这里我们收集前几个 Token 的序列作为指纹)
        fingerprints = set()
        loader = ds.get_dataloader()
        
        print(f"    -> Running Rank {rank}...")
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            
            # 将每个样本的前 10 个 token 转为 tuple 作为指纹
            # 为什么只取前几个？因为 buffer shuffle 可能会打乱 batch 内部顺序，
            # 但每个独立的句子片段内容应该是唯一的。
            batch_ids = batch["input_ids"]
            for sample in batch_ids:
                # 转为 tuple 放入集合
                # 注意：如果 max_length 很短，可能切断了句子，但内容本身不应重叠
                sig = tuple(sample[:10].tolist()) 
                fingerprints.add(sig)
        
        return fingerprints

    # 模拟 World Size = 2
    WS = 2
    
    # 1. 获取 Rank 0 的数据指纹
    fingerprints_rank0 = collect_data_fingerprints(rank=0, world_size=WS)
    print(f"    -> Rank 0 collected {len(fingerprints_rank0)} unique sample signatures.")

    # 2. 获取 Rank 1 的数据指纹
    fingerprints_rank1 = collect_data_fingerprints(rank=1, world_size=WS)
    print(f"    -> Rank 1 collected {len(fingerprints_rank1)} unique sample signatures.")

    # 3. 比较交集
    intersection = fingerprints_rank0.intersection(fingerprints_rank1)
    
    if len(intersection) == 0:
        print("\n✅ [PASS] No overlap found between Rank 0 and Rank 1.")
        print("       The data distribution is completely disjoint.")
    else:
        print(f"\n❌ [FAIL] Found {len(intersection)} overlapping samples!")
        print("       This means Rank 0 and Rank 1 are processing the same data.")
    
    # 4. 验证 Rank 1 是否真的有数据
    if len(fingerprints_rank1) == 0:
        print("❌ [FAIL] Rank 1 yielded no data! (Did islice skip everything?)")
    else:
        print("✅ [PASS] Rank 1 successfully yielded data.")

    print("\n" + "="*60)
    print(">>> All Tests Completed.")
    print("="*60)