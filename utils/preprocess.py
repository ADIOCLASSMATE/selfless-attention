import os
import glob
import time
import multiprocessing
from multiprocessing import Pool
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from datasets import Dataset
import pyarrow as pa
import gc

# 全局变量，用于 Worker 进程初始化
worker_tokenizer = None

def init_worker(model_path):
    """
    Worker 进程初始化函数：每个进程只加载一次 Tokenizer，避免重复开销
    """
    global worker_tokenizer
    # 设为只读，避免并行写入冲突
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        worker_tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"eos_id: {worker_tokenizer.eos_token_id}")
    except Exception as e:
        print(f"Error loading tokenizer in worker: {e}")

def process_text_chunk(text_chunk):
    """
    Worker 任务：接收文本列表，返回 token ids
    """
    global worker_tokenizer
    if not text_chunk:
        return []

    # 批量分词
    batch_encodings = worker_tokenizer(
        text_chunk,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False
    )
    
    input_ids_list = batch_encodings["input_ids"]
    eos_id = worker_tokenizer.eos_token_id
    
    # 展平并添加 EOS (为了减少传输开销，这里可以先不合并，返回列表的列表)
    # 但为了方便主进程，我们在这里做初步处理：给每句话加上 EOS
    processed_ids = []
    for ids in input_ids_list:
        if ids:
            ids.append(eos_id)
            processed_ids.append(ids)
            
    return processed_ids

def save_shard(sequences, output_dir, shard_counter):
    """
    辅助函数：保存分片到磁盘
    """
    if not sequences:
        return 0
        
    dataset = Dataset.from_dict({"input_ids": sequences})
    shard_name = f"shard-{shard_counter:05d}"
    shard_path = os.path.join(output_dir, shard_name)
    dataset.save_to_disk(shard_path)
    return len(sequences)

def main():
    # === 配置 ===
    config_path = "configs/AR/pretraining.yaml"
    output_dir = "/inspire/hdd/global_user/wanjiaxin-253108030048/.cache/huggingface/datasets/fwb-edu-arrow"
    data_path = "/inspire/dataset/fineweb-edu/v1/sample/100BT"
    
    # 分片配置
    SHARD_SIZE = 100_000_000  # 100M tokens per shard
    
    print("=" * 80)
    print("🚀 Starting Dataset Preprocessing (All-Cores-One-File Mode)")
    print("=" * 80)
    
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    max_seq_length = config_dict.get('dataset', {}).get('preprocessing', {}).get('max_seq_length', 2049)
    model_path = config_dict['model']['model_path']
    print(f"📋 Config loaded: max_seq_length = {max_seq_length}")

    # === 1. 扫描文件 ===
    print(f"\n📂 Scanning files in {data_path}...")
    if os.path.isdir(data_path):
        all_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        if not all_files:
            all_files = sorted(glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True))
    else:
        all_files = sorted(glob.glob(data_path)) if isinstance(data_path, str) else data_path

    if not all_files:
        raise ValueError(f"❌ No parquet files found in {data_path}")
    
    print(f"✅ Found {len(all_files)} parquet files")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === 2. 准备多进程 ===
    max_cpu = multiprocessing.cpu_count()
    # 留几个核给主进程做IO和拼装
    num_workers = max(1, max_cpu - 2) 
    print(f"🔧 System: {max_cpu} CPUs available")
    print(f"👷 Using {num_workers} worker processes for tokenization")

    # 初始化进程池 (使用 initializer 加载 tokenizer)
    pool = Pool(processes=num_workers, initializer=init_worker, initargs=(model_path,))

    # === 3. 主循环处理 ===
    shard_counter = 0
    token_buffer = []  # 主进程的大缓冲区
    total_tokens_processed = 0
    total_files_processed = 0
    
    start_time = time.time()
    text_column = "text"

    try:
        overall_pbar = tqdm(total=len(all_files), desc="📂 Files Processed", position=0)
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            # tqdm.write(f"📖 Reading {file_name}...")
            
            # 3.1 读取文件 (主进程)
            try:
                table = pq.read_table(file_path, columns=[text_column])
                text_list = table[text_column].to_pylist()
                
                if not text_list:
                    overall_pbar.update(1)
                    continue
                    
            except Exception as e:
                tqdm.write(f"❌ Error reading {file_name}: {e}")
                continue

            # 3.2 切分任务
            # 将大列表切分为 num_workers * 4 份，保证每个worker有活干且不会一次传输太大
            chunk_size = max(1, len(text_list) // (num_workers * 4))
            chunks = [text_list[i:i + chunk_size] for i in range(0, len(text_list), chunk_size)]
            
            # 释放原始 text_list 内存
            del text_list
            del table
            gc.collect()

            # 3.3 并行分词 (Pool.imap)
            # 使用 imap 而不是 map，这样 worker 一处理完，主进程就能拿到结果，不用等所有都做完
            chunk_iterator = pool.imap(process_text_chunk, chunks)
            
            # 3.4 收集结果并处理缓冲区
            for batch_result_list in chunk_iterator:
                # batch_result_list 是一个 chunk 的分词结果 (list of lists)
                for token_ids in batch_result_list:
                    token_buffer.extend(token_ids)
                
                # 检查是否需要保存分片 (Buffer 足够大)
                while len(token_buffer) >= SHARD_SIZE:
                    # 切出 SHARD_SIZE
                    tokens_to_save = token_buffer[:SHARD_SIZE]
                    token_buffer = token_buffer[SHARD_SIZE:]
                    
                    # 组织成 sequences
                    sequences = []
                    # 这里的切片非常快
                    for i in range(0, len(tokens_to_save), max_seq_length):
                        seq = tokens_to_save[i:i + max_seq_length]
                        if len(seq) == max_seq_length:
                            sequences.append(seq)
                    
                    # 写入磁盘
                    if sequences:
                        save_shard(sequences, output_dir, shard_counter)
                        shard_counter += 1
                        total_tokens_processed += len(sequences) * max_seq_length
                        
                        elapsed = time.time() - start_time
                        speed = total_tokens_processed / elapsed if elapsed > 0 else 0
                        overall_pbar.set_postfix({
                            "Shard": shard_counter, 
                            "Tk/s": f"{speed:.0f}"
                        })

            total_files_processed += 1
            overall_pbar.update(1)
            
            # 手动GC
            gc.collect()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Saving remaining buffer...")
    finally:
        # === 4. 收尾：保存 Buffer 中剩余的数据 ===
        if len(token_buffer) >= max_seq_length:
            tqdm.write(f"💾 Saving residual buffer ({len(token_buffer)} tokens)...")
            sequences = []
            for i in range(0, len(token_buffer), max_seq_length):
                seq = token_buffer[i:i + max_seq_length]
                if len(seq) == max_seq_length:
                    sequences.append(seq)
            
            if sequences:
                save_shard(sequences, output_dir, shard_counter)
                shard_counter += 1
                total_tokens_processed += len(sequences) * max_seq_length

        pool.close()
        pool.join()
        overall_pbar.close()

    # === 5. 统计与元数据 ===
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n{'='*80}")
    print("🎉 Processing Complete!")
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📊 Total Shards: {shard_counter}")
    print(f"📊 Total Tokens: {total_tokens_processed:,}")
    print(f"⚡ Throughput: {total_tokens_processed / elapsed:,.0f} tokens/s")
    print(f"{'='*80}\n")
    
    # 保存 metadata
    import json
    meta_info = {
        "max_seq_length": max_seq_length,
        "total_tokens": total_tokens_processed,
        "total_shards": shard_counter,
        "source_files_processed": total_files_processed,
        "processing_time": elapsed
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta_info, f, indent=4)

if __name__ == "__main__":
    main()