'''
This file is for SDAR model pretraining using its built-in block diffusion mask strategy.
'''
import os
import random
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
import torch.nn.functional as F


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from utils.dataset_utils import get_dataloaders
from utils.wsd_schedule import get_wsd_schedule
from models.logging import set_verbosity_info, set_verbosity_error

from utils.utils import get_config, flatten_omega_conf, load_model_tokenizer, log_grad_norm, AverageMeter, save_checkpoint, save_hf_model

logger = get_logger(__name__, log_level="INFO")

def get_text(logits_pred, label_ids, tokenizer, topk=1):
    """
    logits_pred: shape [L-1, V]  或 [V]
    label_ids: shape [L-1]
    tokenizer: HF tokenizer
    """
    # 取 top-1 token
    if logits_pred.ndim == 2:        # [L, V]
        pred_tokens = logits_pred.argmax(dim=-1)  # [L]
    else:                            # [V]
        pred_tokens = logits_pred.unsqueeze(0).argmax(dim=-1)

    # Convert to list
    pred_tokens = pred_tokens.detach().cpu().tolist()
    label_tokens = label_ids.detach().cpu().tolist()

    # tokenizer decode（skip_special_tokens=True 可以过滤 pad/eos）
    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=False)

    return pred_text, label_text


def main():
    #########################
    #      SETUP Config     #
    #########################
    config = get_config()
        
    total_batch_size_per_gpu = config.training.batch_size
    
    config.experiment.output_dir = os.path.join(config.experiment.output_dir, config.experiment.project)
    config.experiment.logging_dir = os.path.join(config.experiment.output_dir, "logs")
    
    #########################
    # SETUP Accelerator     #
    #########################
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    assert num_processes != -1
    accelerator = Accelerator(
        gradient_accumulation_steps=((config.training.total_batch_size // config.training.batch_size) // num_processes),
        mixed_precision=config.training.mixed_precision,
        log_with="tensorboard",
        project_dir=config.experiment.logging_dir,
        step_scheduler_with_optimizer=config.training.step_scheduler_with_optimizer,
    )
    
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = (
            accelerator.gradient_accumulation_steps
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize trackers
    if accelerator.is_main_process:
        log_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        log_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.name,
            config=log_config,
        )

    # Set training seed
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    #########################
    # MODELS and TOKENIZER  #
    #########################
    logger.info("Loading tokenizer and model")
    model, tokenizer = load_model_tokenizer(config=config, logger=logger)
    
    # Set block_size for SDAR model if specified in config
    model.config.block_size = config.model.block_size
    if logger is not None:
        logger.info(f"Set block_size to {config.model.block_size}")
    
    # SDAR model uses its own built-in mask strategy, no need for DiffusionLanguage
    
    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params

    # No decay on bias and layernorm
    no_decay = ["bias", "layer_norm.weight", "ln_f.weight", "wte.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        num_decay_steps=config.lr_scheduler.params.decay_steps,
        num_training_steps=config.training.max_train_steps,
        min_lr_ratio=optimizer_config.learning_rate * config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")

    seq_len = config.dataset.preprocessing.max_seq_length
    
    train_dataloader, val_dataloader = get_dataloaders(config, tokenizer)

    ##################################
    #       Prepare accelerator     #
    ##################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    ##################################
    #       MODEL RESUME         #
    ##################################
    global_step = 0
    resume_step = 0
    resume_checkpoint_dir = None

    if config.experiment.resume_from_checkpoint is not None:
        candidate_path = Path(config.experiment.resume_from_checkpoint)
        if candidate_path.exists():
            resume_checkpoint_dir = candidate_path
        else:
            logger.warning(f"Specified checkpoint not found: {candidate_path}")

    if resume_checkpoint_dir and resume_checkpoint_dir.exists():
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint_dir}")
        
        # 加载模型权重、优化器状态、RNG 状态
        accelerator.load_state(resume_checkpoint_dir)
        
        metadata_file = resume_checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            resume_step = metadata.get("global_step", 0)
        else:
            logger.error(f"Error loading metadata from {metadata_file}")
        
        global_step = resume_step
        logger.info(f"Resumed at global_step={global_step}")

    else:
        logger.warning("No valid checkpoint found or specified, starting fresh training.")
        global_step = 0
        resume_step = 0

    ##################################
    #             Training           #
    ##################################
    total_batch_size = (
        total_batch_size_per_gpu
        * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )
    logger.info("***** Running SDAR pretraining *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    
    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    batches_to_skip = 0
    if resume_step > 0:
        batches_to_skip = resume_step * accelerator.gradient_accumulation_steps
        logger.info(f"Resuming from step {resume_step}, skipping {batches_to_skip} batches...")
        train_dataloader = accelerator.skip_first_batches(train_dataloader, batches_to_skip)

    model.train()

    train_iter = iter(train_dataloader)

    while global_step < config.training.max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # 一个 epoch 跑完了，重新来
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        # *-------*-------*-------*-------*-------*-------*
        # Data Processing
        # *-------*-------*-------*-------*-------*-------*
        input_ids = batch["input_ids"]
        
        # For pretraining with packed sequences, no padding, so no attention_mask needed
        # Generate position_ids: simple sequential positions starting from 0
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        # For pretraining, labels are the same as input_ids
        # SDAR model will automatically handle masking internally using its built-in strategy
        labels = input_ids.clone()
        
        # *-------*-------*-------*-------*-------*-------*
        # Forward & Backward
        # *-------*-------*-------*-------*-------*-------*
        with accelerator.accumulate(model):
            # SDAR model will automatically:
            # 1. Add mask tokens using its built-in strategy
            # 2. Double the input_ids for block diffusion
            # 3. Create flex attention mask
            # 4. Compute loss with FusedLinearDiffusionCrossEntropyLoss
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
            )
            
            loss = outputs.loss
            
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if config.training.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 记录梯度范数 (可选)
                if (global_step + 1) % config.experiment.log_grad_norm_every == 0 and accelerator.is_main_process:
                    log_grad_norm(model, accelerator, global_step + 1)

        # *-------*-------*-------*-------*-------*-------*
        # Logging & Saving & Validation
        # *-------*-------*-------*-------*-------*-------*
        if accelerator.sync_gradients:
            global_step += 1
            
            batch_time_m.update(time.time() - end)
            end = time.time()

            # Logging
            if global_step % config.experiment.log_every == 0:
                avg_loss = accelerator.reduce(loss.detach(), reduction="mean")

                samples_per_second_per_gpu = (
                        accelerator.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                )

                logs = {
                    "step_loss": avg_loss.item(),
                    "train_ppl": math.exp(avg_loss.item()),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "batch_time": batch_time_m.val,
                }
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    logger.info(
                        f"Step: {global_step} | "
                        f"Loss: {avg_loss.item():0.4f} | "
                        f"PPL: {math.exp(avg_loss.item()):.2f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} | "
                        f"Sec/Iter: {batch_time_m.val:0.4f}"
                    )

                batch_time_m.reset()
                data_time_m.reset()

            # Checkpointing
            if global_step % config.experiment.save_every == 0:
                save_checkpoint(model, config, accelerator, global_step)
            
            if global_step % config.experiment.save_hfmodel_every == 0:
                save_hf_model(model, tokenizer, config, accelerator, global_step)
                
            # Validation
            if global_step % config.experiment.val_every == 0:
                validate(model, val_dataloader, accelerator, global_step)
                
                model.train() 

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_hf_model(model, tokenizer, config, accelerator, "final")
    accelerator.end_training()


@torch.no_grad()
def validate(model, val_dataloader, accelerator, global_step):
    # 初始化统计变量
    local_total_loss = torch.tensor(0.0, device=accelerator.device)
    local_total_count = torch.tensor(0.0, device=accelerator.device)
    
    for step, batch in enumerate(val_dataloader):
        input_ids = batch["input_ids"]
        
        # For pretraining with packed sequences, no padding, so no attention_mask needed
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        # Create labels
        labels = input_ids.clone()
        
        current_batch_size = input_ids.size(0)

        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
        )
        
        loss = outputs.loss
        
        local_total_loss += loss.detach() * current_batch_size
        local_total_count += current_batch_size

    global_total_count = accelerator.reduce(local_total_count, reduction="sum")
    global_total_loss = accelerator.reduce(local_total_loss, reduction="sum")
    avg_loss = (global_total_loss / global_total_count).item()
    ppl = math.exp(avg_loss)

    # ==========================================
    # 4. Logging
    # ==========================================
    if accelerator.is_main_process:
        logs = {
            "val/loss": avg_loss,
            "val/ppl": ppl,
        }
        accelerator.log(logs, step=global_step)
        
        logger.info(
            f"[Validation] Step {global_step + 1} | "
            f"Loss: {avg_loss:.4f} (PPL: {ppl:.2f}) | "
        )

    return avg_loss, ppl


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error", message="None of the inputs have requires_grad=True")
    main()
