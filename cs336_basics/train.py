import os
import time
import yaml
import torch
import numpy as np
from torch import nn
import cs336_basics.Transformer_cs336 as my_tf
import wandb
import subprocess

def get_gpu_stats():
    """获取 GPU 使用情况"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')
    stats = [line.split(', ') for line in output]
    return stats

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    config = load_config("./cs336_basics/configures/m4.yaml")

    wandb_flag = config["training"]["wandb"]

    if wandb_flag:
        # wandb.init(project=config["training"]["wandb_project"])

        # 给当前任务起个名字（比如根据时间、参数等）
        run_name = f"bs{config['training']['batch_size']}_lr{config['optimizer']['learning_rate_max']}_layer24"

        # 初始化 wandb，并传入 name 和 config
        wandb.init(
            project=config["training"]["wandb_project"],
            name=run_name,
            config={
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["optimizer"]["learning_rate_max"],
                "context_length": config["model"]["context_length"],
                "embedding_dim": config["model"]["d_model"],
                "max_iters": config["training"]["max_iters"]
            }
        )
    device = detect_device() if config["training"]["device"] == "auto" else config["training"]["device"]
    print (f"device: {device}")

    train_data = np.memmap(config["dataset"]["train_path"], dtype=np.uint32, mode="r")
    val_data = np.memmap(config["dataset"]["val_path"], dtype=np.uint32, mode="r")

    max_l2_norm = config["optimizer"]["max_l2_norm"]

    model = my_tf.transformer.Transformer(
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff = config["model"]["d_ff"],
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        num_layers=config["model"]["num_layers"],
        max_seq_len=config["model"]["context_length"],
        theta = config["model"]["rope_theta"]
    ).to(device)

    model.train()

    optimizer = my_tf.modules.AdamW(
        params=model.parameters(),
        lr = float(config["optimizer"]["learning_rate_max"]),
        weight_decay=float(config["optimizer"]["weight_decay"])
    )

    for it in range(config["training"]["max_iters"]):
        lr = my_tf.get_lr_cosine_schedule(t=it, 
                                          max_learning_rate=float(config["optimizer"]["learning_rate_max"]),
                                          min_learning_rate=float(config["optimizer"]["learning_rate_min"]),
                                          warmup_iters=config["optimizer"]["warmup_iters"],
                                          cosine_cycle_iters=config["optimizer"]["cosine_iters"]
                                          )
        
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = my_tf.modules.get_batch(dataset=train_data, batch_size=config["training"]["batch_size"],
                                       context_length=config["model"]["context_length"], device=device)
        
        logits = model(x)
        loss = my_tf.modules.get_cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        print ("loss:", loss)

        optimizer.zero_grad()
        loss.backward()
        my_tf.modules.get_gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()

        if wandb_flag:
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "step":it})

        if it % config["training"]["log_every"] == 0:
            print (f"Step {it}: loss = {loss.item():.4f}, lr = {lr:.6f}")

        if it % config["training"]["val_every"] == 0 and it>0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = my_tf.modules.get_batch(
                    dataset = val_data,
                    batch_size = config["training"]["batch_size"],
                    context_length = config["model"]["context_length"],
                    device=device
                )
                logits_val = model(x_val)
                val_loss = my_tf.modules.get_cross_entropy_loss(logits_val[:, -1, :], y_val[:, -1])

                if wandb_flag:
                    wandb.log({"val/loss": val_loss.item(), "step":it})
                print(f"[Validation] Step {it}: val_loss = {val_loss.item():.4f}")
            model.train()
        
        if it % config["training"]["val_every"] == 0:
            my_tf.modules.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=it,
                out=config["training"]["checkpoint_path"]
            )

        if it % config["training"]["gpu_every"] == 0:
            gpu_stats = get_gpu_stats()
            allocated_memory = int(torch.cuda.memory_allocated() / 1024**2)
            cached_memory = (torch.cuda.memory_reserved() / 1024**2)
            wandb.log({f'allocated_memory':allocated_memory, f'cached_memory':cached_memory})
            for idx, stat in enumerate(gpu_stats):
                utilization, memory_used, temperature = map(float, stat)
                
                wandb.log({f'gpu_{idx}_utilization': utilization, f'gpu_{idx}_memory_used': memory_used, f'gpu_{idx}_temperature': temperature})
        
if __name__ == "__main__":
    main()

        




    

