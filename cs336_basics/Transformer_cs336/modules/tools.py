import numpy.typing as npt
import torch
import numpy as np
from torch.nn import Module
from torch.optim import Optimizer

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device:str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start+1, size=batch_size)
    input_np = np.stack([
        dataset[i:i+context_length] for i in start_indices
    ])
    label_np = np.stack([
        dataset[i+1:i+context_length+1] for i in start_indices
    ])
    input_tensor = torch.tensor(input_np, dtype=torch.long, device=device)
    output_tensor= torch.tensor(label_np, dtype=torch.long, device=device)

    return [input_tensor, output_tensor]


def save_checkpoint(model:Module, optimizer:Optimizer, iteration:int, out):
    check_point = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(check_point, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]