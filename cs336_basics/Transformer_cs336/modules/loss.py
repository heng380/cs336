import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import repeat, rearrange, einsum, reduce

def get_cross_entropy_loss(inputs: Float[Tensor, "batch vocab_size"], targets: Int[Tensor, "batch"]):
    x_max = reduce(inputs, "b v-> b 1", "max")
    logits = inputs - x_max
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    target_logits = logits[torch.arange(inputs.shape[0]), targets]
    loss =  target_logits - log_sum_exp
    loss =  -loss.mean()
    return loss

def get_perplexity(inputs:Float[Tensor, "batch vocab_size"], targets: Int[Tensor, "batch"]):
    return torch.exp(get_cross_entropy_loss(inputs, targets))