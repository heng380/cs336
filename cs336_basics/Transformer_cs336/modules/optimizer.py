from collections.abc import Callable, Iterable 
from typing import Optional 
import torch 
import math

def get_gradient_clipping(parameters: Iterable[torch.nn.Parameter], clip: float, eps: float = 1e-6):
    general_norm = 0
    for p in parameters:
        if p.grad is not None:
            general_norm += p.grad.data.norm(2) ** 2
    general_norm = general_norm ** 0.5
    if general_norm > clip:
        norm_coef = clip/(general_norm+eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(norm_coef)

def get_lr_cosine_schedule(t, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if t < warmup_iters:
        return t/warmup_iters*max_learning_rate
    elif t <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (t-warmup_iters) / (cosine_cycle_iters-warmup_iters))) * (max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8) -> None:
        defaults = {
            "lr":lr, 
            "weight_decay":weight_decay,
            "betas":betas,
            "eps":eps
        }
        super().__init__(params, defaults)   # initialize self.param_groups as list, each item is a dict, with key defaults list and "params"
                                           # initialize self.state as dict, each key is Paramter to be optimized

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad == None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                beta1,beta2 = betas

                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v = state["v"]
                v.mul_(beta2).add_(grad*grad, alpha=1-beta2)

                lr_t = lr * math.sqrt(1-beta2**t)/(1-beta1 ** t)
                p.data.addcdiv_(m, v.sqrt()+eps, value=-lr_t)
                p.data.add_(p.data, alpha=-lr*weight_decay)
                state["t"] = t + 1
                
        return loss




class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number or default to 0.
                grad = p.grad.data     # Gradient of loss with respect to p.

                # Update weights with decaying learning rate.
                p.data -= lr / math.sqrt(t + 1) * grad

                # Update iteration count.
                state["t"] = t + 1

        return loss


# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# print (weights)

# opt = SGD([weights], lr=1e1)

# for t in range(10):

#     opt.zero_grad() # Reset the gradients for all learnable parameters. 
#     loss = (weights**2).mean() # Compute a scalar loss value. 
#     print(loss.cpu().item()) 
#     loss.backward() # Run backward pass, which computes gradients. 
#     opt.step() # Run optimizer step.