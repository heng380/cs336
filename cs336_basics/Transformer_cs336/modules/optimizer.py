from collections.abc import Callable, Iterable 
from typing import Optional 
import torch 
import math
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


weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
print (weights)

opt = SGD([weights], lr=1e1)

for t in range(10):

    opt.zero_grad() # Reset the gradients for all learnable parameters. 
    loss = (weights**2).mean() # Compute a scalar loss value. 
    print(loss.cpu().item()) 
    loss.backward() # Run backward pass, which computes gradients. 
    opt.step() # Run optimizer step.