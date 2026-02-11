import torch
import torch.nn as nn
import os

class toy_AdamW(torch.optim.Optimizer):
    def __init__(self, params,betas=(0.9,0.999),weight_decay = 0,eps = 1e-8,lr = 1e-3) -> None:
        defaults = dict(lr=lr,beta1 = betas[0],beta2 = betas[1], decay = weight_decay, eps = eps)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            decay = group["decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) + 1# Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad
                
                m_new = beta1 * m + (1 - beta1)* grad
                v_new = beta2 * v + (1 - beta2)* grad**2
                lr_now = lr * ((1 - beta2**t)**(0.5)) / (1 - beta1**t)
                p -= lr_now * m_new / (v_new**0.5 + eps) + lr*decay*p
                
                state["t"] = t
                state["m"] = m_new
                state["v"] = v_new                
        return loss