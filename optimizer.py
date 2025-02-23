from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                # Initialize state
                state["m_t"] = state.get("m_t", torch.zeros_like(p.data))
                state["v_t"] = state.get("v_t", torch.zeros_like(p.data))
                state["t"] = state.get("t", 0)

                # Main algoritm 
                state["t"] += 1
                g_t = p.grad.data
                state["m_t"] = beta1 * state["m_t"] + (1 - beta1) * g_t
                state["v_t"] = beta2 * state["v_t"] + (1 - beta2) * g_t * g_t
                alpha_t = alpha * ((1 - beta2**state["t"])**0.5) / (1 - beta1**state["t"])
                p.data = p.data - alpha_t * state["m_t"] / (state["v_t"]**0.5 + eps)

                if weight_decay != 0:
                    p.data = p.data - alpha * weight_decay * p.data
                
        return loss
