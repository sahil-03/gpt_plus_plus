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

class DiagonalPreconditioner(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8, max_precond=1e4):
        defaults = dict(lr=lr, beta=beta, eps=eps, max_precond=max_precond)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg_sq = state['exp_avg_sq']
                beta = group['beta']
                
                state['step'] += 1
                
                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                
                bias_correction = 1 - beta ** state['step']
                corrected_exp_avg_sq = exp_avg_sq / bias_correction
                
                preconditioner = (corrected_exp_avg_sq + group['eps']).sqrt_()
                preconditioner.clamp_(min=1/group['max_precond'], max=group['max_precond'])
                
                p.addcdiv_(grad, preconditioner, value=-group['lr'])

class PreconditionedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, precond_beta=0.9, max_precond=1e4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       precond_beta=precond_beta, max_precond=max_precond)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['precond_avg'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                beta1, beta2 = group['betas']
                precond_beta = group['precond_beta']
                
                state['step'] += 1
                
                # Preconditioner update
                precond_avg = state['precond_avg']
                precond_avg.mul_(precond_beta).addcmul_(grad, grad, value=1 - precond_beta)
                
                precond_bias_correction = 1 - precond_beta ** state['step']
                precond = (precond_avg / precond_bias_correction + group['eps']).sqrt_()
                precond.clamp_(min=1/group['max_precond'], max=group['max_precond'])
                
                grad = grad / precond
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                step_size = group['lr']
                if group['betas'][0] != 1:
                    step_size = step_size * math.sqrt(1 - beta2 ** state['step'])
                    step_size = step_size / (1 - beta1 ** state['step'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
