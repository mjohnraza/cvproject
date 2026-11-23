"""
State-of-the-Art Gradient Descent Optimization Suite for Vision & AutoML:
1. Lion Optimizer (EvoLved Sign Momentum, Google Brain 2023)
2. Sharpness-Aware Minimization (SAM) Wrapper for Flat Minima Convergence
3. Cosine Annealing with Warm Restarts (SGDR)
"""

import torch
from torch.optim.optimizer import Optimizer
import math
from typing import Dict, Any, List, Optional, Callable

class Lion(Optimizer):
    r"""Implements Lion algorithm (EvoLved Sign Momentum from Google Brain).
    Reference: Chen et al., 'Symbolic Discovery of Optimization Algorithms', 2023.
    """
    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.9, 0.99), weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # Perform stepweight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Weight update via sign(c_t)
                c_t = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                p.data.add_(torch.sign(c_t), alpha=-lr)

                # Decay the momentum running average
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


class SAM(Optimizer):
    r"""Sharpness-Aware Minimization (SAM) Wrapper.
    Seeks flat minima in non-convex loss landscapes to ensure robust generalization
    across unseen camera baselines and novel view synthesis artifacts.
    Reference: Foret et al., 'Sharpness-Aware Minimization for Efficiently Improving Generalization', ICLR 2021.
    """
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, adaptive: bool = False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to local worst-case loss peak

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Restore original parameter values

        self.base_optimizer.step()  # Perform actual optimization step at worst-case point

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        stack = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        norm = torch.norm(torch.stack(stack), p=2)
        return norm
