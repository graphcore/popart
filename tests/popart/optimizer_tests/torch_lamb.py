# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import math
import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2,
                 biasCorrection=True):
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        biasCorrection=biasCorrection)
        super(Lamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lamb, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("biasCorrection", True)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                if group["biasCorrection"]:
                    bias_correction1 = 1 - beta1**state["step"]
                    bias_correction2 = 1 - beta2**state["step"]
                else:
                    bias_correction1 = 1
                    bias_correction2 = 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"])

                upd = ((exp_avg / bias_correction1) /
                       denom) + group["weight_decay"] * p.data

                r1 = p.data.pow(2).sum().sqrt()
                r2 = upd.pow(2).sum().sqrt()

                if r1 == 0 or r2 == 0:
                    trust = 1.0
                else:
                    trust = r1 / r2

                p.data.add_(upd, alpha=-group['lr'] * trust)

        return loss
