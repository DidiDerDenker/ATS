# Imports
import torch

from torch.optim.optimizer import Optimizer


# Classes
class Adam(Optimizer):
    """
    Implements the adam-algorithm as proposed in the paper "Adam: A Method for Stochastic Optimization".

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond" (default: False)

    ... A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
    ... On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """ Performs a single optimization step and returns the loss. """

        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider using SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    # Exponential moving average of gradient values
                    state["next_m"] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values
                    state["next_v"] = torch.zeros_like(p.data)

                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["betas"]

                # Decay first and second moment running-avg coefficient with inplace-operations to update the averages
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group["eps"])

                # Decay the weights by adding the square of the weights to the loss with non-momentum SGD instead of L2
                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                lr_scheduled = group["lr"]
                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state["step"] += 1

        return loss
