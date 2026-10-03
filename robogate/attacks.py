# White-box L_inf adversarial attacks, pure PyTorch (no extra attack libs).

import torch
import torch.nn.functional as F


class Attack:
    name = "attack"

    def __call__(self, model, x, y, eps):
        raise NotImplementedError


def _project_linf(x_adv, x, eps):
    # keep the perturbation inside the L_inf ball and valid pixel range
    return torch.clamp(x_adv, x - eps, x + eps).clamp(0.0, 1.0)


class FGSM(Attack):
    name = "fgsm"

    def __call__(self, model, x, y, eps):
        model.eval()
        x_adv = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        return _project_linf(x + eps * grad.sign(), x, eps).detach()


class PGD(Attack):
    name = "pgd"

    def __init__(self, steps=20, alpha=None):
        self.steps = steps
        self.alpha = alpha  # None -> eps / 4

    def __call__(self, model, x, y, eps):
        model.eval()
        alpha = self.alpha if self.alpha is not None else eps / 4.0
        # random start inside the epsilon ball
        x_adv = torch.clamp(
            x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0
        )
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = _project_linf(x_adv.detach() + alpha * grad.sign(), x, eps)
        return x_adv.detach()


ATTACKS = {"fgsm": FGSM(), "pgd": PGD()}
