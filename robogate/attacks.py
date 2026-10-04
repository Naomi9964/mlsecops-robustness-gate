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


class BIM(Attack):
    name = "bim"

    def __init__(self, steps=20):
        self.steps = steps

    def __call__(self, model, x, y, eps):
        # like PGD but starts from the clean image (no random start)
        model.eval()
        alpha = eps / self.steps
        x_adv = x.clone().detach()
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = _project_linf(x_adv.detach() + alpha * grad.sign(), x, eps)
        return x_adv.detach()


class NoiseAttack(Attack):
    name = "noise"  # baseline: uniform random noise, no model access at all

    def __call__(self, model, x, y, eps):
        return _project_linf(
            x + torch.empty_like(x).uniform_(-eps, eps), x, eps
        ).detach()


class NESAttack(Attack):
    name = "nes"  # black-box: gradient estimated from query scores only

    def __init__(self, steps=40, samples=12, sigma=0.05):
        self.steps = steps
        self.samples = samples  # per direction; antithetic pairs double it
        self.sigma = sigma

    @torch.no_grad()
    def __call__(self, model, x, y, eps):
        model.eval()
        x_adv = x.clone().detach()
        lr = eps / self.steps * 2.0
        n = self.samples
        for _ in range(self.steps):
            d = torch.randn(n, *x.shape[1:], device=x.device) * self.sigma
            d = torch.cat([d, -d])  # antithetic pairs
            m = d.shape[0]
            b = x.shape[0]
            xa = (x_adv.unsqueeze(1) + d.unsqueeze(0)).clamp(0.0, 1.0)
            logits = model(xa.reshape(b * m, *x.shape[1:]))
            loss = F.cross_entropy(
                logits, y.repeat_interleave(m), reduction="none"
            ).reshape(b, m)
            # NES gradient estimate: mean(loss_i * d_i) / sigma
            grad = (loss.reshape(b, m, 1, 1, 1) * d.unsqueeze(0)).mean(dim=1) / self.sigma
            x_adv = _project_linf(x_adv + lr * grad.sign(), x, eps)
        return x_adv.detach()


ATTACKS = {
    "fgsm": FGSM(),
    "pgd": PGD(),
    "bim": BIM(),
    "nes": NESAttack(),
    "noise": NoiseAttack(),
}
