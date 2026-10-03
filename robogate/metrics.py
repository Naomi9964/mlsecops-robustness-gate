# Clean accuracy, robust accuracy, and attack success rate.

import torch


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total if total else 0.0


def evaluate_attack(model, loader, attack, eps):
    # robust_acc: accuracy on adversarial examples
    # asr: fraction of correctly-classified clean samples that get flipped
    model.eval()
    clean_ok, flipped, total = 0, 0, 0
    for x, y in loader:
        x_adv = attack(model, x, y, eps)
        with torch.no_grad():
            clean_pred = model(x).argmax(1)
            adv_pred = model(x_adv).argmax(1)
        ok = clean_pred == y
        clean_ok += ok.sum().item()
        flipped += (ok & (adv_pred != y)).sum().item()
        total += len(y)
    robust_acc = (clean_ok - flipped) / total if total else 0.0
    asr = flipped / clean_ok if clean_ok else 0.0
    return {"robust_acc": robust_acc, "asr": asr}
