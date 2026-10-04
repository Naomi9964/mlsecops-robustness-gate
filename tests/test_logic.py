# Pure-logic tests. CPU-only, small synthetic data, no downloads.

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogate import (
    ATTACKS, BIM, DEFAULT_POLICY, FGSM, MODELS, DeepCNN, MLP, NESAttack,
    NoiseAttack, Gate, PGD, TinyCNN, accuracy, check, evaluate_attack,
    make_loaders, to_json, to_markdown, train,
)


@pytest.fixture(scope="module")
def trained():
    train_loader, test_loader = make_loaders(n=400, seed=1)
    model = train(TinyCNN(), train_loader, epochs=2, seed=1)
    return model, test_loader


def test_eps_zero_matches_clean_accuracy(trained):
    model, test_loader = trained
    clean = accuracy(model, test_loader)
    m = evaluate_attack(model, test_loader, FGSM(), eps=0.0)
    assert m["robust_acc"] == pytest.approx(clean, abs=1e-6)
    assert m["asr"] == pytest.approx(0.0, abs=1e-6)


def test_pgd_at_least_as_strong_as_fgsm(trained):
    model, test_loader = trained
    r_fgsm = evaluate_attack(model, test_loader, FGSM(), eps=0.1)["robust_acc"]
    r_pgd = evaluate_attack(model, test_loader, PGD(steps=10), eps=0.1)["robust_acc"]
    assert r_pgd <= r_fgsm + 1e-6


def test_perturbation_stays_inside_eps_ball(trained):
    model, test_loader = trained
    x, y = next(iter(test_loader))
    x_adv = PGD(steps=5)(model, x, y, eps=0.1)
    assert (x_adv - x).abs().max().item() <= 0.1 + 1e-6
    assert x_adv.min().item() >= 0.0 and x_adv.max().item() <= 1.0


def test_strict_policy_fails():
    results = {
        "clean_acc": 0.99,
        "attacks": {"fgsm": {0.1: {"robust_acc": 0.5, "asr": 0.5}}},
        "verdict": "", "violations": [],
    }
    policy = {"min_robust_accuracy": [{"attack": "fgsm", "eps": 0.1, "min_acc": 0.99}]}
    verdict, violations = check(results, policy)
    assert verdict == "FAIL" and len(violations) == 1


def test_gate_verdict_and_report_shape(trained):
    model, test_loader = trained
    policy = {"min_robust_accuracy": [{"attack": "fgsm", "eps": 0.1, "min_acc": 0.5}]}
    gate = Gate(model, test_loader, {"fgsm": ATTACKS["fgsm"]},
                epsilons=[0.1], policy=policy)
    results = gate.run()
    assert results["verdict"] in ("PASS", "FAIL")
    md = to_markdown(results)
    assert "Verdict" in md and "fgsm" in md
    assert json_roundtrip(to_json(results))["verdict"] == results["verdict"]


def json_roundtrip(s):
    import json
    return json.loads(s)


def test_clean_drop_violation():
    results = {
        "clean_acc": 0.80,
        "attacks": {},
        "verdict": "", "violations": [],
    }
    policy = {"min_robust_accuracy": [],
              "baseline_clean_accuracy": 0.95, "max_clean_accuracy_drop": 0.05}
    verdict, violations = check(results, policy)
    assert verdict == "FAIL"


def test_bim_respects_eps_ball(trained):
    model, test_loader = trained
    x, y = next(iter(test_loader))
    xa = BIM(steps=5)(model, x, y, eps=0.1)
    assert (xa - x).abs().max().item() <= 0.1 + 1e-6
    assert xa.min().item() >= 0.0 and xa.max().item() <= 1.0


def test_noise_baseline_is_valid_but_weak(trained):
    model, test_loader = trained
    clean = accuracy(model, test_loader)
    m = evaluate_attack(model, test_loader, NoiseAttack(), eps=0.1)
    assert 0.0 <= m["robust_acc"] <= clean + 1e-6


def test_nes_blackbox_runs_and_respects_eps(trained):
    torch.manual_seed(0)
    model, test_loader = trained
    clean = accuracy(model, test_loader)
    m = evaluate_attack(model, test_loader, NESAttack(steps=10, samples=6), eps=0.2)
    assert m["robust_acc"] <= clean + 1e-6  # never helps the defender
    x, y = next(iter(test_loader))
    xa = NESAttack(steps=5, samples=4)(model, x, y, eps=0.2)
    assert (xa - x).abs().max().item() <= 0.2 + 1e-6


def test_zoo_models_train_and_gate_runs():
    # every model in the zoo must train on the synthetic task and be gateable
    train_loader, test_loader = make_loaders(n=800, seed=1)
    for name, cls in MODELS.items():
        model = train(cls(), train_loader, epochs=6, seed=1)
        acc = accuracy(model, test_loader)
        assert acc > 0.65, f"{name} failed to learn: {acc:.2%}"
        m = evaluate_attack(model, test_loader, FGSM(), eps=0.1)
        assert 0.0 <= m["robust_acc"] <= 1.0
