# Gate policy: YAML thresholds turn metrics into a PASS / FAIL verdict.

import yaml

DEFAULT_POLICY = {
    "min_robust_accuracy": [
        {"attack": "fgsm", "eps": 0.1, "min_acc": 0.80},
        {"attack": "pgd", "eps": 0.1, "min_acc": 0.70},
    ],
    "max_clean_accuracy_drop": 0.05,  # vs. the recorded baseline below
    "baseline_clean_accuracy": 0.95,
}


def load_policy(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check(results, policy):
    # results: {"clean_acc": float, "attacks": {name: {eps: {"robust_acc", "asr"}}}}
    violations = []
    for rule in policy.get("min_robust_accuracy", []):
        got = results["attacks"][rule["attack"]][rule["eps"]]["robust_acc"]
        if got < rule["min_acc"]:
            violations.append(
                f"{rule['attack']} @ eps={rule['eps']}: robust_acc {got:.2%} < {rule['min_acc']:.2%}"
            )
    baseline = policy.get("baseline_clean_accuracy")
    max_drop = policy.get("max_clean_accuracy_drop")
    if baseline is not None and max_drop is not None:
        drop = baseline - results["clean_acc"]
        if drop > max_drop:
            violations.append(
                f"clean_acc dropped {drop:.2%} vs baseline {baseline:.2%} (max {max_drop:.2%})"
            )
    return ("PASS" if not violations else "FAIL"), violations
