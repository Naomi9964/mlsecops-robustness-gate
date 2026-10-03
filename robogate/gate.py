# The automated workflow, one stage at a time:
# load -> baseline -> attack -> evaluate -> verdict -> report.

from . import metrics
from .policy import check


class Gate:
    def __init__(self, model, test_loader, attacks, epsilons, policy):
        self.model = model
        self.test_loader = test_loader
        self.attacks = attacks      # dict name -> Attack
        self.epsilons = epsilons    # list of float budgets
        self.policy = policy

    def run(self):
        results = {"attacks": {}}
        # 1. baseline: clean accuracy before any attack
        results["clean_acc"] = metrics.accuracy(self.model, self.test_loader)
        # 2-3. attack suite x epsilon budgets, then evaluate
        for name, attack in self.attacks.items():
            results["attacks"][name] = {}
            for eps in self.epsilons:
                results["attacks"][name][eps] = metrics.evaluate_attack(
                    self.model, self.test_loader, attack, eps
                )
        # 4. verdict against the policy
        verdict, violations = check(results, self.policy)
        results["verdict"] = verdict
        results["violations"] = violations
        return results
