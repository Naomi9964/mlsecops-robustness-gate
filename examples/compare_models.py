# Compare robustness across the model zoo: train every architecture and
# run the same attack suite, then print one comparison table.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogate import ATTACKS, MODELS, accuracy, evaluate_attack, make_loaders, train

EPS = 0.1


def main():
    train_loader, test_loader = make_loaders(n=2000)
    rows = []
    for name, cls in MODELS.items():
        model = train(cls(), train_loader, epochs=5)
        clean = accuracy(model, test_loader)
        row = {"model": name, "clean": clean}
        for atk_name in ["fgsm", "pgd", "bim", "nes", "noise"]:
            m = evaluate_attack(model, test_loader, ATTACKS[atk_name], EPS)
            row[atk_name] = m["robust_acc"]
        rows.append(row)

    header = ["model", "clean"] + ["fgsm", "pgd", "bim", "nes", "noise"]
    print(f"# Architecture robustness comparison (eps={EPS})\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        cells = [r["model"], f"{r['clean']:.1%}"] + [f"{r[a]:.1%}" for a in header[2:]]
        print("| " + " | ".join(cells) + " |")
    print("\nvalues = robust accuracy (higher is better)")


if __name__ == "__main__":
    main()
