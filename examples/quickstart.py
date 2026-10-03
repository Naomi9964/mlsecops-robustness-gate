# Quickstart: train the demo model, run the full gate, write the report.
# Exits non-zero when the gate verdict is FAIL (CI-friendly).

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogate import (
    ATTACKS, DEFAULT_POLICY, Gate, TinyCNN, make_loaders,
    to_json, to_markdown, train,
)


def main():
    train_loader, test_loader = make_loaders(n=2000)
    model = train(TinyCNN(), train_loader, epochs=5)
    gate = Gate(model, test_loader, ATTACKS, epsilons=[0.05, 0.1, 0.2],
                policy=DEFAULT_POLICY)
    results = gate.run()

    root = Path(__file__).resolve().parents[1]
    md = to_markdown(results)
    print(md)
    (root / "report.md").write_text(md)
    (root / "report.json").write_text(to_json(results))
    print(f"\nverdict: {results['verdict']}")
    sys.exit(0 if results["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
