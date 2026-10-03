# Markdown + JSON reports. The Markdown table is what reviewers read;
# the JSON is what CI and dashboards consume.

import json


def to_markdown(results):
    lines = ["# Robustness Gate Report", "", f"**Verdict: {results['verdict']}**", ""]
    if results["violations"]:
        lines += ["## Violations"] + [f"- {v}" for v in results["violations"]] + [""]
    lines += [
        "## Results",
        "",
        f"clean accuracy: {results['clean_acc']:.2%}",
        "",
        "| attack | eps | robust acc | ASR |",
        "| --- | --- | --- | --- |",
    ]
    for name, per_eps in results["attacks"].items():
        for eps, m in per_eps.items():
            lines.append(f"| {name} | {eps} | {m['robust_acc']:.2%} | {m['asr']:.2%} |")
    lines += ["", "ASR = attack success rate (lower is better)."]
    return "\n".join(lines)


def to_json(results):
    # eps floats become strings as JSON keys; keep it simple and explicit
    serializable = {
        "verdict": results["verdict"],
        "violations": results["violations"],
        "clean_acc": results["clean_acc"],
        "attacks": {
            name: {str(eps): m for eps, m in per_eps.items()}
            for name, per_eps in results["attacks"].items()
        },
    }
    return json.dumps(serializable, indent=2)
