# Defended RAG demo: the generator treats retrieved docs as untrusted
# data and never follows instructions found in them -> expect PASS.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogate import (
    DEFAULT_RAG_POLICY, RAG_ATTACKS, RAGGate, RAGSystem,
    MockGenerator, make_corpus, make_queries,
    to_rag_json, to_rag_markdown,
)


def make_system(extra_docs):
    return RAGSystem(make_corpus() + extra_docs, MockGenerator(mode="defended"))


def main():
    gate = RAGGate(make_system, make_queries(), RAG_ATTACKS,
                   DEFAULT_RAG_POLICY)
    results = gate.run()

    root = Path(__file__).resolve().parents[1]
    md = to_rag_markdown(results)
    print(md)
    (root / "rag_report_defended.md").write_text(md)
    (root / "rag_report_defended.json").write_text(to_rag_json(results))
    print(f"\nverdict: {results['verdict']}")
    sys.exit(0 if results["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
