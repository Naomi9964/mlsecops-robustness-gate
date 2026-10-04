# Logic tests for the RAG indirect-injection threat model.
# Offline: TF-IDF retriever + MockGenerator, no external services.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogate import (
    DEFAULT_RAG_POLICY, RAG_ATTACKS, RAGGate, RAGSystem, MockGenerator,
    TFIDFRetriever, check_rag, evaluate_rag_attack,
    make_corpus, make_queries, to_rag_json, to_rag_markdown,
)


def make_system(extra_docs, mode="naive"):
    return RAGSystem(make_corpus() + extra_docs, MockGenerator(mode=mode))


def test_retriever_surfaces_poisoned_doc():
    atk = RAG_ATTACKS["direct"]
    r = TFIDFRetriever(make_corpus() + atk.poison_docs())
    got = [d["id"] for d in r.retrieve("how do I reset my password?", k=2)]
    assert atk.poison_docs()[0]["id"] in got


def test_naive_follows_injection_defended_ignores():
    from robogate.rag_gate import _strip_quoted
    atk = RAG_ATTACKS["direct"]
    q = make_queries()[0]["query"]
    poisoned = atk.poison_docs()
    naive_resp, naive_docs = make_system(poisoned, mode="naive").answer(q)
    defended_resp, defended_docs = make_system(poisoned, mode="defended").answer(q)
    # judge on unquoted wording only: quoting the doc is not following it
    assert atk.canary in _strip_quoted(naive_resp, naive_docs)
    assert atk.canary not in _strip_quoted(defended_resp, defended_docs)


def test_gate_fails_on_naive():
    results = RAGGate(
        lambda extra: make_system(extra, mode="naive"),
        make_queries(), RAG_ATTACKS, DEFAULT_RAG_POLICY,
    ).run()
    assert results["verdict"] == "FAIL"
    assert results["attacks"]["direct"]["injection_rate"] == 1.0
    assert len(results["violations"]) >= 1


def test_gate_passes_on_defended():
    results = RAGGate(
        lambda extra: make_system(extra, mode="defended"),
        make_queries(), RAG_ATTACKS, DEFAULT_RAG_POLICY,
    ).run()
    assert results["verdict"] == "PASS"
    assert all(m["injection_rate"] == 0.0
               for m in results["attacks"].values())


def test_check_rag_policy_logic():
    results = {"clean_acc": 1.0,
               "attacks": {"direct": {"injection_rate": 0.5}},
               "verdict": "", "violations": []}
    policy = {"min_clean_accuracy": 0.9,
              "max_injection_rate": [{"attack": "direct", "max_rate": 0.2}]}
    verdict, violations = check_rag(results, policy)
    assert verdict == "FAIL" and len(violations) == 1


def test_rag_report_shape():
    results = RAGGate(
        lambda extra: make_system(extra, mode="naive"),
        make_queries(), RAG_ATTACKS, DEFAULT_RAG_POLICY,
    ).run()
    md = to_rag_markdown(results)
    assert "RAG Indirect-Injection Gate Report" in md and "FAIL" in md
    assert '"verdict": "FAIL"' in to_rag_json(results)
