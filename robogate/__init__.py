# robogate: automated adversarial-robustness gate for PyTorch models.
from .attacks import ATTACKS, Attack, BIM, FGSM, NESAttack, NoiseAttack, PGD
from .data import make_dataset, make_loaders
from .gate import Gate
from .metrics import accuracy, evaluate_attack
from .model_zoo import MODELS, DeepCNN, MLP, TinyCNN, train
from .malware_attacks import (
    AppendEvasion, EvasionAttack, FGSMevasion, RandomEvasion, make_malware_attacks,
)
from .malware_data import (
    FAMILY_NAMES, MalwareMLP, benign_centroid, make_family_features,
    make_family_loaders, make_malware_loaders, make_pe_features,
    train_malware, train_malware_adversarial,
)
from .malware_gate import (
    DEFAULT_MALWARE_POLICY, MalwareGate, check_malware, detection_rate,
    evaluate_evasion, evaluate_evasion_by_family,
    to_malware_json, to_malware_markdown,
)
from .policy import DEFAULT_POLICY, check, load_policy
from .rag_attacks import (
    AuthorityInjection, DirectInjection, HiddenInjection, PoisonAttack,
    RAG_ATTACKS,
)
from .rag_data import TFIDFRetriever, make_corpus, make_queries
from .rag_gate import (
    DEFAULT_RAG_POLICY, RAGGate, check_rag, clean_accuracy,
    evaluate_rag_attack, to_rag_json, to_rag_markdown,
)
from .rag_targets import MockGenerator, OllamaGenerator, RAGSystem
from .report import to_json, to_markdown

__all__ = [
    "ATTACKS", "Attack", "BIM", "FGSM", "NESAttack", "NoiseAttack", "PGD",
    "make_dataset", "make_loaders",
    "Gate", "accuracy", "evaluate_attack",
    "MODELS", "DeepCNN", "MLP", "TinyCNN", "train",
    "DEFAULT_POLICY", "check", "load_policy",
    "to_json", "to_markdown",
    "AppendEvasion", "EvasionAttack", "FGSMevasion", "RandomEvasion",
    "make_malware_attacks",
    "MalwareMLP", "benign_centroid", "make_malware_loaders", "make_pe_features",
    "train_malware", "train_malware_adversarial",
    "FAMILY_NAMES", "make_family_features", "make_family_loaders",
    "evaluate_evasion_by_family",
    "DEFAULT_MALWARE_POLICY", "MalwareGate", "check_malware", "detection_rate",
    "evaluate_evasion", "to_malware_json", "to_malware_markdown",
    "AuthorityInjection", "DirectInjection", "HiddenInjection", "PoisonAttack",
    "RAG_ATTACKS",
    "TFIDFRetriever", "make_corpus", "make_queries",
    "DEFAULT_RAG_POLICY", "RAGGate", "check_rag", "clean_accuracy",
    "evaluate_rag_attack", "to_rag_json", "to_rag_markdown",
    "MockGenerator", "OllamaGenerator", "RAGSystem",
]
