"""Single-paper title/abstract TRL estimation."""

from .api import analyze
from .explain import compose_rationale, explain_linear, explain_occlusion
from .model import SciBERTAdapter, SVMAdapter, resolve_label_map
from .rules import LEVELS, RuleTRLAnalyzer
from .segmentation import sentences
from .standard import align_to_standard, evaluate_gates, load_standard
from .textmatch import clip_span, term_pattern

__all__ = [
    "LEVELS",
    "RuleTRLAnalyzer",
    "SciBERTAdapter",
    "SVMAdapter",
    "align_to_standard",
    "analyze",
    "clip_span",
    "compose_rationale",
    "evaluate_gates",
    "explain_linear",
    "explain_occlusion",
    "load_standard",
    "resolve_label_map",
    "sentences",
    "term_pattern",
]
__version__ = "0.4.0"
