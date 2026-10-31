"""Optional SVM/SciBERT adapters trained in the 2026-10-23 balanced experiment.

Both adapters expose ``predict`` (label + score) and ``explain`` (sentence-level
attribution). The attribution is a post-hoc reading of the classifier, never a
substitute for the quote evidence produced by the rule backend.

Head width of the current release is **11-way** — levels 1-9 plus the two
abstain labels ``not_applicable`` / ``insufficient_evidence`` — because the
balanced 2026-10-23 artifacts (``svm_local.joblib``, ``scibert_local``) were
trained that way. **Only this schema is supported.** The earlier 9-way
checkpoints from ``paper_trl_external_20261023`` (``svm_masked.joblib`` /
``scibert_trial``: CORDIS project texts, project-level silver labels, cannot
abstain) are rejected with ``ValueError`` — they were trained on a different
input distribution and, in evaluation, collapsed to the training prior on
OpenAlex abstracts. Loading a 9-way head must fail loudly instead of silently
producing a level.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .explain import as_scores, explain_linear, explain_occlusion, level_from_label

# Label space of the current release: 11-way only (levels 1-9 + the two abstain
# labels below), matching models/svm_local.joblib + scibert_local of the
# balanced 2026-10-23 run. The older 9-way head from
# work/paper_trl_external_20261023 (models/svm_masked.joblib + models/scibert_trial,
# CORDIS project texts) is deliberately NOT supported: it cannot abstain and was
# trained on a different input distribution. See unsupported_head_error().
CARD_CLASSES = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "not_applicable", "insufficient_evidence",
)
SUPPORTED_HEAD_WIDTHS = (11,)
CURRENT_HEAD_WIDTH = 11
ABSTAIN_LABELS = {"not_applicable", "insufficient_evidence"}
TRL_LABELS = {str(i) for i in range(1, 10)}
TRL_LABEL = re.compile(r"\s*(?:TRL\s*)?([1-9])\s*", re.I)
GENERIC_LABEL = re.compile(r"LABEL_\d+")
DEFAULT_MAX_LENGTH = 512
MIN_MAX_LENGTH = 32
MAX_MAX_LENGTH = 512


def _clean(text: str) -> str:
    return re.sub(r"\b(?:TRLs?|technology\s+readiness\s+levels?)\s*(?:of|is|at|from|to|level|[:=\-])?\s*[1-9](?:\s*(?:[-–/]\s*|to|and)\s*[1-9])?", "readiness stage", text, flags=re.I)


def _compose_text(title: str, abstract: str = "") -> str:
    return f"{str(title or '').strip()}\n{str(abstract or '').strip()}".strip()


def _parse_label(label: str) -> tuple[int | None, str]:
    if label in ABSTAIN_LABELS:
        return None, label
    match = TRL_LABEL.fullmatch(label)
    if match:
        return int(match.group(1)), label
    return None, label


def unsupported_head_error(width: int | None, detail: str = "") -> ValueError:
    """Uniform error for every head that is not the current 11-class schema."""
    extra = f" ({detail})" if detail else ""
    return ValueError(
        f"unsupported classifier head{extra}: got {width} classes, but this release only "
        "supports the 11-class head (TRL 1-9 + 'not_applicable' + 'insufficient_evidence'). "
        "Use models/svm_local.joblib or models/scibert_local from "
        "paper_trl_balanced_20261023. The 9-class head from paper_trl_external_20261023 "
        "(models/svm_masked.joblib, models/scibert_trial; CORDIS project texts, no abstain "
        "classes) is no longer supported."
    )


def resolve_label_map(id2label: dict[int, str] | None, num_labels: int) -> dict[int, tuple[int | None, str]]:
    """Map head indices of the **11-class** head to ``(level, label)``.

    A re-headed or re-ordered checkpoint would silently map levels wrong if this
    always assumed ``index + 1``. The rule is:

    1. if ``id2label`` contains recognisable TRL / abstain names, trust it — but
       it must cover levels 1-9 and both abstain labels;
    2. generic ``LABEL_n`` checkpoints are mapped by the 11-class layout
       (indices 0-8 = levels 1-9, 9 = not_applicable, 10 = insufficient_evidence);
    3. any other width is rejected (notably the legacy 9-class head).
    """
    if id2label:
        values = {int(k): str(v) for k, v in id2label.items()}
        generic = {k: bool(GENERIC_LABEL.fullmatch(v)) for k, v in values.items()}
        if not all(generic.values()):
            if any(generic.values()):
                raise ValueError(
                    f"id2label mixes generic LABEL_n and semantic names: {id2label}"
                )
            parsed = {k: _parse_label(v) for k, v in values.items()}
            unknown = [values[k] for k, (level, _) in parsed.items()
                       if level is None and values[k] not in ABSTAIN_LABELS]
            if unknown:
                raise ValueError(f"unrecognised id2label names: {unknown}")
            levels = [level for level, _ in parsed.values() if level is not None]
            if len(levels) != len(set(levels)):
                raise ValueError(f"duplicate TRL levels in id2label: {id2label}")
            labels = {label for _, label in parsed.values()}
            if set(levels) != {int(i) for i in TRL_LABELS} or not ABSTAIN_LABELS <= labels:
                missing = sorted(TRL_LABELS - {str(lv) for lv in levels}) + sorted(ABSTAIN_LABELS - labels)
                raise unsupported_head_error(len(values), f"id2label missing {missing}")
            return parsed
        # generic LABEL_0..LABEL_n checkpoints carry no semantic order: use the
        # 11-class layout and refuse anything else instead of guessing.
    if num_labels == CURRENT_HEAD_WIDTH:
        return {
            i: ((i + 1, CARD_CLASSES[i]) if i < 9 else (None, CARD_CLASSES[i]))
            for i in range(CURRENT_HEAD_WIDTH)
        }
    raise unsupported_head_error(num_labels)


class SVMAdapter:
    """TF-IDF + linear SVM; explanations come from the linear decision function."""

    model_name = "balanced_svm_20261023"

    def __init__(self, model_path: str | Path):
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install energy-paper-trl[models]") from exc
        self.model_path = str(model_path)
        self.model = joblib.load(model_path)
        steps = getattr(self.model, "named_steps", None) or {}
        clf = steps.get("clf")
        self.classes = [str(c) for c in clf.classes_] if clf is not None else []
        if self.classes:
            self.validate_classes(self.classes)

    @staticmethod
    def validate_classes(classes: list[str]) -> None:
        """Accept only the current 11-class head; reject the legacy 9-class one."""
        labels = {str(c) for c in classes}
        if len(classes) != CURRENT_HEAD_WIDTH or labels != (TRL_LABELS | ABSTAIN_LABELS):
            raise unsupported_head_error(len(classes) or None, "loaded SVM classes_")

    def _classes(self) -> list[str]:
        classes = getattr(self, "classes", None)
        if classes:
            return list(classes)
        steps = getattr(self.model, "named_steps", None) or {}
        clf = steps.get("clf")
        return [str(c) for c in clf.classes_] if clf is not None else []

    def predict(self, title: str, abstract: str = "") -> dict[str, Any]:
        text = _clean(_compose_text(title, abstract))
        raw = self.model.predict([text])[0]
        label = str(raw)
        level = level_from_label(raw)
        scores = as_scores(self.model.decision_function([text]))
        classes = self._classes()
        if len(scores) == 1 and len(classes) == 2:
            scores = [-scores[0], scores[0]]
        if classes and label in classes and len(scores) == len(classes):
            winner = classes.index(label)
        else:
            winner = max(range(len(scores)), key=scores.__getitem__)
        others = [i for i in range(len(scores)) if i != winner]
        runner = max(others, key=scores.__getitem__) if others else winner
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:3]
        return {
            "level": level,
            "level_label": label,
            # Margin over the runner-up: the raw multiclass decision value is often
            # negative and reads like an error to a human reviewer.
            "score": float(scores[winner] - scores[runner]),
            "score_kind": "decision_margin",
            "decision_value": float(scores[winner]),
            "runner_up_level": level_from_label(classes[runner]) if runner < len(classes) else None,
            "alternatives": [
                {
                    "class": classes[i] if i < len(classes) else None,
                    "level": level_from_label(classes[i]) if i < len(classes) else None,
                    "score": round(float(scores[i]), 6),
                }
                for i in ranked
            ],
            "score_calibrated": False,
            "model": self.model_name,
            "model_path": getattr(self, "model_path", None),
            "classes": len(classes),
            "head_width": len(classes) or None,
            "evidence": [],
        }

    def explain(
        self, title: str, abstract: str = "", *, predicted_level: int | None = None
    ) -> dict[str, Any]:
        text = _compose_text(title, abstract)
        return explain_linear(
            self.model,
            text,
            mask=_clean,
            predicted_level=predicted_level,
            title_chars=len(str(title or "").strip()),
        )


class SciBERTAdapter:
    """Fine-tuned SciBERT head; explanations come from sentence occlusion."""

    model_name = "balanced_scibert_20261023"

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        max_length: int | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install energy-paper-trl[models]") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = str(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(self.device).eval()
        config = getattr(self.model, "config", None)
        id2label = {int(k): str(v) for k, v in (getattr(config, "id2label", None) or {}).items()}
        self.id2label = id2label
        self.num_labels = int(
            getattr(config, "num_labels", 0) or len(id2label) or 0
        )
        self.label_map = resolve_label_map(id2label, self.num_labels)
        self.max_length = self._resolve_max_length(model_dir, max_length)

    @staticmethod
    def _resolve_max_length(model_dir: str | Path, override: int | None) -> int:
        """Tokenization width, never a generation default.

        ``transformers.PretrainedConfig`` exposes ``max_length = 20`` (a generation
        default) even when the checkpoint was trained at 256 or 512, so the config
        object is not a reliable source. Read the raw ``config.json`` for an
        explicit tokenization width instead, then the tokenizer, then 512.
        """
        if override is not None:
            width = int(override)
            if not MIN_MAX_LENGTH <= width <= MAX_MAX_LENGTH:
                raise ValueError(f"max_length={width} outside {MIN_MAX_LENGTH}..{MAX_MAX_LENGTH}")
            return width
        raw = Path(model_dir) / "config.json"
        if raw.is_file():
            try:
                payload = json.loads(raw.read_text())
            except (OSError, ValueError):
                payload = {}
            for key in ("tokenizer_max_length", "max_position_embeddings"):
                value = payload.get(key)
                if isinstance(value, int) and MIN_MAX_LENGTH <= value <= MAX_MAX_LENGTH:
                    return value
        return DEFAULT_MAX_LENGTH

    # -- label space -------------------------------------------------------
    def level_for_index(self, index: int) -> tuple[int | None, str | None]:
        """Map a head index to a TRL level via ``resolve_label_map``."""
        index = int(index)
        label_map = getattr(self, "label_map", None)
        if label_map is None:
            label_map = resolve_label_map(
                getattr(self, "id2label", None), int(getattr(self, "num_labels", 0) or 0)
            )
        if index in label_map:
            return label_map[index]
        label = (getattr(self, "id2label", None) or {}).get(index)
        return _parse_label(label) if label is not None else (None, None)

    # -- inference ---------------------------------------------------------
    def _encode(self, texts: list[str]):
        return self.tokenizer(
            [_clean(t) for t in texts],
            max_length=self.max_length, truncation=True, padding=True,
            return_tensors="pt",
        )

    def _probs_many(self, texts: list[str]) -> list[list[float]]:
        encoded = self._encode(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits.float()
        return self.torch.softmax(logits, dim=-1).tolist()

    def _probs(self, text: str) -> list[float]:
        return self._probs_many([text])[0]

    def predict(self, title: str, abstract: str = "") -> dict[str, Any]:
        raw = _compose_text(title, abstract)
        probs = self._probs(raw)
        index = max(range(len(probs)), key=probs.__getitem__)
        level, label = self.level_for_index(index)
        truncated = (
            len(self.tokenizer(_clean(raw), truncation=False)["input_ids"]) > self.max_length
        )
        alternatives = [
            {
                "index": i,
                "level": self.level_for_index(i)[0],
                "label": self.level_for_index(i)[1],
                "score": round(float(probs[i]), 6),
            }
            for i in sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:3]
        ]
        return {
            "level": level,
            "level_label": label,
            "level_index": index,
            "score": float(probs[index]),
            "score_calibrated": False,
            "alternatives": alternatives,
            "text_truncated": truncated,
            "max_length": self.max_length,
            "model": self.model_name,
            "model_path": self.model_dir,
            "num_labels": self.num_labels,
            "head_width": len(self.label_map) if self.label_map else (self.num_labels or None),
            "evidence": [],
        }

    def explain(
        self, title: str, abstract: str = "", *, predicted_level: int | None = None
    ) -> dict[str, Any]:
        raw = _compose_text(title, abstract)
        probs = self._probs(raw)
        index = int(max(range(len(probs)), key=probs.__getitem__))
        if predicted_level is not None:
            for candidate in range(len(probs)):
                if self.level_for_index(candidate)[0] == int(predicted_level):
                    index = candidate
                    break
        attribution = explain_occlusion(
            self._probs, raw, index,
            title_chars=len(str(title or "").strip()),
            probs_batch_fn=self._probs_many,
        )
        level, label = self.level_for_index(index)
        attribution["target"] = {
            "index": index,
            "level": level,
            "label": label,
            "probability": round(float(probs[index]), 6),
        }
        return attribution
