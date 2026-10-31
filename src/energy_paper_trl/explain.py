"""Post-hoc, sentence-level explanations for the optional classifier backends.

An attribution explains what a classifier reacted to. It never turns a model
prediction into verified evidence: only exact quotes from the input can be
checked, and even those stay title/abstract evidence.

Two methods are implemented here, both dependency-free (callers pass their own
model objects):

* ``explain_linear`` for the TF-IDF + linear-SVM pipeline: positive token
  contributions aggregated to the sentence that contains them.
* ``explain_occlusion`` for any black box with a probability function: drop one
  sentence at a time and measure how much the target class probability falls.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from .rules import BACKGROUND, LEVELS, NEGATIVE
from .segmentation import sentences
from .textmatch import clip_span, term_pattern

ATTRIBUTION_NOTE = "归因说明分类器关注了什么，不等于该阶段已被原文证据支持。"
MAX_QUOTE_CHARS = 320


def rhetorical_context(text: str) -> str | None:
    """Flag sentences whose wording commonly fools a bag-of-words model."""
    if NEGATIVE.search(text):
        return "negated_or_future"
    if BACKGROUND.search(text):
        return "background"
    return None


def as_scores(raw: Any) -> list[float]:
    """Flatten sklearn ``decision_function`` output (binary, 1-d or 2-d)."""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (int, float)):
        raw = [raw]
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [float(v) for v in raw]


def _level_name(level: int | None) -> str | None:
    return LEVELS[level][0] if isinstance(level, int) and level in LEVELS else None


def level_from_label(label: Any) -> int | None:
    """Parse a class label into a TRL level; abstain labels map to ``None``.

    The model artifacts use string labels, including ``insufficient_evidence``
    and ``not_applicable``, which are not TRL levels.
    """
    match = re.fullmatch(r"\s*(?:LABEL_)?([1-9])\s*", str(label))
    return int(match.group(1)) if match else None


def _origin(span_start: int, title_chars: int | None) -> str:
    if title_chars is None:
        return "input"
    return "title" if span_start < title_chars else "abstract"


def explain_linear(
    pipeline: Any,
    text: str,
    *,
    mask: Callable[[str], str] | None = None,
    predicted_level: int | None = None,
    top_sentences: int = 3,
    top_terms: int = 6,
    title_chars: int | None = None,
) -> dict[str, Any]:
    """Attribute a TF-IDF + linear classifier to sentences of ``text``.

    Contributions are ``tfidf_value * (coef[winner] - coef[runner])``, i.e. they
    explain why this class rather than the next best one. Sentences are scored on
    their masked form (explicit "TRL n" phrasing removed) so the attribution stays
    consistent with the model input, while quotes come from the original text.
    """
    spans = sentences(text)
    base: dict[str, Any] = {
        "method": "tfidf_linear_attribution",
        "sentences": [],
        "top_terms": [],
        "not_evidence": True,
        "note": ATTRIBUTION_NOTE,
    }
    if not spans:
        return {**base, "note": ATTRIBUTION_NOTE + "输入为空，无法归因。"}
    steps = getattr(pipeline, "named_steps", None)
    if not steps or "vect" not in steps or "clf" not in steps:
        raise ValueError("explain_linear expects a Pipeline with 'vect' and 'clf' steps")

    vec = steps["vect"]
    clf = steps["clf"]
    masked = [mask(span.text) if mask else span.text for span in spans]
    matrix = vec.transform([" ".join(masked)])

    scores = as_scores(clf.decision_function(matrix))
    classes = [str(c) for c in clf.classes_]
    levels = [level_from_label(c) for c in classes]
    if len(scores) == 1 and len(classes) == 2:
        # Binary heads report one signed score; expand to a two-class vector.
        scores = [-scores[0], scores[0]]
    if predicted_level is not None and int(predicted_level) in levels:
        winner = levels.index(int(predicted_level))
    else:
        winner = max(range(len(scores)), key=scores.__getitem__)
    others = [i for i in range(len(scores)) if i != winner]
    runner = max(others, key=scores.__getitem__) if others else winner

    coef = clf.coef_
    if len(classes) == 1:
        weights = coef[0]
    elif coef.shape[0] == 1:  # binary classifier stores a single weight row
        weights = coef[0] if winner == 1 else -coef[0]
    else:
        weights = coef[winner] - coef[runner]

    names = vec.get_feature_names_out()
    contributions = sorted(
        (
            (str(names[pos]), float(value) * float(weights[pos]))
            for pos, value in zip(matrix.indices.tolist(), matrix.data.tolist(), strict=True)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    positives = [(term, value) for term, value in contributions if value > 0]

    rows: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        matched: list[dict[str, Any]] = []
        score = 0.0
        for term, value in positives:
            # One hit per sentence: repetition should not multiply the score.
            if term_pattern(term).search(masked[index]):
                score += value
                matched.append({"term": term, "contribution": round(value, 6)})
        if score <= 0:
            continue
        matched.sort(key=lambda item: item["contribution"], reverse=True)
        quote = clip_span(text, span.start, span.end, MAX_QUOTE_CHARS)
        rows.append(
            {
                **quote.as_dict(),
                "score": round(score, 6),
                "terms": matched[:top_terms],
                "context": rhetorical_context(span.text),
                "origin": _origin(span.start, title_chars),
            }
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    total = sum(row["score"] for row in rows)
    for row in rows:
        row["share"] = round(row["score"] / total, 4) if total else 0.0

    return {
        **base,
        "method": "tfidf_linear_attribution",
        "predicted_class": classes[winner],
        "predicted_level": levels[winner],
        "level_name": _level_name(levels[winner]),
        "runner_up": {
            "class": classes[runner],
            "level": levels[runner],
            "margin": round(scores[winner] - scores[runner], 6),
        },
        "sentences": rows[:top_sentences],
        "top_terms": [
            {"term": term, "contribution": round(value, 6)}
            for term, value in positives[:top_terms]
        ],
        "masked_input_terms": mask is not None,
    }


def explain_occlusion(
    probs_fn: Callable[[str], Sequence[float]],
    text: str,
    target_index: int,
    *,
    top_sentences: int = 3,
    min_drop: float = 1e-4,
    title_chars: int | None = None,
    probs_batch_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> dict[str, Any]:
    """Sentence occlusion: how much does dropping a sentence cost the target class?

    ``probs_fn`` must return the full class distribution for a raw text and apply
    the same preprocessing as prediction. ``probs_batch_fn`` (optional) scores all
    leave-one-out variants in one forward pass; when absent the variants are scored
    one by one. Only sentences whose removal lowers the probability are reported as
    supporting.
    """
    spans = sentences(text)
    base: dict[str, Any] = {
        "method": "sentence_occlusion",
        "sentences": [],
        "not_evidence": True,
        "note": ATTRIBUTION_NOTE,
    }
    if len(spans) < 3:
        return {**base, "note": ATTRIBUTION_NOTE + "句数过少，遮挡分析不稳定，未给出归因。"}

    variants = [" ".join(s.text for i, s in enumerate(spans) if i != index)
                for index in range(len(spans))]
    if probs_batch_fn is not None:
        batch_probs = [[float(p) for p in row] for row in probs_batch_fn(variants)]
        if len(batch_probs) != len(spans):
            raise ValueError("probs_batch_fn returned a different number of rows")
    else:
        batch_probs = []

    baseline = [float(p) for p in probs_fn(text)]
    if not 0 <= target_index < len(baseline):
        raise ValueError("target_index out of range for the probability vector")

    target_probability = baseline[target_index]
    rows: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        prob = batch_probs[index] if batch_probs else [float(p) for p in probs_fn(variants[index])]
        if target_index >= len(prob):
            raise ValueError("probs_fn returned an inconsistent number of classes")
        drop = target_probability - prob[target_index]
        quote = clip_span(text, span.start, span.end, MAX_QUOTE_CHARS)
        rows.append(
            {
                **quote.as_dict(),
                "drop": round(drop, 6),
                "probability_without": round(prob[target_index], 6),
                "context": rhetorical_context(span.text),
                "origin": _origin(span.start, title_chars),
            }
        )

    supporting = [row for row in rows if row["drop"] > min_drop]
    supporting.sort(key=lambda row: row["drop"], reverse=True)
    total = sum(row["drop"] for row in supporting)
    for row in supporting:
        row["share"] = round(row["drop"] / total, 4) if total else 0.0

    return {
        **base,
        "method": "sentence_occlusion",
        "target": {"index": target_index, "probability": round(target_probability, 6)},
        "sentences": supporting[:top_sentences],
        "n_sentences": len(spans),
    }


def disagreement_summary(rule_level: int | None, model_level: int | None) -> dict[str, Any] | None:
    """Make a rule/model conflict explicit instead of silently picking one."""
    if rule_level is None and model_level is None:
        return None
    if rule_level is None or model_level is None:
        return {
            "rules_level": rule_level,
            "model_level": model_level,
            "agree": False,
            "abs_gap": None,
            "action": "manual_review",
            "reason": "只有一个后端给出了阶段，无法交叉核对。",
        }
    gap = abs(int(model_level) - int(rule_level))
    return {
        "rules_level": int(rule_level),
        "model_level": int(model_level),
        "agree": gap == 0,
        "abs_gap": gap,
        "action": None if gap == 0 else "manual_review",
        "reason": "规则与模型一致。" if gap == 0 else "规则与模型不一致，需人工复核。",
    }


def compose_rationale(
    *,
    backend: str,
    model_level: int | None,
    rule_level: int | None,
    prediction: dict[str, Any],
    attribution: dict[str, Any] | None,
    top_terms_limit: int = 4,
    credited_level: int | None = None,
) -> str:
    """Assemble a short Chinese rationale from label + attribution + rule evidence.

    ``model_level``, ``rule_level`` and ``credited_level`` are kept separate:
    the standard layer may credit a different level than either backend, and that
    must be visible instead of silently reported as the rule conclusion.
    """
    name = _level_name(model_level)
    score = prediction.get("score")
    score_text = ""
    if isinstance(score, (int, float)):
        kind = "决策边际" if backend == "svm" else "softmax 分数"
        score_text = f"（{kind} {float(score):.3f}，未校准）"
    abstain_labels = {"not_applicable", "insufficient_evidence"}
    if model_level is None:
        label = str(prediction.get("level_label") or "")
        if label in abstain_labels:
            head = f"{backend} 模型判为 {label}（不属于 TRL 1–9）{score_text}"
        else:
            head = f"{backend} 模型未给出可用的 TRL 等级{score_text}"
    else:
        head = f"{backend} 模型判为 TRL {model_level}（{name}）{score_text}"
    parts = [head + "。"]
    if backend == "scibert" and isinstance(score, (int, float)) and float(score) < 0.5:
        alternatives = [a for a in (prediction.get("alternatives") or [])[1:2] if isinstance(a, dict)]
        tail = ""
        if alternatives:
            tail = f"，次优为 TRL {alternatives[0].get('level')}（{float(alternatives[0].get('score', 0)):.3f}）"
        parts.append(f"模型自身并不确定：最高类概率不足一半{tail}。")

    if attribution:
        rows = attribution.get("sentences") or []
        if rows:
            top = rows[0]
            terms = "、".join(
                str(item.get("term")) for item in (top.get("terms") or [])[:top_terms_limit]
            )
            fragment = f"模型主要依据该句：“{top.get('quote')}”"
            if terms:
                fragment += f"（关键特征：{terms}）"
            context = top.get("context")
            if context == "negated_or_future":
                fragment += "；该句含否定或未来时态，可能是虚假依据"
            elif context == "background":
                fragment += "；该句更像背景引用，而非本论文成果"
            parts.append(fragment + "。")
        elif attribution.get("method") == "sentence_occlusion":
            parts.append("遮挡分析未找到明显支撑句，模型的判断缺乏可定位的文本依据。")
        elif attribution.get("method") == "tfidf_linear_attribution":
            parts.append("归因未定位到正向特征句，常见于模型词表外的语言（如中文摘要）。")
        if attribution.get("error"):
            parts.append("句子级归因计算失败，理由退化为规则证据。")

    if credited_level is not None:
        parts.append(
            f"标准证据层采信 TRL {credited_level}（{_level_name(credited_level)}）"
            f"；规则后端为 {rule_level if rule_level is not None else '无证据'}"
            f"，模型为 {model_level if model_level is not None else '未给出'}。"
        )
    elif rule_level is None:
        parts.append("规则后端未找到足以支持 TRL 1–9 的已完成证据。")
    elif model_level is None:
        parts.append(f"规则后端判为 TRL {rule_level}，模型本次弃权，需人工复核。")
    elif int(model_level) == int(rule_level):
        parts.append(f"规则后端同判 TRL {rule_level}，逐句引文见 evidence。")
    else:
        parts.append(f"规则后端判为 TRL {rule_level}，与模型不一致，建议人工复核。")

    if attribution:
        parts.append(
            "说明：归因只反映分类器关注点，不能替代原文证据；输入仅为标题和摘要，"
            "未核读全文、测试记录或专家意见，不构成正式 TRL 认证。"
        )
    else:
        parts.append(
            "说明：模型分数未校准，本身不构成原文证据；输入仅为标题和摘要，"
            "未核读全文、测试记录或专家意见，不构成正式 TRL 认证。"
        )
    return "".join(parts)
