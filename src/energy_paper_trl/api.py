from __future__ import annotations

from typing import Any

from .explain import compose_rationale, disagreement_summary
from .rules import RuleTRLAnalyzer
from .standard import align_to_standard

STANDARD_CHOICES = {'qgdw12566', None, 'none'}


def analyze(
    title: str,
    abstract: str = "",
    *,
    doc_id: str | None = None,
    backend: str = "rules",
    model: Any = None,
    explain: bool = True,
    standard: str | None = "qgdw12566",
    include_gates: bool = False,
) -> dict[str, Any]:
    """Analyze one paper and return exactly one credited ``level``.

    Public contract
    ---------------
    ``level`` / ``level_name`` / ``level_source``
        The single credited level. With the standard layer enabled (default) it
        is evidence-first: ``level = max{l : Gate(l)}``, i.e. the same value as
        ``standard_alignment.evidence_level``. Without the standard layer it is
        the rule level, or the model level when the rules find nothing.
    ``status``
        Semantic status of the result: the standard's ``evaluation_status``
        (``public_evidence_preliminary`` / ``insufficient_evidence`` /
        ``not_applicable`` / ``evidence_conflict``) when aligned, otherwise the
        rule/model status.
    ``levels``
        All three opinions side by side (``evidence`` / ``rule`` / ``model``) so
        nothing has to be inferred from prose.
    ``standard_alignment``
        Full standard view. It reports ``evidence_level`` and never a second
        ``level``; the per-attribute ``gates`` audit is only included when
        ``include_gates=True`` (otherwise ``gates_summary`` keeps the payload
        small).
    """
    if standard not in STANDARD_CHOICES:
        raise ValueError("standard must be 'qgdw12566' or None")
    if standard == 'none':
        standard = None

    rule = RuleTRLAnalyzer().analyze(title, abstract, doc_id=doc_id)
    rule_level = rule["level"]
    prediction: dict[str, Any] | None = None
    attribution: dict[str, Any] | None = None

    if backend not in {"rules", "svm", "scibert"}:
        raise ValueError("backend must be rules, svm, or scibert")
    if backend != "rules":
        if model is None:
            raise ValueError("model adapter is required for backend=" + backend)
        prediction = model.predict(title, abstract)
        if explain:
            attribution = _safe_explain(model, title, abstract, prediction)

    model_level = prediction.get("level") if prediction else None
    base: dict[str, Any] = dict(rule)
    if prediction is not None:
        base.update({
            "model_prediction": prediction,
            "attribution": attribution,
            "disagreement": disagreement_summary(rule_level, model_level),
        })

    # The one credited level: evidence-first when the standard is on.
    if standard:
        alignment = align_to_standard(
            base, title, abstract, include_gates=include_gates
        )
        base["standard_alignment"] = alignment
        level = alignment["evidence_level"]
        level_source = "evidence" if level is not None else None
        level_name = alignment["level_name"]
        status = alignment["evaluation_status"]
        evidence_level = alignment["evidence_level"]
    else:
        evidence_level = None
        if rule_level is not None:
            level, level_source, level_name = rule_level, "rules", rule["level_name"]
        elif model_level is not None:
            level = model_level
            level_source = "model"
            level_name = _level_name(model_level)
        else:
            level = level_source = level_name = None
        status = rule["status"]
        if prediction is not None:
            if rule_level is not None:
                status = "rule_and_model_estimate"
            elif model_level is not None:
                status = "model_estimate_only"
            else:
                status = "insufficient_text_evidence"

    base.update({
        "level": level,
        "level_name": level_name,
        "level_source": level_source,
        "levels": {"evidence": evidence_level, "rule": rule_level, "model": model_level},
        "rule_level": rule_level,
        "rule_status": rule["status"],
        "rule_rationale": rule["rationale"],
        "status": status,
        "uncertainty": _uncertainty(
            evidence_level=evidence_level,
            rule_level=rule_level,
            model_level=model_level,
            prediction=prediction,
            attribution=attribution,
            has_abstract=bool(str(abstract or "").strip()),
        ),
    })
    if prediction is not None:
        base["rationale"] = compose_rationale(
            backend=backend,
            model_level=model_level,
            rule_level=rule_level,
            prediction=prediction,
            attribution=attribution,
            credited_level=level,
        )
    return base


def _level_name(level: int | None) -> str | None:
    from .rules import LEVELS

    return LEVELS[level][0] if isinstance(level, int) and level in LEVELS else None


def _uncertainty(
    *,
    evidence_level: int | None,
    rule_level: int | None,
    model_level: int | None,
    prediction: dict[str, Any] | None,
    attribution: dict[str, Any] | None,
    has_abstract: bool,
) -> dict[str, Any]:
    """Coarse band driven by disagreement, not only by "is there evidence"."""
    reasons: list[str] = ["仅使用标题和摘要", "未核读全文、测试记录或专家意见"]
    distress = 0
    if not has_abstract:
        distress += 2
        reasons.append("缺少摘要，仅有标题")
    if evidence_level is None:
        distress += 2
        reasons.append("逐字证据不足以支撑任一等级")
    opinions = [v for v in (evidence_level, rule_level, model_level) if v is not None]
    if len(opinions) > 1 and len(set(opinions)) > 1:
        distress += 1
        reasons.append(f"各层结论不一致：{sorted(set(opinions))}")
    if prediction is not None:
        label = prediction.get("level_label")
        if label in {"not_applicable", "insufficient_evidence"}:
            distress += 1
            reasons.append("分类器弃权")
        elif isinstance(prediction.get("score"), (int, float)) and not prediction.get("score_calibrated", False):
            reasons.append("模型分数未校准")
    if attribution and attribution.get("error"):
        distress += 1
        reasons.append("句子级归因计算失败")
    if distress >= 3:
        band = "high"
    elif distress == 0:
        band = "low"
    else:
        band = "medium"
    return {"band": band, "reasons": reasons, "distress_score": distress}


def _safe_explain(
    model: Any, title: str, abstract: str, prediction: dict[str, Any]
) -> dict[str, Any] | None:
    """Explanations must never hide the classification result."""
    explainer = getattr(model, "explain", None)
    if not callable(explainer):
        return None
    try:
        return explainer(title, abstract, predicted_level=prediction.get("level"))
    except Exception as exc:  # attribution is best-effort; the label is not
        return {
            "method": None,
            "sentences": [],
            "not_evidence": True,
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "note": "归因计算失败；分类结果仍然有效，理由退化为规则证据。",
        }
