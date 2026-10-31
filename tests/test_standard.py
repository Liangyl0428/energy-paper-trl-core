"""Alignment with Q/GDW 12566—2025 表1: 11-class output stays, standard view is added."""
from __future__ import annotations

import pytest

from energy_paper_trl.api import analyze
from energy_paper_trl.standard import align_to_standard, evaluate_gates, load_standard

LAB_TEXT = (
    "The key technology of the modular multilevel converter is studied. "
    "Laboratory experiments and simulation were carried out on the converter. "
    "The experimental results show that the control strategy was validated."
)
REVIEW_TEXT = (
    "This paper reviews previous work on grid converters. "
    "Existing commercial equipment is described for comparison."
)


def _prediction(level, label=None):
    return {
        "level": level,
        "level_label": label if label is not None else (str(level) if level else None),
        "score": 0.5,
        "score_calibrated": False,
        "model": "fake",
        "evidence": [],
    }


class FakeModel:
    def __init__(self, level, label=None):
        self._prediction = _prediction(level, label)

    def predict(self, title: str, abstract: str = ""):
        return dict(self._prediction)


def _gate(gates, level):
    return next(item for item in gates if item["level"] == level)


def test_spec_matches_the_standard_table():
    spec = load_standard()
    assert spec["scale_version"] == "QGDW_12566_2025"
    assert spec["standard_id"] == "Q/GDW 12566—2025"
    assert spec["levels"]["3"]["definition"] == "关键功能通过理论可行性验证"
    assert spec["levels"]["4"]["definition"] == "系统关键技术得到实验环境验证"
    assert spec["levels"]["8"]["definition"] == "产品挂网试运行"
    assert "实验室环境" in spec["levels"]["4"]["definition_note"]
    assert spec["source"]["pdf_page"] == 8


def test_level4_gate_is_met_by_lab_simulation_with_results():
    gates, skipped = evaluate_gates("Grid converter", LAB_TEXT)
    assert _gate(gates, 4)["status"] == "met"
    assert _gate(gates, 4)["missing_attributes"] == []
    assert _gate(gates, 5)["status"] != "met"
    assert skipped == []


def test_future_lab_work_does_not_pass_the_gate():
    gates, skipped = evaluate_gates(
        "Grid converter",
        "Future laboratory experiments will validate the key technology of the converter.",
    )
    assert _gate(gates, 4)["status"] != "met"
    assert skipped and skipped[0]["reason"] == "negated_or_future"


def test_level3_requires_design_plus_theoretical_analysis():
    gates, _ = evaluate_gates(
        "Converter control",
        "We determine the technical route and complete the design scheme. "
        "A theoretical analysis confirms the feasibility of the critical function.",
    )
    assert _gate(gates, 3)["status"] == "met"


def test_model_over_grade_is_reported_against_the_standard():
    result = {"model_prediction": _prediction(8)}
    aligned = align_to_standard(result, "Grid converter", LAB_TEXT)
    assert aligned["evaluation_status"] == "public_evidence_preliminary"
    assert aligned["trl_model"] == 8
    assert aligned["trl_evidence"] == 4
    assert aligned["evidence_level"] == 4
    assert aligned["level_name"] == "系统关键技术得到实验环境验证"
    assert aligned["model_vs_evidence"]["gap"] == 4
    assert aligned["model_vs_evidence"]["verdict"] == "model_over_grades"
    assert aligned["next_level_gap"]["level"] == 5
    assert aligned["next_level_gap"]["missing_attributes"]
    assert aligned["trl_final"] is None
    assert aligned["formal_certified_level"] is None


def test_review_without_evidence_is_not_applicable():
    aligned = align_to_standard(
        {"model_prediction": _prediction(None, "not_applicable")}, "A review", REVIEW_TEXT
    )
    assert aligned["evaluation_status"] == "not_applicable"
    assert aligned["evidence_level"] is None
    assert aligned["not_applicable_basis"]["requires"]
    assert aligned["object_check"]["is_evaluation_object"] is False


def test_review_with_own_simulation_stays_an_evaluation_object():
    text = (
        "This paper presents a decoupled control for grid-connected solar PV systems. "
        "Simulations in various conditions are conducted to prove the effectiveness of "
        "the proposed control strategies and the results show improved stability."
    )
    aligned = align_to_standard({"model_prediction": _prediction(4)}, "Review of Grid-Connected PV", text)
    assert aligned["object_check"]["review_like"] is True
    assert aligned["object_check"]["own_work_markers"] is True
    assert aligned["evaluation_status"] == "public_evidence_preliminary"
    assert aligned["evidence_level"] == 4


def test_review_with_a_model_level_is_flagged_not_promoted():
    aligned = align_to_standard({"model_prediction": _prediction(4)}, "A review", REVIEW_TEXT)
    assert aligned["evaluation_status"] == "not_applicable"
    assert aligned["evidence_level"] is None
    assert aligned["conflict"] is not None
    assert aligned["model_vs_evidence"]["verdict"] == "not_an_evaluation_object"
    assert aligned["next_level_gap"] is None


def test_model_level_without_quotes_never_becomes_the_level():
    aligned = align_to_standard(
        {"model_prediction": _prediction(4)},
        "Mesoporous carbon",
        "Graphitic mesoporous carbon enhances corrosion stability while preserving high performance.",
    )
    assert aligned["evaluation_status"] == "insufficient_evidence"
    assert aligned["evidence_level"] is None
    assert aligned["trl_model"] == 4
    assert aligned["model_vs_evidence"]["verdict"] == "model_unsupported_by_quotes"


def test_abstain_classes_are_preserved_in_the_alignment():
    for label in ("not_applicable", "insufficient_evidence"):
        aligned = align_to_standard(
            {"model_prediction": _prediction(None, label)}, "A review", REVIEW_TEXT
        )
        assert aligned["model_class"] == label
        assert aligned["trl_model"] is None
        assert aligned["evidence_level"] is None


def test_api_attaches_alignment_and_keeps_the_eleven_class_output():
    out = analyze("Grid converter", LAB_TEXT, backend="scibert", model=FakeModel(8))
    assert out["model_prediction"]["level"] == 8
    alignment = out["standard_alignment"]
    assert alignment["scale_version"] == "QGDW_12566_2025"
    assert alignment["evidence_level"] == 4
    assert alignment["trl_model"] == 8


def test_api_rules_backend_can_be_aligned_too():
    out = analyze("Grid converter", LAB_TEXT, backend="rules")
    assert out["standard_alignment"]["evidence_level"] == 4
    assert out["standard_alignment"]["model_class"] is None
    assert out["level"] == 4


def test_api_can_skip_the_standard_view():
    out = analyze("Grid converter", LAB_TEXT, backend="rules", standard=None)
    assert "standard_alignment" not in out


def test_api_rejects_unknown_standard():
    with pytest.raises(ValueError):
        analyze("Grid converter", LAB_TEXT, backend="rules", standard="nasa")


def test_gate_quotes_keep_exact_spans_when_clipped():
    long_sentence = (
        "The proposed converter was validated in the laboratory under a representative "
        "environment with a comprehensive set of tests, and results demonstrated that "
        "efficiency, capacity and retention all met the requirements under nominal and "
        "fault conditions for a prolonged period of time while the controller remained stable."
    )
    title = "Long"
    full = f"{title}. {long_sentence}"
    gates, skipped = evaluate_gates(title, long_sentence)
    checked = 0
    for gate in gates:
        for attribute in gate["attributes"].values():
            for quote in attribute["quotes"]:
                source = full[quote["start"]:quote["end"]]
                assert source.startswith(quote["quote"].rstrip("…"))
                assert quote["truncated"] is (len(source) > len(quote["quote"]))
                checked += 1
    assert checked > 0
