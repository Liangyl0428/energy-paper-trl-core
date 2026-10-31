"""Explainability contract: quotes stay auditable, attributions stay labelled."""
from __future__ import annotations

import pytest

from energy_paper_trl.api import analyze
from energy_paper_trl.explain import (
    compose_rationale,
    disagreement_summary,
    explain_linear,
    explain_occlusion,
)
from energy_paper_trl.model import SciBERTAdapter, resolve_label_map
from energy_paper_trl.rules import RuleTRLAnalyzer
from energy_paper_trl.segmentation import sentences

LAB = [
    "laboratory experiments validated the converter",
    "laboratory tests validated the prototype",
    "experimental validation in the laboratory",
    "test results validated the design experimentally",
]
CONCEPT = [
    "we propose a conceptual application framework",
    "a concept for future applications is proposed",
    "proposed application concept without experiments",
    "concept proposal for application",
]


def _toy_pipeline():
    pytest.importorskip("sklearn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LinearSVC(C=10, dual="auto", random_state=0)),
        ]
    )
    pipeline.fit(LAB + CONCEPT, [4] * len(LAB) + [2] * len(CONCEPT))
    return pipeline


def _mask(text: str) -> str:
    return text.replace("TRL 4", "readiness stage")


def test_sentence_spans_are_exact():
    text = "  First sentence.  Second one!  "
    spans = sentences(text)
    assert [s.text for s in spans] == ["First sentence", "Second one"]
    for span in spans:
        assert text[span.start:span.end] == span.text


def test_rule_evidence_offsets_recover_the_quote():
    abstract = (
        "We developed a converter. Laboratory experiments validated the proposed "
        "voltage control and test results met the target."
    )
    out = RuleTRLAnalyzer().analyze("Grid converter", abstract)
    assert out["level"] == 4
    full = f"Grid converter. {abstract}"
    assert out["evidence"]
    for item in out["evidence"]:
        source = full[item["start"]:item["end"]]
        shown = item["quote"].rstrip("…")
        assert source.startswith(shown)
        assert item["truncated"] is (len(source) > len(item["quote"]))


def test_rule_matching_uses_word_boundaries():
    # "test" must not match "Latest"; this used to be graded TRL 4.
    out = RuleTRLAnalyzer().analyze(
        "Latest advances in grid converters",
        "We review the latest results on converter design and the proposed methods.",
    )
    assert out["level"] != 4


def test_no_doubt_is_not_treated_as_negation():
    out = RuleTRLAnalyzer().analyze(
        "A controller",
        "There is no doubt that the laboratory test validated the design, and results show 99% accuracy.",
    )
    assert out["level"] == 4


def test_compared_with_keeps_the_result_clause():
    out = RuleTRLAnalyzer().analyze(
        "A converter",
        "The proposed converter was tested in the laboratory. Compared with the conventional "
        "scheme, efficiency improved by 3% and results met the target.",
    )
    assert out["level"] == 4


def test_component_keyword_alone_does_not_credits_level5():
    # "representative environment" alone used to satisfy level 5 via the old
    # group logic; without a component/subsystem core term it must not.
    out = RuleTRLAnalyzer().analyze(
        "Building energy optimisation",
        "An integrated cost-benefit analysis under a representative environment improved savings.",
    )
    assert out["level"] != 5


def test_level5_requires_a_component_core_plus_support():
    out = RuleTRLAnalyzer().analyze(
        "Converter module",
        "The key component was tested in a representative environment and integration was validated.",
    )
    assert out["level"] == 5


def test_linear_attribution_ranks_the_validation_sentence():
    pipeline = _toy_pipeline()
    text = (
        "We propose a concept for an application. "
        "Laboratory experiments validated the converter prototype."
    )
    out = explain_linear(pipeline, text, predicted_level=4)
    assert out["method"] == "tfidf_linear_attribution"
    assert out["not_evidence"] is True
    assert out["predicted_level"] == 4
    top = out["sentences"][0]
    assert top["quote"].startswith("Laboratory")
    assert top["start"] == text.index("Laboratory")
    assert text[top["start"]:top["end"]] == top["quote"]
    assert top["terms"] and top["share"] == pytest.approx(1.0, abs=0.5)


def test_linear_attribution_quotes_original_text_and_masks_model_input():
    pipeline = _toy_pipeline()
    text = "Laboratory tests validated the prototype at TRL 4."
    out = explain_linear(pipeline, text, mask=_mask, predicted_level=4)
    quotes = [row["quote"] for row in out["sentences"]]
    assert quotes and "TRL 4" in quotes[0]  # quote from source, not from masked input
    assert out["masked_input_terms"] is True


def test_occlusion_finds_the_supporting_sentence():
    def probs(text: str) -> list[float]:
        hit = 1.0 if "hidden evidence" in text.casefold() else 0.0
        return [1.0 - (0.2 + 0.7 * hit), 0.2 + 0.7 * hit]

    text = "Background sentence. Hidden evidence here. Another statement."
    out = explain_occlusion(probs, text, 1)
    assert out["method"] == "sentence_occlusion"
    assert out["target"]["index"] == 1
    assert out["sentences"], "the removed evidence sentence should be reported"
    assert "Hidden evidence" in out["sentences"][0]["quote"]
    assert out["sentences"][0]["drop"] > 0.5


def test_occlusion_refuses_tiny_inputs():
    out = explain_occlusion(lambda text: [0.5, 0.5], "Only one sentence.", 1)
    assert out["sentences"] == []
    assert out["method"] == "sentence_occlusion"


def test_attribution_marks_whether_the_quote_is_title_or_abstract():
    pipeline = _toy_pipeline()
    title = "Grid converter prototype"
    text = f"{title}\nLaboratory experiments validated the converter prototype."
    out = explain_linear(pipeline, text, predicted_level=4, title_chars=len(title))
    origins = {row["origin"] for row in out["sentences"]}
    assert "title" in origins and "abstract" in origins


def test_rationale_warns_when_scibert_is_unsure():
    text = compose_rationale(
        backend="scibert",
        model_level=8,
        rule_level=6,
        prediction={
            "score": 0.235,
            "alternatives": [
                {"level": 8, "score": 0.235},
                {"level": 5, "score": 0.173},
            ],
        },
        attribution=None,
    )
    assert "并不确定" in text
    assert "次优为 TRL 5" in text


def test_scibert_level_mapping_requires_11_class_head():
    adapter = object.__new__(SciBERTAdapter)
    adapter.id2label = {}
    adapter.num_labels = 11
    assert adapter.level_for_index(0) == (1, "1")
    assert adapter.level_for_index(3) == (4, "4")
    assert adapter.level_for_index(9) == (None, "not_applicable")
    assert adapter.level_for_index(10) == (None, "insufficient_evidence")

    adapter.num_labels = 9  # legacy paper_trl_external_20261023 head: rejected
    with pytest.raises(ValueError, match="11-class"):
        adapter.level_for_index(0)

    adapter.num_labels = 0
    adapter.id2label = {0: "LABEL_3"}
    with pytest.raises(ValueError):
        adapter.level_for_index(0)  # generic LABEL_n at an unknown width: refuse


def test_label_map_requires_the_full_11_class_schema():
    generic = {i: f"LABEL_{i}" for i in range(11)}
    assert resolve_label_map(generic, 11)[0] == (1, "1")
    assert resolve_label_map(generic, 11)[9] == (None, "not_applicable")
    assert resolve_label_map(generic, 11)[10] == (None, "insufficient_evidence")

    # an explicitly named head is trusted even if it is not index+1
    named = {
        0: "9", 1: "4", 2: "1", 3: "2", 4: "3",
        5: "5", 6: "6", 7: "7", 8: "8",
        9: "not_applicable", 10: "insufficient_evidence",
    }
    mapping = resolve_label_map(named, 11)
    assert mapping[0] == (9, "9") and mapping[1] == (4, "4")
    assert mapping[9] == (None, "not_applicable")

    # 9-class heads (levels only, no abstain labels) are no longer supported
    with pytest.raises(ValueError, match="11-class"):
        resolve_label_map({i: f"LABEL_{i}" for i in range(9)}, 9)
    with pytest.raises(ValueError, match="11-class"):
        resolve_label_map({i: str(i + 1) for i in range(9)}, 9)
    with pytest.raises(ValueError):
        resolve_label_map({0: "4", 1: "4"}, 2)  # duplicate levels
    with pytest.raises(ValueError):
        resolve_label_map({0: "LABEL_0", 1: "4"}, 2)  # mixed generic/semantic
    with pytest.raises(ValueError):
        resolve_label_map({}, 7)  # unknown width: refuse instead of guessing


def test_current_release_head_width_is_11():
    from energy_paper_trl.model import CURRENT_HEAD_WIDTH, SUPPORTED_HEAD_WIDTHS

    assert CURRENT_HEAD_WIDTH == 11
    assert SUPPORTED_HEAD_WIDTHS == (11,)
    mapping = resolve_label_map({}, CURRENT_HEAD_WIDTH)
    assert mapping[0] == (1, "1")
    assert mapping[9] == (None, "not_applicable")
    assert mapping[10] == (None, "insufficient_evidence")


def test_svm_adapter_rejects_nine_class_head():
    from energy_paper_trl.model import SVMAdapter

    with pytest.raises(ValueError, match="11-class"):
        SVMAdapter.validate_classes([str(i) for i in range(1, 10)])
    SVMAdapter.validate_classes(
        [str(i) for i in range(1, 10)] + ["not_applicable", "insufficient_evidence"]
    )


def test_scibert_predict_reports_head_width():
    adapter = object.__new__(SciBERTAdapter)
    adapter.id2label = {}
    adapter.num_labels = 11
    adapter.label_map = resolve_label_map({}, 11)
    adapter.max_length = 512
    adapter.model_name = "balanced_scibert_20261023"
    adapter.model_dir = "/tmp/model"
    adapter._probs = lambda text: [0.1] * 11

    class _Tok:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    adapter.tokenizer = _Tok()
    out = adapter.predict("title", "abstract")
    assert out["head_width"] == 11
    assert out["num_labels"] == 11
    assert out["level"] == 1  # all probabilities tie; argmax picks index 0 = TRL 1


def _string_label_pipeline():
    """Mimic the balanced 11-way SVM: string labels, including abstain classes."""
    pytest.importorskip("sklearn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    lab = [
        "laboratory experiments validated the converter",
        "laboratory tests validated the prototype",
        "experimental validation in the laboratory",
    ]
    concept = [
        "we propose a conceptual application framework",
        "a concept for future applications is proposed",
        "proposed application concept without experiments",
    ]
    abstain = [
        "this review surveys previous commercial equipment",
        "background of prior installations is compared",
        "existing commercial equipment is described",
    ]
    pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LinearSVC(C=10, dual="auto", random_state=0)),
        ]
    )
    pipeline.fit(lab + concept + abstain, ["4"] * 3 + ["2"] * 3 + ["insufficient_evidence"] * 3)
    return pipeline


def test_svm_adapter_accepts_string_labels_and_abstains():
    from energy_paper_trl.model import SVMAdapter

    adapter = object.__new__(SVMAdapter)
    adapter.model = _string_label_pipeline()

    review = adapter.predict(
        "Review of installations",
        "This review surveys previous commercial equipment. Background of prior installations is compared.",
    )
    assert review["level"] is None
    assert review["level_label"] == "insufficient_evidence"

    lab = adapter.predict(
        "Grid converter",
        "Laboratory experiments validated the converter.",
    )
    assert lab["level"] == 4
    assert lab["level_label"] == "4"
    # head_width mirrors clf.classes_ (this toy head has 3 classes: "2","4",abstain)
    assert lab["head_width"] == lab["classes"] == 3


def test_linear_attribution_handles_abstain_classes():
    pipeline = _string_label_pipeline()
    out = explain_linear(
        pipeline,
        "This review surveys previous commercial equipment. Background of prior installations is compared.",
    )
    assert out["predicted_level"] is None
    assert out["predicted_class"] in {"insufficient_evidence", "not_applicable"}
    assert out["sentences"], "attribution must still rank sentences for an abstain prediction"


class FakeModel:
    def __init__(self, level: int, *, fail: bool = False):
        self.level = level
        self.fail = fail

    def predict(self, title: str, abstract: str = ""):
        return {
            "level": self.level,
            "score": 0.5,
            "score_calibrated": False,
            "model": "fake",
            "evidence": [],
        }

    def explain(self, title: str, abstract: str = "", *, predicted_level=None):
        if self.fail:
            raise RuntimeError("boom")
        return {
            "method": "sentence_occlusion",
            "not_evidence": True,
            "sentences": [
                {
                    "quote": "Laboratory experiments validated the design",
                    "start": 0,
                    "end": 45,
                    "context": None,
                }
            ],
        }


LAB_ABSTRACT = (
    "We developed a converter. Laboratory experiments validated the proposed "
    "voltage control and test results met the target."
)


def test_api_merges_attribution_and_reports_one_credited_level():
    out = analyze("Grid converter", LAB_ABSTRACT, backend="svm", model=FakeModel(4))
    assert out["level"] == 4 and out["level_source"] == "evidence"
    assert out["rule_level"] == 4
    assert out["levels"] == {"evidence": 4, "rule": 4, "model": 4}
    assert out["attribution"]["not_evidence"] is True
    assert out["disagreement"]["agree"] is True
    assert "标准证据层采信 TRL 4" in out["rationale"]
    assert "Laboratory experiments validated the design" in out["rationale"]
    assert out["rule_rationale"] != out["rationale"]


def test_api_flags_disagreement_instead_of_hiding_it():
    out = analyze("Grid converter", LAB_ABSTRACT, backend="svm", model=FakeModel(2))
    assert out["level"] == 4 and out["levels"]["model"] == 2
    assert out["disagreement"]["agree"] is False
    assert out["disagreement"]["abs_gap"] == 2
    assert out["disagreement"]["action"] == "manual_review"
    assert "标准证据层采信 TRL 4" in out["rationale"]
    assert "模型判为 TRL 2" in out["rationale"]
    assert out["uncertainty"]["band"] in {"medium", "high"}


def test_api_survives_a_failing_explainer():
    out = analyze("Grid converter", LAB_ABSTRACT, backend="svm", model=FakeModel(4, fail=True))
    assert out["level"] == 4
    assert out["attribution"]["error"].startswith("RuntimeError")
    assert "归因计算失败" in out["rationale"]


def test_api_can_skip_explanations():
    out = analyze("Grid converter", LAB_ABSTRACT, backend="svm", model=FakeModel(4), explain=False)
    assert out["attribution"] is None
    assert out["model_prediction"]["level"] == 4
    assert "归因只反映" not in out["rationale"]
    assert "模型分数未校准" in out["rationale"]


def test_disagreement_summary_edge_cases():
    assert disagreement_summary(None, None) is None
    partial = disagreement_summary(None, 5)
    assert partial["action"] == "manual_review" and partial["abs_gap"] is None
    agree = disagreement_summary(3, 3)
    assert agree["agree"] is True and agree["action"] is None


class AbstainModel(FakeModel):
    def predict(self, title: str, abstract: str = ""):
        return {
            "level": None,
            "level_label": "insufficient_evidence",
            "score": 0.34,
            "score_calibrated": False,
            "model": "fake-balanced-scibert",
            "alternatives": [
                {"index": 10, "level": None, "label": "insufficient_evidence", "score": 0.34},
                {"index": 6, "level": 7, "label": "7", "score": 0.21},
            ],
            "evidence": [],
        }


def test_model_abstention_is_not_reported_as_a_level():
    out = analyze(
        "Passive solar heating",
        "This paper reviews previous work on solar heating and existing commercial equipment.",
        backend="scibert",
        model=AbstainModel(None),
    )
    assert out["level"] is None
    assert out["status"] in {"not_applicable", "insufficient_evidence"}
    assert out["levels"]["model"] is None
    assert out["model_prediction"]["level_label"] == "insufficient_evidence"
    assert "不属于 TRL 1–9" in out["rationale"]


def test_rule_level_survives_model_abstention():
    out = analyze("Grid converter", LAB_ABSTRACT, backend="scibert", model=AbstainModel(None))
    assert out["level"] == 4 and out["level_source"] == "evidence"
    assert out["levels"] == {"evidence": 4, "rule": 4, "model": None}
    assert "模型为 未给出" in out["rationale"]
    assert out["disagreement"]["abs_gap"] is None


def test_one_level_contract_holds_for_every_backend():
    for backend, model in (("rules", None), ("svm", FakeModel(7)), ("scibert", AbstainModel(None))):
        out = analyze("Grid converter", LAB_ABSTRACT, backend=backend, model=model)
        alignment = out["standard_alignment"]
        assert "level" not in alignment, "the standard view must not expose a second level"
        assert out["level"] == alignment["evidence_level"] == out["levels"]["evidence"]
        assert out["status"] == alignment["evaluation_status"]


def test_full_gates_are_opt_in():
    small = analyze("Grid converter", LAB_ABSTRACT)
    assert "gates" not in small["standard_alignment"]
    assert len(small["standard_alignment"]["gates_summary"]) == 9
    full = analyze("Grid converter", LAB_ABSTRACT, include_gates=True)
    assert len(full["standard_alignment"]["gates"]) == 9
    assert "attributes" in full["standard_alignment"]["gates"][0]


def test_compose_rationale_without_attribution():
    text = compose_rationale(
        backend="svm",
        model_level=None,
        rule_level=None,
        prediction={"score": 0.1},
        attribution=None,
    )
    assert "未给出可用的 TRL 等级" in text
    assert "规则后端未找到" in text


def test_max_length_is_not_the_generation_default(tmp_path):
    # PretrainedConfig exposes max_length=20 even for 512-token checkpoints; the
    # adapter must not truncate abstracts to 20 tokens because of it.
    from energy_paper_trl.model import SciBERTAdapter

    (tmp_path / "config.json").write_text('{"model_type": "bert"}')
    assert SciBERTAdapter._resolve_max_length(tmp_path, None) == 512
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 256}')
    assert SciBERTAdapter._resolve_max_length(tmp_path, None) == 256
    assert SciBERTAdapter._resolve_max_length(tmp_path, 128) == 128
    with pytest.raises(ValueError):
        SciBERTAdapter._resolve_max_length(tmp_path, 20)
