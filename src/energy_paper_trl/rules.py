"""Conservative bilingual evidence rules for a single paper.

The result is a text-supported stage estimate. It is not a formal certification
and never infers a completed test from future, negated, or background language.

Matching rules (v2, 2026-10-31):

* every keyword goes through :func:`textmatch.term_pattern`, so "test" no longer
  matches "latest" and the rule layer agrees with the gate/attribution layers;
* each level has *core* terms (the defining object of that level) and *support*
  terms (completion evidence). A level is only credited when its core matches and
  its support/completion evidence is present, which stops a single side keyword
  ("representative environment") from promoting a paper;
* negation/background filtering happens per clause, so
  "Compared with X, efficiency improved 3%" keeps the result clause, and
  "no doubt that the test validated" is not treated as negated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .segmentation import sentences
from .textmatch import clean_clauses, clip_span, term_pattern

# level -> (name, core terms, support terms)
# core   = the object/state that defines the level (component, prototype, field test ...)
# support = completed-work language that has to accompany high-stage core terms
LEVELS: dict[int, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    1: ("基本原理",
        # "mechanism" alone is far too common ("reaction mechanism") to mean a
        # discovered basic principle; keep the explicit principle phrases only.
        ("basic principle", "fundamental principle", "first principle",
         "first-principle", "principle", "principles",
         "基本原理", "原理", "机理研究"),
        ()),
    2: ("应用设想",
        ("application concept", "conceptual design", "application scenario", "concept",
         "propose", "proposed", "proposal", "设想", "构想", "应用设想", "提出"),
        ()),
    3: ("理论可行性验证",
        ("theoretical", "theoretical analysis", "feasibility", "analytical model",
         "mathematical model", "first principle", "first-principle", "governing equation",
         "理论分析", "理论可行性", "可行性", "理论"),
        ("design", "framework", "methodology", "设计", "方案设计", "技术路线")),
    4: ("实验环境验证",
        ("laboratory", "laboratories", "lab", "lab-scale", "experiment", "experiments",
         "experimental", "experimentally", "simulation", "simulations", "simulated",
         "test bench", "test rig", "benchtop", "hardware-in-the-loop", "in situ",
         "实验", "实验室", "仿真", "试验台", "测试"),
        # Completed-work markers. Past-tense activity plus a real result word is the
        # minimum for "已完成的实验室工作"; the stricter result-only requirement lives
        # in the standard gate layer (config/qgdw_12566_2025.json, level 4).
        ("validated", "validates", "verified", "verifies", "results", "result",
         "measured", "measurement", "demonstrated", "confirms", "confirmed",
         "synthesized", "prepared", "fabricated", "manufactured", "assembled",
         "characterized", "tested", "observed", "obtained", "investigated",
         "evaluated", "compared", "achieved", "improved", "reduced", "exhibited",
         "showed", "shown", "reported", "found", "reached", "recorded",
         "validate", "affirmed", "successful", "successfully", "satisfactory",
         "验证", "结果", "达标", "满足要求", "合成", "制备", "测试", "表征", "测量")),
    5: ("部件功能测试",
        # Bare "component" appears in "blend components", "system components" and
        # promoted material papers to TRL 5; require an explicit part-of-system noun.
        ("key component", "key components", "subsystem", "subsystems", "sub-assembly",
         "module", "modules", "关键部件", "分系统", "子系统", "部件", "模块"),
        ("representative environment", "simulated operating environment",
         "simulated environment", "integration", "integrated", "assembly",
         "模拟使用环境", "模拟环境", "集成", "组装")),
    6: ("原型验证",
        ("prototype", "breadboard", "pilot", "pilot project", "demonstration",
         "原型样机", "原型", "中试", "示范"),
        ("tested", "validated", "verified", "demonstrated", "performance",
         "measured", "evaluated", "compared", "achieved", "investigated",
         "验证", "测试", "性能")),
    7: ("工程测试",
        ("engineering prototype", "engineering test", "field test", "field trial",
         "type test", "type tests", "type testing", "qualification test",
         "environmental test", "environmental tests", "qualification", "mil-std",
         "工程样机", "工程测试", "现场测试", "现场试验", "型式试验", "现场"),
        ("passed", "passes", "qualified", "completed", "affirmed", "successful",
         "successfully", "satisfactory", "verified", "validated", "viable",
         "通过", "合格", "完成")),
    8: ("产品试运行",
        ("trial operation", "trial run", "online trial", "commissioned", "commissioning",
         "pilot operation", "grid-connected trial", "试运行", "挂网", "并网投运",
         "上线试运行", "试运行阶段"),
        ("comprehensive test", "full test", "site acceptance test", "completed",
         "全面测试", "完成试运行", "完成")),
    9: ("产品应用",
        # Bare "deployed"/"commercial" mostly appears in "potentially deployed" or
        # "commercial products are compared"; require an operating-application phrase.
        ("in operation", "in service", "in commercial operation", "commercial operation",
         "commercially deployed", "mass production", "large-scale application",
         "commercial deployment", "商业化", "批量应用", "实际应用", "投入运行",
         "规模化应用"),
        ()),
}

NEGATIVE = re.compile(
    r"\b(?:not|no|never|without|future|will|planned?|plan to|proposes? to|"
    r"expected to|to be validated|no evidence|has not|have not|"
    r"does not|did not)\b"
    r"|尚未|未开展|未来|计划|拟|有待|尚无|并非|不涉及|后续|将开展|将进行|待验证|尚待|未验证",
    re.I,
)
BACKGROUND = re.compile(
    r"\b(?:previous work|previous studies|prior work|prior studies|background|"
    r"state of the art|commercial equipment|for example)\b"
    r"|已有研究|背景技术|例如",
    re.I,
)


@dataclass
class RuleTRLAnalyzer:
    """Analyze title + abstract and return exact quote evidence."""

    algorithm_version: str = "paper-trl-evidence-20261031-v2-rules"

    def analyze(self, title: str, abstract: str = "", doc_id: str | None = None) -> dict[str, Any]:
        title = str(title or "").strip()
        abstract = str(abstract or "").strip()
        full = f"{title}. {abstract}" if abstract else title
        evidence: dict[int, list[dict[str, Any]]] = {level: [] for level in LEVELS}

        for span in sentences(full):
            good = clean_clauses(span.text, NEGATIVE, BACKGROUND)
            if not good:
                continue
            quote = clip_span(full, span.start, span.end)
            for level, (_, core, support) in LEVELS.items():
                for kind, terms in (("core", core), ("support", support)):
                    hits = _hits(good, terms)
                    if not hits:
                        continue
                    if any(
                        item["start"] == span.start and item["kind"] == kind
                        for item in evidence[level]
                    ):
                        continue
                    evidence[level].append({
                        **quote.as_dict(),
                        "matched_terms": hits,
                        "kind": kind,
                    })

        def core(level: int) -> bool:
            return any(item["kind"] == "core" for item in evidence[level])

        def support(level: int) -> bool:
            return any(item["kind"] == "support" for item in evidence[level])

        def lower_core(level: int) -> bool:
            return any(core(candidate) for candidate in range(max(1, level - 2), level))

        def credited(level: int) -> bool:
            if not core(level):
                return False
            if level <= 2:
                return True
            if level <= 8:
                return support(level)
            return support(level) or lower_core(level)

        supported: int | None = None
        for level in range(9, 0, -1):
            if level == 3:
                # "已开展仿真/实验但无结果" 按协议最高记 TRL 3，而不是弃权。
                if core(3) or (core(4) and not support(4)):
                    supported = 3
                    break
                continue
            if credited(level):
                supported = level
                break

        selected = evidence[supported] if supported else []
        missing = [] if supported is None else [
            {"level": higher, "name": LEVELS[higher][0],
             "reason": "title/abstract contains no completed evidence for this higher stage"}
            for higher in range(supported + 1, 10)
        ]
        return {
            "doc_id": doc_id,
            "algorithm_version": self.algorithm_version,
            "standard_reference": "Q/GDW 12566—2025 project operationalization",
            "level": supported,
            "level_name": LEVELS[supported][0] if supported else None,
            "status": "text_supported_estimate" if supported else "insufficient_text_evidence",
            "scope": self._scope(full),
            "rationale": self._rationale(supported, selected),
            "evidence": selected[:8],
            "missing_higher_stage_evidence": missing,
            "uncertainty": {
                "band": "high" if not abstract or not supported else "medium",
                "reasons": ["仅使用标题和摘要", "未核读全文、测试记录或专家意见"],
                "matching": "word_boundary_terms + clause_level_negation_filter",
            },
            "formal_certified_level": None,
        }

    @staticmethod
    def _scope(text: str) -> str:
        low = text.casefold()
        if any(x in low for x in ("algorithm", "software", "controller", "模型", "算法", "控制器")):
            return "software"
        if any(x in low for x in ("material", "electrode", "catalyst", "材料", "电极", "催化剂")):
            return "material"
        if any(x in low for x in ("system", "plant", "grid", "device", "系统", "装置", "电网")):
            return "system"
        if any(x in low for x in ("component", "cell", "module", "部件", "电池", "模块")):
            return "component"
        return "unknown"

    @staticmethod
    def _rationale(level: int | None, evidence: list[dict[str, Any]]) -> str:
        if level is None:
            return "标题和摘要没有提供足以支持 TRL 1–9 的已完成证据；不能把提出方案或背景技术当成验证结果。"
        core_terms = [t for item in evidence if item["kind"] == "core" for t in item["matched_terms"]]
        quote = evidence[0]["quote"] if evidence else ""
        terms = "、".join(dict.fromkeys(core_terms))[:80]
        return (
            f"文本证据支持 TRL {level}（{LEVELS[level][0]}）。"
            f"核心特征：{terms}；依据：{quote}。"
            "更高阶段所需的工程、现场、上线或规模应用证据未在输入文本中确认。"
        )


def _hits(clauses: list[tuple[int, int, str]], terms: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for term in terms:
        pattern = term_pattern(term)
        if any(pattern.search(clause) for _, _, clause in clauses):
            found.append(term)
    return found
