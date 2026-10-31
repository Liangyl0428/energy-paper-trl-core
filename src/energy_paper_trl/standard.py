"""Align the 11-class classifier output with Q/GDW 12566—2025 (表1, 1–9 级).

The classifier keeps its own 11-way label space (TRL 1–9 plus
``not_applicable`` / ``insufficient_evidence``). This layer answers a different
question: *given the standard's definitions and the evidence in the text, what
level may be claimed, and which conditions are missing?*

It follows the project method document (§5.1 Gate definition):

    Gate(l)        = every applicable attribute of level l has completed evidence
    TRL_evidence   = max { l : Gate(l) }        (null when no level holds)
    TRL_model      = the classifier's suggestion (kept, never silently promoted)
    TRL_final      = null            (needs self-assessment + expert review, 标准 5.5)

``TRL_evidence`` is evidence-first: a model suggestion without matching quotes is
reported as a gap, not as the level. Gate keywords are the project's default
细则, not thresholds mandated by the standard; per-technology 细则 still require
expert configuration.

Quote contract: every quote carries ``start``/``end`` (offsets into
``f"{title}. {abstract}"``), ``truncated`` and the clipped ``quote``. The
unclipped source is always recoverable with ``full[start:end]``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rules import BACKGROUND, NEGATIVE
from .segmentation import sentences
from .textmatch import clean_clauses, clip_span, term_pattern

SPEC_ENV = 'ENERGY_PAPER_TRL_STANDARD'
ABSTAIN_CLASSES = {'not_applicable', 'insufficient_evidence'}
MAX_QUOTE_CHARS = 240

# Standard 1 范围: the object is a technology/technical carrier. Review, survey and
# perspective papers describe other people's work, so they are not an evaluation
# object unless the paper also reports its own validation. Chinese and English
# review markers are both required — the corpus is bilingual.
REVIEW_TITLE = re.compile(
    r'^\s*(?:a\s+|an\s+)?(?:review|survey|overview|state[-\s]of[-\s]the[-\s]art|perspectives?)\b'
    r'|综述|回顾|研究进展|进展|评述|述评|展望',
    re.I,
)
REVIEW_TEXT = re.compile(
    r'this\s+(?:paper|article|review)\s+(?:reviews|surveys|summari[sz]es|presents\s+a\s+review)'
    r'|\bthis\s+review\b'
    r'|we\s+review|literature\s+review|are\s+summari[sz]ed|recent\s+progress|in\s+this\s+review'
    r'|本文综述|本文回顾|对.{0,20}进行了综述|综述了|研究进展|文献综述',
    re.I,
)
OWN_WORK = re.compile(
    r'we\s+(?:propose|present|develop|design|fabricate|build|test|measure|conduct|demonstrate)'
    r'|this\s+paper\s+(?:proposes|presents|develops|introduces|reports)'
    r'|this\s+work\s+(?:proposes|presents|develops|reports)'
    r'|in\s+this\s+(?:study|work|paper)'
    r'|our\s+(?:results|findings|experiments|measurements)'
    r'|本文提出|本文设计|本文研制|本文构建|我们提出|本研究',
    re.I,
)


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    override = os.environ.get(SPEC_ENV)
    if override:
        paths.append(Path(override))
    package_root = Path(__file__).resolve().parent
    paths.append(package_root / 'config' / 'qgdw_12566_2025.json')
    # repository layout: <repo>/src/energy_paper_trl/standard.py -> <repo>/config
    paths.append(package_root.parents[1] / 'config' / 'qgdw_12566_2025.json')
    return paths


def load_standard(path: str | Path | None = None) -> dict[str, Any]:
    candidates = [Path(path)] if path else _candidate_paths()
    for candidate in candidates:
        if candidate.is_file():
            spec = json.loads(candidate.read_text())
            _validate_spec(spec)
            return spec
    searched = ', '.join(str(p) for p in candidates)
    raise FileNotFoundError(
        f'Q/GDW 12566 spec not found (searched: {searched}); '
        f'set {SPEC_ENV} to the JSON path'
    )


def _validate_spec(spec: dict[str, Any]) -> None:
    levels = spec.get('levels') or {}
    missing = [str(i) for i in range(1, 10) if str(i) not in levels]
    if missing:
        raise ValueError(f'standard spec is missing levels: {missing}')
    for key, level in levels.items():
        if not level.get('definition') or not level.get('definition_note'):
            raise ValueError(f'level {key} lacks definition/definition_note')
    if not spec.get('scale_version'):
        raise ValueError('standard spec lacks scale_version')


@dataclass
class GateHit:
    status: str
    hits: list[str] = field(default_factory=list)
    quotes: list[dict[str, Any]] = field(default_factory=list)


def _attribute_status(
    usable_clauses: list[tuple[int, int, str, str]],
    terms: list[str],
    full: str,
    max_quote_chars: int,
) -> GateHit:
    """Any-of keyword match inside clean clauses; quotes keep exact spans."""
    hits: list[str] = []
    quotes: list[dict[str, Any]] = []
    for sent_start, sent_end, _sentence, clause in usable_clauses:
        for term in terms:
            if not term_pattern(term).search(clause):
                continue
            if term not in hits:
                hits.append(term)
            if not any(q['start'] == sent_start for q in quotes):
                quote = clip_span(full, sent_start, sent_end, max_quote_chars)
                quotes.append({**quote.as_dict(), 'matched_term': term})
    status = 'met' if hits else 'not_met'
    return GateHit(status=status, hits=hits, quotes=quotes[:2])


def evaluate_gates(
    title: str,
    abstract: str,
    spec: dict[str, Any] | None = None,
    *,
    max_quote_chars: int = MAX_QUOTE_CHARS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(gates, skipped_sentences)``; see the module quote contract."""
    spec = spec or load_standard()
    title = str(title or '').strip()
    abstract = str(abstract or '').strip()
    full = f'{title}. {abstract}' if abstract else title

    usable_clauses: list[tuple[int, int, str, str]] = []
    skipped: list[dict[str, Any]] = []
    for span in sentences(full):
        good = clean_clauses(span.text, NEGATIVE, BACKGROUND)
        if not good:
            probe = span.text
            reason = 'negated_or_future' if NEGATIVE.search(probe) else 'background_or_other_study'
            quote = clip_span(full, span.start, span.end, max_quote_chars)
            skipped.append({**quote.as_dict(), 'reason': reason})
            continue
        for _lo, _hi, clause in good:
            usable_clauses.append((span.start, span.end, span.text, clause))

    gates: list[dict[str, Any]] = []
    for level_no in range(1, 10):
        level = spec['levels'][str(level_no)]
        attributes: dict[str, Any] = {}
        assumptions: list[dict[str, str]] = []
        for name, rule in (level.get('attributes') or {}).items():
            if not rule or not rule.get('any'):
                continue
            hit = _attribute_status(usable_clauses, list(rule['any']), full, max_quote_chars)
            required = bool(rule.get('required', True))
            attributes[name] = {
                'status': hit.status,
                'required': required,
                'hits': hit.hits,
                'quotes': hit.quotes,
            }
            if not required and hit.status != 'met':
                assumptions.append({
                    'attribute': name,
                    'assumption': rule.get('assumption', '该属性未明示，按默认情况假定，需人工核对'),
                })
        required_names = [n for n, v in attributes.items() if v['required']]
        missing_required = [n for n in required_names if attributes[n]['status'] != 'met']
        any_hit = any(v['hits'] for v in attributes.values())
        if required_names and not missing_required:
            status = 'met'
        elif any_hit:
            status = 'partial'
        else:
            status = 'not_met'
        gates.append({
            'level': level_no,
            'definition': level['definition'],
            'definition_note': level['definition_note'],
            'status': status,
            'attributes': attributes,
            'missing_attributes': missing_required,
            'assumptions': assumptions,
        })
    return gates, skipped


def _model_class(result: dict[str, Any]) -> tuple[int | None, str | None]:
    prediction = result.get('model_prediction') or {}
    label = prediction.get('level_label')
    level = prediction.get('level')
    if label is not None:
        label = str(label)
    if label in ABSTAIN_CLASSES:
        return None, label
    if isinstance(level, int) and 1 <= level <= 9:
        return level, label or str(level)
    return None, label


def align_to_standard(
    result: dict[str, Any],
    title: str,
    abstract: str = '',
    *,
    spec: dict[str, Any] | None = None,
    include_gates: bool = False,
) -> dict[str, Any]:
    """Attach a Q/GDW 12566—2025 view to an ``analyze()`` result.

    ``evidence_level`` is the only level this layer reports; the public
    ``analyze()`` result re-exports it as ``level`` when the standard is enabled.
    ``gates`` (the full per-attribute audit) is only included when
    ``include_gates=True``; otherwise a compact ``gates_summary`` is returned.
    """
    spec = spec or load_standard()
    gates, skipped = evaluate_gates(title, abstract, spec)
    met_levels = [gate['level'] for gate in gates if gate['status'] == 'met']
    evidence_level = max(met_levels) if met_levels else None
    trl_model, raw_class = _model_class(result)

    combined = f'{title} {abstract}'
    review_like = bool(REVIEW_TITLE.search(title) or REVIEW_TEXT.search(combined))
    own_work = bool(OWN_WORK.search(combined))
    is_object = (not review_like) or own_work

    if not is_object:
        status = 'not_applicable'
    elif raw_class == 'not_applicable' and evidence_level is None:
        status = 'not_applicable'
    elif evidence_level is None:
        status = 'insufficient_evidence'
    elif raw_class == 'not_applicable':
        status = 'evidence_conflict'
    else:
        status = 'public_evidence_preliminary'

    gap = None
    verdict = 'no_model_suggestion'
    if status == 'not_applicable':
        verdict = 'not_an_evaluation_object'
    elif trl_model is not None and evidence_level is not None:
        gap = trl_model - evidence_level
        if gap == 0:
            verdict = 'consistent'
        elif gap > 0:
            verdict = 'model_over_grades'
        else:
            verdict = 'model_under_grades'
    elif trl_model is not None:
        verdict = 'model_unsupported_by_quotes'

    credited_level = evidence_level if status in {'public_evidence_preliminary', 'evidence_conflict'} else None
    aligned = next((g for g in gates if g['level'] == credited_level), None)
    next_level = (credited_level + 1) if credited_level else None
    next_gate = next((g for g in gates if g['level'] == next_level), None)

    out: dict[str, Any] = {
        'standard_id': spec['standard_id'],
        'scale_version': spec['scale_version'],
        'table': spec['table'],
        'source_page': spec.get('source', {}).get('pdf_page'),
        'evaluation_status': status,
        'evidence_level': credited_level,
        'level_name': aligned['definition'] if aligned else None,
        'level_definition_note': aligned['definition_note'] if aligned else None,
        'trl_model': trl_model,
        'trl_evidence': evidence_level if status != 'not_applicable' else None,
        'trl_final': None,
        'model_class': raw_class,
        'model_vs_evidence': {'gap': gap, 'verdict': verdict},
        'gates_summary': [
            {'level': g['level'], 'status': g['status'], 'missing_attributes': g['missing_attributes']}
            for g in gates
        ],
        'next_level_gap': {
            'level': next_gate['level'],
            'definition': next_gate['definition'],
            'missing_attributes': next_gate['missing_attributes'],
        } if next_gate else None,
        'not_applicable_basis': ({
            'reason': ('综述/进展类文本，未报告自有技术验证，按标准 1 范围不属于评价对象'
                       if not is_object else
                       '模型判为非技术成熟度评价对象，且文本未发现可核对的分级证据'),
            'requires': spec['scope']['not_applicable_requires'],
        } if status == 'not_applicable' else None),
        'conflict': ({
            'reason': ('模型判为 not_applicable，但文本存在可由证据支持的分级线索'
                       if status == 'evidence_conflict' else
                       '模型给出了 TRL 等级，但文本为综述且无自有验证，等级不成立'),
            'note': '按标准 not_applicable 须有技术细则依据与批准人；结论应先复核模型判断',
        } if (status == 'evidence_conflict' or (status == 'not_applicable' and trl_model is not None)) else None),
        'object_check': {
            'review_like': review_like,
            'own_work_markers': own_work,
            'is_evaluation_object': is_object,
        },
        'skipped_sentences': skipped,
        'gates_are_project_defaults': True,
        'gates_credited': credited_level is not None,
        'formal_certified_level': None,
        'caveats': [
            '公开证据初评，未完成标准 5.5 规定的自评与专家评价',
            '门槛关键词为本项目默认细则，非标准统一规定；具体技术需专家配置',
            '仅使用标题与摘要，未核读全文、型式试验报告或运行记录',
        ],
    }
    if include_gates:
        out['gates'] = gates
    return out
