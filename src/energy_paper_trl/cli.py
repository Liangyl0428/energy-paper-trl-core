from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import analyze
from .model import SciBERTAdapter, SVMAdapter


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate one paper's evidence-supported TRL")
    ap.add_argument("--title", required=True)
    ap.add_argument("--abstract", default="")
    ap.add_argument("--doc-id")
    ap.add_argument("--backend", choices=["rules", "svm", "scibert"], default="rules")
    ap.add_argument("--model-path", type=Path)
    ap.add_argument(
        "--no-explain", action="store_true",
        help="skip the sentence-level attribution (label + rule evidence only)",
    )
    ap.add_argument(
        "--standard", choices=["qgdw12566", "none"], default="qgdw12566",
        help="align the output with Q/GDW 12566—2025 表1 (default) or skip it",
    )
    ap.add_argument(
        "--full-gates", action="store_true",
        help="include the full per-attribute gate audit (large JSON); default is gates_summary",
    )
    args = ap.parse_args()
    model = None
    if args.backend == "svm":
        if not args.model_path:
            ap.error("--model-path is required for svm")
        model = SVMAdapter(args.model_path)
    elif args.backend == "scibert":
        if not args.model_path:
            ap.error("--model-path is required for scibert")
        model = SciBERTAdapter(args.model_path)
    print(json.dumps(
        analyze(
            args.title, args.abstract, doc_id=args.doc_id, backend=args.backend,
            model=model, explain=not args.no_explain,
            standard=args.standard if args.standard == "qgdw12566" else None,
            include_gates=args.full_gates,
        ),
        ensure_ascii=False, indent=2,
    ))
