"""Offline evaluation harness.

Metrics
  * retrieval hit@k     - the expected document is among the passages kept after grading
  * citation precision  - share of cited passages that belong to the expected document
  * answer recall       - the answer contains the expected key facts (e.g. '4.5%')
  * abstention accuracy - unanswerable questions get "I don't know" instead of a made-up answer
  * latency p50 / max

Usage: finrag-eval [--dataset data/eval.jsonl] [--min-score 0.8] [--json report.json]
Exits with status 1 if the overall score is below --min-score (used as a CI quality gate).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from finrag.factory import build_agent

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "eval.jsonl"


def evaluate(dataset: Path) -> dict:
    agent = build_agent()
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    details, latencies = [], []
    hits = recalls = abstain_ok = 0
    precisions: list[float] = []
    n_answerable = sum(r["answerable"] for r in rows)
    n_unanswerable = len(rows) - n_answerable

    for row in rows:
        res = agent.ask(row["question"])
        latencies.append(res.latency_ms)
        retrieved_docs = {cid.split("#")[0] for cid in res.retrieved}
        if row["answerable"]:
            hit = row["expected_doc"] in retrieved_docs
            cited_docs = [c.split("#")[0] for c in res.citations]
            precision = (
                (sum(d == row["expected_doc"] for d in cited_docs) / len(cited_docs)) if cited_docs else 0.0
            )
            recall = all(k.lower() in res.answer.lower() for k in row["must_contain"])
            hits += hit
            recalls += recall
            precisions.append(precision)
            details.append(
                {
                    "q": row["question"],
                    "hit": hit,
                    "precision": round(precision, 2),
                    "recall": recall,
                    "answer": res.answer,
                }
            )
        else:
            ok = res.abstained
            abstain_ok += ok
            details.append({"q": row["question"], "abstained": ok, "answer": res.answer})

    metrics = {
        "generator": agent.generator.name,
        "questions": len(rows),
        "retrieval_hit": round(hits / n_answerable, 3),
        "citation_precision": round(statistics.mean(precisions), 3) if precisions else 0.0,
        "answer_recall": round(recalls / n_answerable, 3),
        "abstention_accuracy": round(abstain_ok / n_unanswerable, 3) if n_unanswerable else 1.0,
        "latency_ms_p50": round(statistics.median(latencies), 1),
        "latency_ms_max": round(max(latencies), 1),
    }
    metrics["overall"] = round(
        statistics.mean(
            [
                metrics["retrieval_hit"],
                metrics["citation_precision"],
                metrics["answer_recall"],
                metrics["abstention_accuracy"],
            ]
        ),
        3,
    )
    return {"metrics": metrics, "details": details}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)

    report = evaluate(args.dataset)
    m = report["metrics"]
    width = max(len(k) for k in m)
    for k, v in m.items():
        print(f"{k.ljust(width)} : {v}")
    failures = [
        d for d in report["details"] if not d.get("recall", d.get("abstained", True)) or d.get("hit") is False
    ]
    for d in failures:
        print(f"  ! {d['q']} -> {d['answer'][:120]}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if m["overall"] < args.min_score:
        print(f"FAIL: overall {m['overall']} < {args.min_score}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
