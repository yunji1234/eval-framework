"""
evaluator.py — Runs the eval dataset against the RAG system and scores each answer.

Usage:
    python evaluator.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from rag import search, build_prompt, SYSTEM_PROMPT, CLAUDE_MODEL, MAX_TOKENS
from judge import judge

load_dotenv()

PASS_THRESHOLD = 0.7
DATASET_PATH   = Path(__file__).parent / "eval_dataset.json"
REPORTS_DIR    = Path(__file__).parent / "reports"


def get_actual_answer(question: str) -> tuple[str, list[dict]]:
    chunks = search(question)
    prompt = build_prompt(question, chunks)
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    if not message.content:
        raise RuntimeError(f"Claude returned no content for question: {question!r}")
    return message.content[0].text, chunks


def case_passed(scores: dict) -> bool:
    return all(
        scores[dim] >= PASS_THRESHOLD
        for dim in ("correctness", "faithfulness", "hallucination", "appropriate_refusal")
    )


def run_evaluation() -> list[dict]:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    total = len(dataset)
    results = []

    for i, case in enumerate(dataset, 1):
        question_preview = case["question"][:60] + ("…" if len(case["question"]) > 60 else "")
        print(f"Testing {i}/{total}: {question_preview}")

        actual_answer, retrieved_chunks = get_actual_answer(case["question"])

        chunk_texts = [c["text"] for c in retrieved_chunks]

        scores = judge(
            question=case["question"],
            expected_answer=case["expected_answer"],
            actual_answer=actual_answer,
            retrieved_chunks=chunk_texts,
            should_refuse=case.get("should_refuse", False),
        )

        passed = case_passed(scores)

        results.append({
            "id":               case["id"],
            "question":         case["question"],
            "category":         case["category"],
            "should_refuse":    case.get("should_refuse", False),
            "expected_answer":  case["expected_answer"],
            "actual_answer":    actual_answer,
            "retrieved_chunks": retrieved_chunks,
            "scores":           scores,
            "passed":           passed,
        })

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] correctness={scores['correctness']:.2f}  "
              f"faithfulness={scores['faithfulness']:.2f}  "
              f"hallucination={scores['hallucination']:.2f}  "
              f"refusal={scores['appropriate_refusal']:.2f}")

    return results


def compute_summary(results: list[dict]) -> dict:
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])

    dims = ("correctness", "faithfulness", "hallucination", "appropriate_refusal")
    avg_scores = {
        dim: round(sum(r["scores"][dim] for r in results) / total, 3)
        for dim in dims
    }

    categories = {r["category"] for r in results}
    pass_rate_by_category = {}
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        pass_rate_by_category[cat] = round(
            sum(1 for r in cat_results if r["passed"]) / len(cat_results), 3
        )

    return {
        "total":                 total,
        "passed":                passed,
        "overall_pass_rate":     round(passed / total, 3),
        "average_scores":        avg_scores,
        "pass_rate_by_category": pass_rate_by_category,
    }


def save_report(summary: dict, results: list[dict]) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary":   summary,
        "results":   results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


if __name__ == "__main__":
    print(f"Running evaluation on {DATASET_PATH.name}…\n")

    results = run_evaluation()
    summary = compute_summary(results)
    report_path = save_report(summary, results)

    print("\n── Summary ─────────────────────────────────────")
    print(f"  Total:             {summary['total']}")
    print(f"  Passed:            {summary['passed']}")
    print(f"  Overall pass rate: {summary['overall_pass_rate']:.1%}")
    print(f"\n  Average scores:")
    for dim, score in summary["average_scores"].items():
        print(f"    {dim:<22} {score:.3f}")
    print(f"\n  Pass rate by category:")
    for cat, rate in summary["pass_rate_by_category"].items():
        print(f"    {cat:<22} {rate:.1%}")
    print(f"\n  Report saved to: {report_path}")
