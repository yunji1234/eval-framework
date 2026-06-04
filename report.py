"""
report.py — Renders a terminal report from the most recent evaluation.

Usage:
    python report.py
"""

import json
import sys
import textwrap
from pathlib import Path

REPORTS_DIR    = Path(__file__).parent / "reports"
PASS_THRESHOLD = 0.80
BAR_WIDTH      = 20
WIDTH          = 72

# ── ANSI codes ────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"


def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI escape codes when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + str(text) + RESET


def _bar(score: float) -> str:
    filled = round(score * BAR_WIDTH)
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _score_color(score: float) -> str:
    if score >= 0.80:
        return GREEN
    if score >= 0.60:
        return YELLOW
    return RED


def _pct(score: float) -> str:
    return f"{score * 100:5.1f}%"


def _rule(char: str = "─") -> str:
    return "  " + char * (WIDTH - 4)


def _box_line(text: str) -> str:
    """Center text inside a full-width box row."""
    return "║" + text.center(WIDTH - 2) + "║"


def _wrap(text: str, prefix: str) -> str:
    """Wrap long text with a labeled prefix, indenting continuation lines."""
    indent = " " * len(prefix)
    return textwrap.fill(text, width=WIDTH,
                         initial_indent=prefix,
                         subsequent_indent=indent)


def _truncate(text: str, max_chars: int = 220) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


# ── Sections ──────────────────────────────────────────────────────────────────

def _render_header(ts: str, total: int) -> None:
    date, _, time = ts.partition("T")
    subtitle = f"{date}  {time[:8]}  ·  {total} test cases"
    print(_c("╔" + "═" * (WIDTH - 2) + "╗", BOLD))
    print(_c(_box_line("HR POLICY RAG  —  EVALUATION REPORT"), BOLD))
    print(_c(_box_line(subtitle), DIM))
    print(_c("╚" + "═" * (WIDTH - 2) + "╝", BOLD))
    print()


def _render_overall(passed: int, total: int, overall: float) -> None:
    verdict_pass = overall >= PASS_THRESHOLD
    clr          = GREEN if verdict_pass else RED
    symbol       = "✔" if verdict_pass else "✗"
    word         = "PASS" if verdict_pass else "FAIL"

    print(_c("  OVERALL PASS RATE", BOLD))
    print(_rule())
    print()
    print(_c(f"          {passed} of {total} cases passed", BOLD + clr))
    print()
    print(_c(f"          {_bar(overall)}  {_pct(overall)}", clr))
    print()
    print(_c(f"          {symbol}  {word}  —  threshold is {PASS_THRESHOLD:.0%}", BOLD + clr))
    print()
    print()


def _render_dimensions(avg_scores: dict) -> None:
    dims = [
        ("Correctness",         "correctness",         ""),
        ("Faithfulness",        "faithfulness",         ""),
        ("Hallucination",       "hallucination",        "  (1.0 = no hallucination)"),
        ("Appropriate refusal", "appropriate_refusal",  ""),
    ]
    label_w = max(len(label) for label, _, _ in dims)

    print(_c("  SCORES BY DIMENSION", BOLD))
    print(_rule())
    print()

    for label, key, note in dims:
        score = avg_scores[key]
        clr   = _score_color(score)
        line  = f"  {label:<{label_w}}  {_bar(score)}  {_c(_pct(score), BOLD + clr)}"
        if note:
            line += _c(note, DIM)
        print(line)

    print()
    print()


def _render_categories(summary: dict, results: list[dict]) -> None:
    print(_c("  RESULTS BY CATEGORY", BOLD))
    print(_rule())
    print()

    for cat in ("factual", "edge_case", "out_of_scope"):
        rate = summary["pass_rate_by_category"].get(cat)
        if rate is None:
            continue
        cat_rs = [r for r in results if r["category"] == cat]
        n = len(cat_rs)
        p = sum(1 for r in cat_rs if r["passed"])
        clr = _score_color(rate)
        print(f"  {cat:<18}  {_bar(rate)}  {p:>2} / {n}   {_c(_pct(rate), BOLD + clr)}")

    print()
    print()


def _render_failures(results: list[dict]) -> None:
    failures = sorted(
        [r for r in results if not r["passed"]],
        key=lambda r: sum(r["scores"][d] for d in
                          ("correctness", "faithfulness",
                           "hallucination", "appropriate_refusal")),
    )
    top = failures[:3]

    if not top:
        print(_c("  NO FAILURES — perfect run!", BOLD + GREEN))
        print()
        return

    n_label = f"TOP {len(top)} FAILURE{'S' if len(top) != 1 else ''}"
    print(_c(f"  {n_label}", BOLD))
    print(_rule())

    for rank, r in enumerate(top, 1):
        s = r["scores"]
        scores_str = (
            f"correctness {s['correctness']:.2f}  ·  "
            f"faithfulness {s['faithfulness']:.2f}  ·  "
            f"hallucination {s['hallucination']:.2f}  ·  "
            f"refusal {s['appropriate_refusal']:.2f}"
        )
        print()
        print(_c(f"  #{rank}  {r['id']}", BOLD) + "   " + _c(scores_str, DIM))
        print(_wrap(r["question"],          "       Q:  "))
        print(_c(_wrap(_truncate(r["expected_answer"]), "      Exp: "), DIM))
        print(_c(_wrap(_truncate(r["actual_answer"]),   "      Got: "), RED))

    print()


def _render_verdict(overall: float) -> None:
    verdict_pass = overall >= PASS_THRESHOLD
    clr          = GREEN if verdict_pass else RED
    symbol       = "✔" if verdict_pass else "✗"
    word         = "PASS" if verdict_pass else "FAIL"
    meets        = "meets" if verdict_pass else "is below"

    bar_line  = _c("═" * WIDTH, BOLD + clr)
    text_line = _c(
        f"  {symbol}  {word}  —  overall pass rate "
        f"{_pct(overall).strip()} {meets} the {PASS_THRESHOLD:.0%} threshold",
        BOLD + clr,
    )
    print(bar_line)
    print(text_line)
    print(bar_line)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def load_latest_report() -> dict:
    reports = sorted(REPORTS_DIR.glob("report_*.json"))
    if not reports:
        print("No reports found in reports/. Run evaluator.py first.", file=sys.stderr)
        sys.exit(1)
    with open(reports[-1]) as f:
        return json.load(f)


def render(report: dict) -> None:
    summary = report["summary"]
    results = report["results"]

    _render_header(report["timestamp"], summary["total"])
    _render_overall(summary["passed"], summary["total"], summary["overall_pass_rate"])
    _render_dimensions(summary["average_scores"])
    _render_categories(summary, results)
    _render_failures(results)
    _render_verdict(summary["overall_pass_rate"])


if __name__ == "__main__":
    render(load_latest_report())
