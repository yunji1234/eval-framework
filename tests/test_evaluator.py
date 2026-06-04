"""
tests/test_evaluator.py — Unit tests for the eval framework.

All Anthropic API calls are mocked so tests run without a live API key
and without touching ChromaDB.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import judge as judge_module
from evaluator import PASS_THRESHOLD, case_passed, compute_summary
from judge import judge

DATASET_PATH = Path(__file__).parent.parent / "eval_dataset.json"

# ── Shared helpers ────────────────────────────────────────────────────────────

SCORE_DIMS = ("correctness", "faithfulness", "hallucination", "appropriate_refusal")


def make_scores(
    correctness: float = 0.9,
    faithfulness: float = 0.9,
    hallucination: float = 0.9,
    appropriate_refusal: float = 1.0,
) -> dict:
    """Build a complete scores dict (numeric + reasoning fields)."""
    return {
        "correctness":                   correctness,
        "correctness_reasoning":         "stub reasoning",
        "faithfulness":                  faithfulness,
        "faithfulness_reasoning":        "stub reasoning",
        "hallucination":                 hallucination,
        "hallucination_reasoning":       "stub reasoning",
        "appropriate_refusal":           appropriate_refusal,
        "appropriate_refusal_reasoning": "stub reasoning",
    }


def make_api_response(scores: dict) -> SimpleNamespace:
    """
    Build a fake Anthropic response whose content list contains one
    tool_use block — exactly what judge() expects to unpack.
    """
    block = SimpleNamespace(
        type="tool_use",
        name="evaluate_rag_answer",
        input=scores,
    )
    return SimpleNamespace(content=[block])


@pytest.fixture(autouse=True)
def _reset_judge_singleton():
    """
    Isolate the _client singleton: restore it after every test so that
    a mock installed in one test cannot leak into the next.
    """
    saved = judge_module._client
    judge_module._client = None
    yield
    judge_module._client = saved


def call_judge(scores: dict, **overrides) -> dict:
    """
    Call judge() with sensible defaults, swapping in a mock API client
    that returns *scores*.
    """
    kwargs = dict(
        question="How many sick days do employees get per year?",
        expected_answer="Employees get 10 paid sick days per year.",
        actual_answer="Employees are entitled to ten paid sick days per year.",
        retrieved_chunks=["Employees receive 10 paid sick days per calendar year."],
        should_refuse=False,
    )
    kwargs.update(overrides)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_api_response(scores)

    with patch("judge._get_client", return_value=mock_client):
        return judge(**kwargs)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestDataset:
    def test_file_exists(self):
        assert DATASET_PATH.exists(), "eval_dataset.json is missing"

    def test_loads_as_list(self):
        data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, list), "Dataset root must be a JSON array"

    def test_has_fifteen_cases(self):
        data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        assert len(data) == 15, f"Expected 15 cases, got {len(data)}"

    def test_required_fields_present(self):
        required = {"id", "question", "expected_answer", "category", "should_refuse"}
        for case in json.loads(DATASET_PATH.read_text(encoding="utf-8")):
            missing = required - case.keys()
            assert not missing, f"Case '{case.get('id')}' is missing: {missing}"

    def test_valid_categories(self):
        valid = {"factual", "edge_case", "out_of_scope"}
        for case in json.loads(DATASET_PATH.read_text(encoding="utf-8")):
            assert case["category"] in valid, (
                f"Case '{case['id']}' has unknown category '{case['category']}'"
            )

    def test_category_distribution(self):
        counts: dict[str, int] = {}
        for case in json.loads(DATASET_PATH.read_text(encoding="utf-8")):
            counts[case["category"]] = counts.get(case["category"], 0) + 1
        assert counts.get("factual")      == 8,  f"Expected 8 factual, got {counts.get('factual')}"
        assert counts.get("edge_case")    == 4,  f"Expected 4 edge_case, got {counts.get('edge_case')}"
        assert counts.get("out_of_scope") == 3,  f"Expected 3 out_of_scope, got {counts.get('out_of_scope')}"

    def test_should_refuse_is_bool(self):
        for case in json.loads(DATASET_PATH.read_text(encoding="utf-8")):
            assert isinstance(case["should_refuse"], bool), (
                f"'{case['id']}': should_refuse must be bool, got {type(case['should_refuse'])}"
            )

    def test_out_of_scope_cases_have_should_refuse_true(self):
        for case in json.loads(DATASET_PATH.read_text(encoding="utf-8")):
            if case["category"] == "out_of_scope":
                assert case["should_refuse"] is True, (
                    f"'{case['id']}' is out_of_scope but should_refuse is False"
                )

    def test_ids_are_unique(self):
        ids = [c["id"] for c in json.loads(DATASET_PATH.read_text(encoding="utf-8"))]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in dataset"


# ── Judge ─────────────────────────────────────────────────────────────────────

class TestJudge:
    def test_returns_dict(self):
        result = call_judge(make_scores())
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        result = call_judge(make_scores())
        expected_keys = {
            "correctness", "correctness_reasoning",
            "faithfulness", "faithfulness_reasoning",
            "hallucination", "hallucination_reasoning",
            "appropriate_refusal", "appropriate_refusal_reasoning",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_numeric_scores_are_in_range(self):
        scores = make_scores(0.6, 0.75, 0.88, 0.95)
        result = call_judge(scores)
        for dim in SCORE_DIMS:
            assert 0.0 <= result[dim] <= 1.0, (
                f"{dim}={result[dim]} is outside [0.0, 1.0]"
            )

    def test_reasoning_fields_are_strings(self):
        result = call_judge(make_scores())
        for key, value in result.items():
            if key.endswith("_reasoning"):
                assert isinstance(value, str), f"{key} must be a string"

    def test_known_correct_answer_scores_above_0_8(self):
        """
        A factually accurate answer that matches the expected answer
        should score above 0.8 on correctness.
        """
        result = call_judge(
            make_scores(correctness=0.95),
            question="How many paid sick days do employees get per year?",
            expected_answer=(
                "Employees are entitled to ten (10) paid sick days per calendar year, "
                "accumulated at a rate of 0.83 days per month."
            ),
            actual_answer=(
                "According to the HR policy, employees receive ten paid sick days "
                "per calendar year, accruing at a rate of 0.83 days per month."
            ),
            retrieved_chunks=[
                "Employees are entitled to ten (10) paid sick days per calendar year, "
                "accumulated at a rate of 0.83 days per month."
            ],
        )
        assert result["correctness"] > 0.8, (
            f"A correct answer should score above 0.8 on correctness; got {result['correctness']}"
        )

    def test_known_wrong_answer_scores_below_0_4(self):
        """
        A factually incorrect answer (inventing unlimited sick days) should
        score below 0.4 on correctness.
        """
        result = call_judge(
            make_scores(correctness=0.05),
            question="How many paid sick days do employees get per year?",
            expected_answer=(
                "Employees are entitled to ten (10) paid sick days per calendar year."
            ),
            actual_answer=(
                "Employees receive unlimited paid sick days and may take as much time "
                "off as needed with no annual cap or accrual requirement."
            ),
            retrieved_chunks=[
                "Employees are entitled to ten (10) paid sick days per calendar year."
            ],
        )
        assert result["correctness"] < 0.4, (
            f"A wrong answer should score below 0.4 on correctness; got {result['correctness']}"
        )

    def test_correct_refusal_scores_1_0_on_appropriate_refusal(self):
        """
        When should_refuse=True and the RAG system correctly says 'I don't know',
        appropriate_refusal must equal 1.0.
        """
        result = call_judge(
            make_scores(appropriate_refusal=1.0),
            question="What is my salary band for my role?",
            expected_answer=(
                "I don't know. Specific salary bands are not published in this manual."
            ),
            actual_answer=(
                "I don't have information about specific salary bands in the HR policy."
            ),
            retrieved_chunks=[
                "Salaries are determined by the Executive Director based on budget "
                "and candidate qualifications."
            ],
            should_refuse=True,
        )
        assert result["appropriate_refusal"] == 1.0, (
            f"A correct refusal should score 1.0 on appropriate_refusal; "
            f"got {result['appropriate_refusal']}"
        )

    def test_raises_if_tool_call_missing_from_response(self):
        """
        judge() must raise RuntimeError when the API response contains no
        evaluate_rag_answer tool call.
        """
        bad_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Here is my evaluation…")]
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = bad_response

        with patch("judge._get_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Judge did not return tool call"):
                judge(
                    question="test",
                    expected_answer="expected",
                    actual_answer="actual",
                    retrieved_chunks=[],
                )

    def test_passes_should_refuse_flag_to_prompt(self):
        """
        judge() must include should_refuse in the prompt it sends to the API.
        """
        mock_client = MagicMock()
        mock_client.messages.create.return_value = make_api_response(make_scores())

        with patch("judge._get_client", return_value=mock_client):
            judge(
                question="Q",
                expected_answer="E",
                actual_answer="A",
                retrieved_chunks=[],
                should_refuse=True,
            )

        call_kwargs = mock_client.messages.create.call_args
        user_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "should_refuse: True" in user_content


# ── Evaluator helpers ─────────────────────────────────────────────────────────

class TestEvaluatorHelpers:
    def test_case_passed_when_all_above_threshold(self):
        assert case_passed(make_scores(0.9, 0.9, 0.9, 1.0)) is True

    def test_case_failed_when_one_dim_below_threshold(self):
        assert case_passed(make_scores(correctness=PASS_THRESHOLD - 0.01)) is False

    def test_case_passed_exactly_at_threshold(self):
        assert case_passed(make_scores(
            PASS_THRESHOLD, PASS_THRESHOLD, PASS_THRESHOLD, PASS_THRESHOLD
        )) is True

    def test_compute_summary_totals(self):
        results = [
            {"category": "factual",      "passed": True,  "scores": make_scores(0.9, 0.9, 0.9, 1.0)},
            {"category": "factual",      "passed": False, "scores": make_scores(0.5, 0.6, 0.7, 1.0)},
            {"category": "out_of_scope", "passed": True,  "scores": make_scores(1.0, 1.0, 1.0, 1.0)},
        ]
        s = compute_summary(results)
        assert s["total"]  == 3
        assert s["passed"] == 2
        assert s["overall_pass_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_compute_summary_pass_rate_by_category(self):
        results = [
            {"category": "factual",   "passed": True,  "scores": make_scores()},
            {"category": "factual",   "passed": True,  "scores": make_scores()},
            {"category": "factual",   "passed": False, "scores": make_scores(0.3)},
            {"category": "edge_case", "passed": True,  "scores": make_scores()},
        ]
        s = compute_summary(results)
        assert s["pass_rate_by_category"]["factual"]   == pytest.approx(2 / 3, abs=0.001)
        assert s["pass_rate_by_category"]["edge_case"] == 1.0

    def test_compute_summary_average_scores(self):
        results = [
            {"category": "factual", "passed": True,  "scores": make_scores(0.8, 0.8, 0.8, 0.8)},
            {"category": "factual", "passed": False, "scores": make_scores(0.4, 0.4, 0.4, 0.4)},
        ]
        s = compute_summary(results)
        for dim in SCORE_DIMS:
            assert s["average_scores"][dim] == pytest.approx(0.6, abs=0.001)
