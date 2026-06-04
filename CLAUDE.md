# eval-framework

An LLM evaluation framework that uses Claude as a judge to score a RAG system built on the Archery Canada HR Policy Manual.

## Project structure

```
eval-framework/
├── eval_dataset.json      # 15 labeled test cases — source of truth for the eval
├── evaluator.py           # main eval loop: RAG answer → judge → aggregate → save
├── judge.py               # calls Claude Haiku via tool_use for structured scores
├── report.py              # terminal report renderer (ASCII bars, ANSI color)
├── rag.py                 # copied from rag-system: search(), build_prompt(), ask()
├── config.py              # shared constants (collection name, db path, embedding model)
├── conftest.py            # adds project root to sys.path for pytest
├── requirements.txt
├── .env.example
└── tests/
    └── test_evaluator.py
```

## Environment

Required `.env` variables:

```
ANTHROPIC_API_KEY=...
RAG_DB_PATH=../rag-system/chroma_db   # path to the ingested ChromaDB store
```

## Running

```bash
python evaluator.py    # run the full evaluation, saves to reports/
python report.py       # render the most recent report in the terminal
pytest tests/          # run unit tests (no API key needed — all mocked)
```

## Key design decisions

**Two models, intentionally.** `evaluator.py` uses `claude-sonnet-4-6` to generate RAG answers (the system under test). `judge.py` uses `claude-haiku-4-5-20251001` to score them. This prevents the model from scoring its own outputs.

**Judge uses tool_use for structured output.** `judge()` forces a `tool_choice={"type": "tool", "name": "evaluate_rag_answer"}` call so Claude returns a typed JSON dict instead of free text. This makes score extraction deterministic.

**Prompt caching on the judge system prompt.** The judge's system prompt is marked `cache_control: {"type": "ephemeral"}`. It's identical across all 15 calls per run, so it's cached after the first request.

**`rag.py` and `config.py` are copied, not imported.** The rag-system lives in a sibling directory and may not be present on a remote server. Copying these two files makes eval-framework self-contained. Do not add a path dependency on rag-system.

**`DEFAULT_DB_PATH` reads from `RAG_DB_PATH` env var.** This lets the same code run locally (`../rag-system/chroma_db`) and on remote servers (absolute path to a copied ChromaDB store) without code changes.

**Pass threshold is 0.7 per dimension, 0.80 overall.** `PASS_THRESHOLD = 0.7` in `evaluator.py` governs per-case pass/fail. `PASS_THRESHOLD = 0.80` in `report.py` governs the final PASS/FAIL verdict. Adjust these constants if the team changes the bar.

## Test patterns

All Anthropic API calls are mocked. The canonical mock pattern used in tests:

```python
mock_client = MagicMock()
mock_client.messages.create.return_value = make_api_response(scores_dict)

with patch("judge._get_client", return_value=mock_client):
    result = judge(...)
```

The `_reset_judge_singleton` fixture (autouse) resets `judge._client = None` before every test so singleton state from one test cannot leak into the next.

`make_api_response()` returns a `SimpleNamespace` that mimics the shape `judge()` expects: `response.content[0].type == "tool_use"`, `.name == "evaluate_rag_answer"`, `.input == scores_dict`.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY and RAG_DB_PATH
```
