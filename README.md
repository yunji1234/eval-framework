# eval-framework

**[▶ Watch the demo](https://www.loom.com/share/d3d38dabdc2f4fc7b1768e274afe5747)**

An automated evaluation framework for the Archery Canada HR Policy RAG system. Uses Claude as an LLM judge to score retrieval quality, answer correctness, and appropriate refusal across 15 curated test cases.

## Architecture

```
eval_dataset.json          15 labeled test cases (factual / edge_case / out_of_scope)
       │
       ▼
evaluator.py  ─────────────────────────────────────────────────────────────────
  │  For each test case:                                                        │
  │  1. search(question)     →  ChromaDB top-8 chunks + ±2 neighbours each     │
  │  2. build_prompt(...)    →  formats context + question                      │
  │  3. Claude Sonnet 4.6    →  generates the RAG answer                        │
  │  4. judge(...)           →  Claude Haiku scores the answer                 │
  │                                                                             │
  └──► reports/report_{timestamp}.json                                          │
             │                                                                  │
             ▼                                                                  │
         report.py  →  terminal report with ASCII bar charts                   │
                                                                                │
rag.py + config.py         copied from rag-system for standalone operation ────┘
```

**Two Claude models are used intentionally:**
- `claude-sonnet-4-6` — the system under test (generates HR answers)
- `claude-haiku-4-5-20251001` — the judge (scores those answers, with prompt caching)

Using different models eliminates self-scoring bias.

## Retrieval Strategy

`search()` fetches the top 8 chunks by vector similarity, then automatically expands each result by ±2 adjacent chunks. This ensures policy content split across chunk boundaries is always captured — for example, a section header in one chunk and its body in the next. All chunks are deduplicated and sorted by document position before being passed to the model.

## Evaluation Metrics

Each test case is scored 0.0–1.0 on four dimensions:

| Dimension | What it measures |
|---|---|
| **correctness** | Factual accuracy vs. expected answer (valid inferences count) |
| **faithfulness** | Grounding in retrieved chunks |
| **hallucination** | Inverted — 1.0 means no hallucination |
| **appropriate_refusal** | For answerable questions: did it answer directly? For out-of-scope: did it decline? |

A test case **passes** when all four dimensions score ≥ 0.7. The overall evaluation **passes** when ≥ 80% of cases pass.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
ANTHROPIC_API_KEY=sk-ant-...
RAG_DB_PATH=../rag-system/chroma_db   # path to the ingested ChromaDB store
```

## Usage

### Run the evaluation

```bash
python evaluator.py
```

Prints live progress as it runs:

```
Testing 1/15: How many paid sick days are employees entitled …
  [PASS] correctness=0.95  faithfulness=0.92  hallucination=0.98  refusal=1.00
Testing 2/15: What are the regular office hours and core oper…
  [PASS] correctness=0.91  faithfulness=0.88  hallucination=0.96  refusal=1.00
...
── Summary ─────────────────────────────────────
  Total:             15
  Passed:            12
  Overall pass rate: 80.0%
  Report saved to: reports/report_20260528_143022.json
```

### View the report

```bash
python report.py
```

Renders the most recent report as a formatted terminal dashboard with ASCII bar charts, per-category breakdown, and top failures.

### Run the tests

```bash
pytest tests/
```

All tests mock Anthropic API calls — no API key required, no ChromaDB needed.

## Project Structure

```
eval-framework/
├── eval_dataset.json        # 15 labeled test cases
├── evaluator.py             # orchestrates the full eval loop
├── judge.py                 # Claude-as-judge scoring via tool use
├── report.py                # terminal report renderer
├── rag.py                   # copied from rag-system (search + build_prompt + ask)
├── config.py                # RAG constants (collection name, embedding model)
├── conftest.py              # pytest sys.path setup
├── requirements.txt
├── .env.example
└── tests/
    └── test_evaluator.py    # unit tests (all API calls mocked)
```

## Running on a Remote Server

1. Copy the entire `eval-framework/` directory to the server.
2. Copy (or re-ingest) the ChromaDB store and note its path.
3. Install dependencies: `pip install -r requirements.txt`
4. Set `RAG_DB_PATH` in `.env` to the absolute path of the ChromaDB store.
5. Run `python evaluator.py`.

If you need to re-ingest documents on the server, copy `ingest.py` and your source PDFs from the rag-system project, then run `python ingest.py`.

## Adding Test Cases

Add entries to `eval_dataset.json` following this schema:

```json
{
  "id": "factual_009",
  "question": "What is the RRSP matching contribution rate?",
  "expected_answer": "...",
  "category": "factual",
  "should_refuse": false
}
```

Categories: `factual`, `edge_case`, `out_of_scope`. Set `should_refuse: true` for questions whose correct answer is "I don't know."
