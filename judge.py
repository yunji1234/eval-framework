import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_SYSTEM_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Your task is to score the quality of a RAG system's answer along four dimensions.

Scoring guidelines:

- correctness (0.0–1.0): How factually accurate is the answer compared to the expected answer?
  1.0 = fully correct, 0.5 = partially correct, 0.0 = wrong or missing key facts.
  Valid inferences count: if the expected answer is derived by implication from the context
  (e.g. a policy lists only permitted uses, implying unlisted uses are forbidden), an actual
  answer that correctly captures that inference should score high even if not verbatim.

- faithfulness (0.0–1.0): Is the answer grounded in the retrieved chunks?
  1.0 = every claim traces back to the chunks, 0.0 = answer ignores or contradicts the chunks.

- hallucination (0.0–1.0): Did the model invent facts not present in the retrieved chunks?
  1.0 = no hallucination at all, 0.0 = answer is entirely fabricated.

- appropriate_refusal (0.0–1.0): Score this dimension based on should_refuse:
  - When should_refuse is FALSE (the question is answerable from the policy):
      1.0 = gave a clear, direct, useful answer.
      0.5 = answered but with unnecessary hedging or excessive caveats.
      0.0 = refused or said "I don't know" when the answer was available in context.
  - When should_refuse is TRUE (the question is genuinely out of scope):
      1.0 = correctly said it doesn't know / the information isn't in the manual.
      0.5 = partially answered but acknowledged the limitation.
      0.0 = confidently gave a specific answer to a question it cannot know.

Be strict and consistent. Always provide concise reasoning for each score."""

_EVAL_TOOL = {
    "name": "evaluate_rag_answer",
    "description": "Record evaluation scores and reasoning for a RAG system answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correctness": {
                "type": "number",
                "description": "Factual accuracy score (0.0–1.0)",
            },
            "correctness_reasoning": {
                "type": "string",
                "description": "Brief explanation for the correctness score",
            },
            "faithfulness": {
                "type": "number",
                "description": "Grounding in retrieved chunks score (0.0–1.0)",
            },
            "faithfulness_reasoning": {
                "type": "string",
                "description": "Brief explanation for the faithfulness score",
            },
            "hallucination": {
                "type": "number",
                "description": "Inverted hallucination score — 1.0 means no hallucination (0.0–1.0)",
            },
            "hallucination_reasoning": {
                "type": "string",
                "description": "Brief explanation for the hallucination score",
            },
            "appropriate_refusal": {
                "type": "number",
                "description": "Refusal appropriateness score (0.0–1.0)",
            },
            "appropriate_refusal_reasoning": {
                "type": "string",
                "description": "Brief explanation for the appropriate_refusal score",
            },
        },
        "required": [
            "correctness",
            "correctness_reasoning",
            "faithfulness",
            "faithfulness_reasoning",
            "hallucination",
            "hallucination_reasoning",
            "appropriate_refusal",
            "appropriate_refusal_reasoning",
        ],
    },
}


def judge(
    question: str,
    expected_answer: str,
    actual_answer: str,
    retrieved_chunks: list[str],
    should_refuse: bool = False,
) -> dict:
    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
    )

    user_message = f"""Evaluate the following RAG system response.

Question: {question}

Expected answer: {expected_answer}

Retrieved chunks:
{chunks_text}

Actual answer from RAG system: {actual_answer}

should_refuse: {should_refuse}
(If True, the question is out of scope and the system should have said it doesn't know.)

Call evaluate_rag_answer with your scores and reasoning."""

    client = _get_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_EVAL_TOOL],
        tool_choice={"type": "tool", "name": "evaluate_rag_answer"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "evaluate_rag_answer":
            return block.input

    raise RuntimeError(f"Judge did not return tool call. Response: {response}")
