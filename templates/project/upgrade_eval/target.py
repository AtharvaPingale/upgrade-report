"""The integration contract: the only code a project writes for upgrade-report.

upgrade-report calls `run` once per example, per arm, per repeat, with the
arm's model and prompt reference. Keep it a thin wrapper around your real
application entry point, so the eval exercises production code.
"""

from __future__ import annotations


def run(inputs: dict, model: str, prompt: str) -> dict:
    """Return at least {"answer": str}.

    Optional keys that enable richer metrics and diffs:
      "doc_ids":    retrieved document ids, in rank order (recall@k)
      "contexts":   retrieved passages (registry:faithfulness)
      "citations":  citations in the answer (citations per answer)
      "tool_calls": [{"name": ...}, ...] for agentic targets
      "usage":      {"input_tokens": n, "output_tokens": n} for cost; or call
                    upgrade_report.record_usage(...) from your LLM wrapper, or
                    rely on LangSmith trace token counts.
    """
    # TODO: replace this stub with a call into your application, for example:
    #   result = my_app.answer(inputs["question"], model=model, prompt_ref=prompt)
    #   return {"answer": result.text, "doc_ids": result.doc_ids,
    #           "usage": {"input_tokens": result.usage.input, "output_tokens": result.usage.output}}
    question = inputs["question"]
    answer = f"Based on our documentation: {question.rstrip('?')}."
    return {
        "answer": answer,
        "usage": {"input_tokens": 800 + len(question) // 4, "output_tokens": len(answer) // 4},
    }
