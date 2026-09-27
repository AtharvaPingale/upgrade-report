"""Evaluators for tests, in LangSmith's argument style."""


def quality(outputs: dict) -> float:
    return float(outputs["quality"])


def format_compliance(outputs: dict) -> bool:
    return outputs["answer"].startswith(("The answer", "I'm sorry"))


def correct(outputs: dict, reference_outputs: dict) -> bool:
    return reference_outputs["answer"].lower() in outputs["answer"].lower()


def legacy_style(run, example):
    """Old (run, example) signature returning an EvaluationResult-like dict."""
    return {"key": "has_citation", "score": bool(run.outputs.get("citations"))}
