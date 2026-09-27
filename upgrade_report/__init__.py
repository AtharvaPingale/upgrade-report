"""upgrade-report: is the candidate at least as good as what we run today, where does it differ,
and what will it cost?"""

__version__ = "0.1.0"

from .usage import record_usage  # noqa: E402

__all__ = ["__version__", "record_usage"]
