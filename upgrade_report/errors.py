"""Exception types. Anything derived from UpgradeReportError exits with code 2."""


class UpgradeReportError(Exception):
    """Base class for tool and configuration errors."""


class ConfigError(UpgradeReportError):
    """The config file, pricing file, or a referenced object is invalid."""


class DatasetError(UpgradeReportError):
    """The dataset could not be loaded."""


class JudgeError(UpgradeReportError):
    """The judge returned something unusable (refusal, truncation, unparseable)."""


class JudgeDriftError(JudgeError):
    """The judge answered with a different model than the pinned one."""
