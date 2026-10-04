"""Compatibility import for the former human_oracle module."""

from .human_guide import HumanGuideBackend, HumanOracleBackend

__all__ = ["HumanGuideBackend", "HumanOracleBackend"]
