"""Backward-compatible alias for the active slcosmo tools module."""

try:
    from .tools import tool
except ImportError:  # pragma: no cover - supports direct script-style imports.
    from tools import tool

__all__ = ["tool"]
