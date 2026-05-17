"""Type-safe result for FITS validation / metadata extraction.

Ported from the setup-sw-db reference project. Callers check
``result.success`` rather than ``isinstance(result, str)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ValidationResult:
    """Result of FITS validation + metadata extraction.

    Attributes:
        success: Whether validation succeeded.
        metadata: Extracted metadata on success, else None.
        error: Error category on failure (e.g. ``invalid_file``,
            ``invalid_header``, ``invalid_data``, ``non_zero_quality``),
            else None.
        file_path: The file the result refers to.
    """

    success: bool
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    file_path: Optional[str] = None

    @classmethod
    def ok(cls, metadata: Dict[str, Any], file_path: Optional[str] = None) -> "ValidationResult":
        return cls(success=True, metadata=metadata, file_path=file_path)

    @classmethod
    def fail(cls, error: str, file_path: Optional[str] = None) -> "ValidationResult":
        return cls(success=False, error=error, file_path=file_path)
