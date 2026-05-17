"""Generalized FITS-directory registration.

`register_fits_dir` scans a directory tree for FITS files, validates
each via a `FitsHandler`, bulk-upserts the valid ones into a DB table,
and (optionally) moves files into an archive layout. The flow is
generalized from the setup-sw-db `scripts/register_sdo.py` so it works
for any handler/table, not just SDO.

Validation is header-only by default and I/O-bound, so parallelism
uses a thread pool (no pickling constraints; the handler may hold
state). The DB upsert is delegated to
`egghouse.database.upsert_dataframe` and is idempotent — re-running
over an already-registered tree inserts nothing.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .handlers import FitsHandler
from .result import ValidationResult


@dataclass
class RegisterReport:
    scanned: int = 0
    valid: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    errors: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"scanned files          : {self.scanned}",
            f"valid                  : {self.valid}",
            f"inserted (DB)          : {self.inserted}",
            f"skipped (target exists): {self.skipped_existing}",
        ]
        if self.errors:
            lines.append("errors:")
            for cat, n in sorted(self.errors.items()):
                lines.append(f"  {cat}: {n}")
        return "\n".join(lines)


def scan_fits(
    scan_dir: str,
    *,
    pattern: str = "*.fits",
    exclude_substrings: Sequence[str] = ("spike",),
) -> List[Path]:
    """Recursively list FITS files, excluding name substrings.

    The default excludes AIA ``spike`` artifact files.
    """
    root = Path(scan_dir)
    if not root.exists():
        return []
    files = sorted(root.rglob(pattern))
    if exclude_substrings:
        lowered = tuple(s.lower() for s in exclude_substrings)
        files = [f for f in files if not any(s in f.name.lower() for s in lowered)]
    return files


def _validate_many(
    handler: FitsHandler, files: Sequence[Path], parallel: int
) -> List[Tuple[Path, ValidationResult]]:
    if parallel > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            results = list(ex.map(lambda f: (f, handler.extract_metadata(str(f))), files))
    else:
        results = [(f, handler.extract_metadata(str(f))) for f in files]
    return results


def register_fits_dir(
    scan_dir: str,
    *,
    handler: FitsHandler,
    table: str,
    db_config: Dict[str, Any],
    conflict_columns: Sequence[str],
    move_root: Optional[str] = None,
    error_dirs: Optional[Dict[str, str]] = None,
    pattern: str = "*.fits",
    exclude_substrings: Sequence[str] = ("spike",),
    parallel: int = 1,
    batch_size: int = 1000,
    verbose: bool = False,
) -> RegisterReport:
    """Scan, validate, upsert, and optionally archive a FITS tree.

    Args:
        scan_dir: Directory to scan recursively.
        handler: A `FitsHandler` (e.g. `AiaFitsHandler`).
        table: Target DB table.
        db_config: kwargs for `PostgresManager`.
        conflict_columns: Composite conflict target for the idempotent
            upsert (e.g. ``["telescope", "channel", "datetime"]``).
        move_root: If set, valid files are moved under
            ``handler.target_dir(move_root, metadata)``; files whose
            target already exists are counted as ``skipped_existing``
            and left in place. If None, files are registered where they
            are.
        error_dirs: Optional ``{error_category: subdir}`` under
            ``move_root`` to relocate invalid files into. Ignored when
            ``move_root`` is None.
        pattern / exclude_substrings: Passed to `scan_fits`.
        parallel: Thread workers for header validation.
        batch_size: Upsert batch size.

    Returns:
        `RegisterReport` with reconcilable counts (scanned == valid +
        sum(errors) + skipped_existing).
    """
    from egghouse.database import upsert_dataframe

    report = RegisterReport()
    files = scan_fits(scan_dir, pattern=pattern, exclude_substrings=exclude_substrings)
    report.scanned = len(files)
    if not files:
        return report

    for start in range(0, len(files), batch_size):
        batch = files[start : start + batch_size]
        validated = _validate_many(handler, batch, parallel)

        records: List[Dict[str, Any]] = []
        valid_moves: List[Tuple[Path, Path]] = []
        invalid_moves: List[Tuple[Path, Path]] = []

        for path, result in validated:
            if not result.success:
                report.errors[result.error] = report.errors.get(result.error, 0) + 1
                if move_root and error_dirs and result.error in error_dirs:
                    tgt = Path(move_root) / error_dirs[result.error] / path.name
                    if tgt != path:
                        invalid_moves.append((path, tgt))
                if verbose:
                    print(f"  {result.error}: {path.name}")
                continue

            meta = result.metadata
            if move_root:
                tgt = handler.target_dir(move_root, meta) / path.name
                if tgt == path:
                    final_path = path
                elif tgt.exists():
                    report.skipped_existing += 1
                    if verbose:
                        print(f"  skipped (exists): {path.name}")
                    continue
                else:
                    final_path = tgt
                    valid_moves.append((path, tgt))
            else:
                final_path = path

            records.append(handler.to_db_record(str(final_path), meta))
            report.valid += 1
            if verbose:
                print(f"  valid: {path.name}")

        if records:
            import pandas as pd

            inserted = upsert_dataframe(
                pd.DataFrame(records),
                table,
                db_config,
                conflict_columns=list(conflict_columns),
                batch=batch_size,
            )
            report.inserted += inserted

        # Move only after a successful DB write for the batch.
        for src, tgt in valid_moves:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(tgt))
        for src, tgt in invalid_moves:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(tgt))

    return report
