"""Report rendering: vault Markdown notes to PDF (requires the ``report`` extra + Chromium).

Example:
    >>> from egghouse.report import render_pdf
    >>> render_pdf('planning/weekly-reports/2026-W41.md')
"""

from .md2pdf import render_pdf, resolve_links

__all__ = ["render_pdf", "resolve_links"]
