"""CLI: python -m egghouse.report <note.md> [--vault <root>] [--out <file.pdf>]."""

import sys

from .md2pdf import main

sys.exit(main())
