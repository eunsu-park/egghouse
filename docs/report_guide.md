# egghouse.report Usage Guide

Render a vault Markdown note (weekly report, experiment note) to an A4 PDF.

---

## Overview

`egghouse.report.md2pdf` converts one Obsidian note to PDF in three steps:

1. **Obsidian syntax** — `![[figure.png]]` becomes an image found anywhere under the vault
   root; `[[note]]` and `[[path/note|alias]]` become their display text.
2. **Markdown → HTML** with the `markdown` package (tables, fenced code), styled for print:
   bordered tables with a grey header, grey quote boxes, inline code on grey.
3. **HTML → PDF** with headless Chromium through Playwright (A4, 16 mm / 15 mm margins).

Korean notes (`*.ko.md`) render with Noto Sans CJK KR.

---

## One-time setup

```bash
conda create -n mdpdf python=3.12
conda activate mdpdf
pip install -e "<path to egghouse>[report]"   # markdown, playwright
conda install -c conda-forge alsa-lib         # libasound.so.2 for Chromium, no sudo
NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt python -m playwright install chromium
```

- **`alsa-lib`** — the Chromium headless shell needs `libasound.so.2`. If the system lacks it,
  the conda-forge copy is used: `render_pdf` puts `$CONDA_PREFIX/lib` on Chromium's
  `LD_LIBRARY_PATH`. Check with
  `ldd ~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell | grep "not found"`.
- **`NODE_EXTRA_CA_CERTS`** — on a network that re-signs TLS (the download fails with
  `SELF_SIGNED_CERT_IN_CHAIN`), point the Playwright downloader at the system CA bundle. Do not
  disable certificate checks.
- **Korean font** — if `fc-list : family | grep -i cjk` is empty, install Noto Sans CJK KR:

  ```bash
  mkdir -p ~/.local/share/fonts && cd ~/.local/share/fonts
  for w in Regular Bold; do
    curl -fsSLO https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-$w.otf
  done
  fc-cache -f
  ```

  Without it Korean text renders as boxes.

---

## Usage

```bash
conda activate mdpdf
# writes 2026-W41.pdf next to the note (overwrites)
python -m egghouse.report ~/Vaults/GeoIndex/planning/weekly-reports/2026-W41.md
python -m egghouse.report ~/Vaults/GeoIndex/planning/weekly-reports/2026-W41.ko.md

# render elsewhere first, then copy after checking
python -m egghouse.report <note.md> --out /tmp/check.pdf
```

`--vault` sets the root searched for embedded figures. The default is three levels above the
note (`<vault>/<area>/<folder>/<note>.md`, e.g. `planning/weekly-reports/`).

```python
from egghouse.report import render_pdf
render_pdf("2026-W41.md", vault="~/Vaults/GeoIndex", out="/tmp/2026-W41.pdf")
```

---

## Checks after rendering

- Page count and layout look like the previous week's PDF.
- No raw Markdown in the text: no `[[`, `|---`, `**`.
- Every figure appears (an embed whose file was not found is left as `![[name]]`).

The English PDF uses DejaVu Sans when Helvetica Neue is absent (Linux); the layout is the same as a
macOS print, the glyphs differ.
