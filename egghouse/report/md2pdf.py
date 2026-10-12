"""Render an Obsidian vault note (e.g. a weekly report) to an A4 PDF with headless Chromium.

Obsidian embeds (``![[figure.png]]``) become images found anywhere under the vault root;
wikilinks (``[[note]]``, ``[[path/note|alias]]``) become their display text. The page is
then printed by Chromium (Playwright), the same Chrome/Skia engine as a browser print.

Usage:
    python -m egghouse.report <note.md> [--vault <root>] [--out <file.pdf>]
"""

import argparse
import os
import pathlib
import re
import sys

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, "Noto Sans CJK KR", sans-serif;
       font-size: 10pt; line-height: 1.5; color: #222; }
h1 { font-size: 19pt; line-height: 1.3; margin: 0 0 14pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; }
h3 { font-size: 12.5pt; margin: 16pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #ddd; }
h4 { font-size: 11pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.6pt; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3pt 6pt; vertical-align: top; text-align: left; }
th { background: #f3f3f3; font-weight: bold; }
blockquote { margin: 10pt 0; padding: 6pt 12pt; background: #f6f6f6; border-left: 4px solid #ccc; color: #555; }
code { font-family: Menlo, "DejaVu Sans Mono", monospace; font-size: 8.6pt; background: #f2f2f2;
       padding: 0 2pt; border-radius: 2px; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
hr { border: 0; border-top: 1px solid #ddd; margin: 12pt 0; }
li { margin: 2pt 0; }
"""


def resolve_links(text: str, vault: pathlib.Path) -> str:
    """Turn Obsidian embeds into <img> tags and wikilinks into their display text.

    Args:
        text: Markdown source.
        vault: Vault root searched (recursively) for embedded files.

    Returns:
        Markdown with no ``[[...]]`` left except embeds whose file was not found.
    """
    def embed(m):
        hits = list(vault.rglob(m.group(1)))
        return f'<img src="{hits[0].as_uri()}">' if hits else m.group(0)
    text = re.sub(r"!\[\[([^\]|]+)\]\]", embed, text)
    text = re.sub(r"(?<!!)\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    return re.sub(r"(?<!!)\[\[([^\]]+)\]\]", lambda m: m.group(1).split("/")[-1], text)


def render_pdf(src, vault=None, out=None) -> pathlib.Path:
    """Render one Markdown note to PDF.

    Args:
        src: Path to the ``.md`` note.
        vault: Vault root for resolving embeds; default is three levels above ``src``
            (``<vault>/<area>/<folder>/<note>.md``).
        out: Output path; default is ``src`` with a ``.pdf`` suffix (overwrites).

    Returns:
        The path of the written PDF.
    """
    import markdown
    from playwright.sync_api import sync_playwright

    src = pathlib.Path(src).expanduser().resolve()
    vault = pathlib.Path(vault).expanduser().resolve() if vault else src.parents[2]
    out = pathlib.Path(out).expanduser() if out else src.with_suffix(".pdf")
    body = markdown.markdown(resolve_links(src.read_text(), vault),
                             extensions=["tables", "fenced_code", "sane_lists"])
    lang = "ko" if src.name.endswith(".ko.md") else "en"
    html = f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><style>{CSS}</style><body>{body}</body></html>'
    tmp = src.with_suffix(".tmp.html")
    tmp.write_text(html)
    try:
        with sync_playwright() as p:
            # libasound may come from the active conda env (conda-forge alsa-lib), not the system.
            env = {**os.environ, "LD_LIBRARY_PATH": f"{sys.prefix}/lib"}
            page = p.chromium.launch(env=env).new_page()
            page.goto(tmp.as_uri(), wait_until="networkidle")
            page.pdf(path=str(out), format="A4", print_background=True,
                     margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"})
    finally:
        tmp.unlink()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", help="Markdown note to render")
    p.add_argument("--vault", help="vault root for ![[embeds]] (default: three levels above src)")
    p.add_argument("--out", help="output PDF (default: src with .pdf, overwritten)")
    a = p.parse_args()
    print("wrote", render_pdf(a.src, a.vault, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
