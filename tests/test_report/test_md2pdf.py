"""Tests for egghouse.report.md2pdf link resolution (no browser needed)."""

from egghouse.report.md2pdf import resolve_links


def test_wikilink_plain_and_alias(tmp_path):
    out = resolve_links("see [[l2-lead1-images]] and [[reference/GLOSSARY|GLOSSARY]]", tmp_path)
    assert out == "see l2-lead1-images and GLOSSARY"


def test_wikilink_path_keeps_basename(tmp_path):
    assert resolve_links("[[planning/2026-W36]]", tmp_path) == "2026-W36"


def test_embed_found_becomes_img(tmp_path):
    fig = tmp_path / "experiments" / "figures" / "curve.png"
    fig.parent.mkdir(parents=True)
    fig.write_bytes(b"")
    assert resolve_links("![[curve.png]]", tmp_path) == f'<img src="{fig.as_uri()}">'


def test_embed_missing_is_left(tmp_path):
    assert resolve_links("![[missing.png]]", tmp_path) == "![[missing.png]]"
