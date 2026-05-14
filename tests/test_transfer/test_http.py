"""Tests for egghouse.transfer.http retry semantics."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from egghouse.transfer import http as transfer_http
from egghouse.transfer.http import (
    _is_transient_error,
    download_single_file,
    get_file_list,
)


# --- _is_transient_error ---


def test_is_transient_connection_error():
    assert _is_transient_error(requests.ConnectionError("DNS fail")) is True


def test_is_transient_timeout():
    assert _is_transient_error(requests.Timeout("read timeout")) is True


def test_is_transient_http_500():
    resp = MagicMock(status_code=503)
    err = requests.HTTPError("503 Server Error")
    err.response = resp
    assert _is_transient_error(err) is True


def test_is_terminal_http_404():
    resp = MagicMock(status_code=404)
    err = requests.HTTPError("404 Not Found")
    err.response = resp
    assert _is_transient_error(err) is False


def test_is_terminal_http_403():
    resp = MagicMock(status_code=403)
    err = requests.HTTPError("403 Forbidden")
    err.response = resp
    assert _is_transient_error(err) is False


def test_is_transient_http_408_timeout_response():
    resp = MagicMock(status_code=408)
    err = requests.HTTPError("408 Request Timeout")
    err.response = resp
    assert _is_transient_error(err) is True


# --- get_file_list ---


def _make_response(status_code: int, body: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = body
    if 400 <= status_code < 600:
        err = requests.HTTPError(f"{status_code} error")
        err.response = resp
        resp.raise_for_status = MagicMock(side_effect=err)
    else:
        resp.raise_for_status = MagicMock()
    return resp


def test_get_file_list_404_returns_empty_no_retry(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _make_response(404)

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    out = get_file_list("https://example/nope", ["fts"], max_retries=3)
    assert out == []
    assert calls["n"] == 1  # 404 should NOT trigger retry


def test_get_file_list_403_returns_empty_no_retry(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _make_response(403)

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    out = get_file_list("https://example/", ["fts"], max_retries=3)
    assert out == []
    assert calls["n"] == 1


def test_get_file_list_connection_error_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        raise requests.ConnectionError("DNS fail")

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    # Patch sleep to keep the test fast.
    monkeypatch.setattr(transfer_http.time, "sleep", lambda s: None)

    with pytest.raises(requests.ConnectionError):
        get_file_list("https://example/", ["fts"], max_retries=2)
    assert calls["n"] == 3  # initial + 2 retries


def test_get_file_list_500_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _make_response(503)

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    monkeypatch.setattr(transfer_http.time, "sleep", lambda s: None)

    with pytest.raises(requests.HTTPError):
        get_file_list("https://example/", ["fts"], max_retries=1)
    assert calls["n"] == 2  # initial + 1 retry


def test_get_file_list_recovers_on_second_attempt(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("transient")
        body = '<a href="a.fts">a.fts</a><a href="b.fts">b.fts</a>'
        return _make_response(200, body)

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    monkeypatch.setattr(transfer_http.time, "sleep", lambda s: None)

    out = get_file_list("https://example/", ["fts"], max_retries=3)
    assert out == ["a.fts", "b.fts"]
    assert calls["n"] == 2


def test_get_file_list_parses_extensions_case_insensitively(monkeypatch):
    body = (
        '<a href="a.FTS">a.FTS</a>'
        '<a href="b.fts">b.fts</a>'
        '<a href="c.txt">c.txt</a>'
    )
    monkeypatch.setattr(
        transfer_http.requests, "get",
        lambda url, **k: _make_response(200, body),
    )
    out = get_file_list("https://example/", ["fts"], max_retries=0)
    assert sorted(out) == ["a.FTS", "b.fts"]


# --- download_single_file ---


def test_download_single_file_404_returns_false_no_retry(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _make_response(404)

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    dest = tmp_path / "nope.fts"
    ok = download_single_file("https://example/nope.fts", str(dest), max_retries=3)
    assert ok is False
    assert calls["n"] == 1
    assert not dest.exists()


def test_download_single_file_connection_error_retries_then_returns_false(
    monkeypatch, tmp_path
):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        raise requests.ConnectionError("DNS fail")

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    monkeypatch.setattr(transfer_http.time, "sleep", lambda s: None)
    dest = tmp_path / "x.fts"
    ok = download_single_file("https://example/x.fts", str(dest), max_retries=2)
    assert ok is False
    assert calls["n"] == 3  # initial + 2 retries


def test_download_single_file_recovers_on_second_attempt(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("read timeout")
        resp = _make_response(200, "body")
        resp.content = b"data"
        return resp

    monkeypatch.setattr(transfer_http.requests, "get", fake_get)
    monkeypatch.setattr(transfer_http.time, "sleep", lambda s: None)
    dest = tmp_path / "ok.fts"
    ok = download_single_file("https://example/ok.fts", str(dest), max_retries=3)
    assert ok is True
    assert calls["n"] == 2
    assert dest.read_bytes() == b"data"


def test_download_single_file_skips_when_exists_and_not_overwrite(tmp_path):
    dest = tmp_path / "existing.fts"
    dest.write_bytes(b"old")
    ok = download_single_file("https://example/x.fts", str(dest), overwrite=False)
    assert ok is True
    assert dest.read_bytes() == b"old"
