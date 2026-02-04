"""Tests for egghouse.sdo.quality module."""

import pytest

from egghouse.sdo import (
    decode_quality,
    format_quality,
    is_quality_ok,
    get_quality_summary,
)


class TestDecodeQuality:
    """Tests for decode_quality function."""

    def test_nominal_quality(self):
        """Test nominal (zero) quality."""
        result = decode_quality(0)
        assert len(result) == 1
        assert result[0]["bit"] == -1
        assert result[0]["description"] == "nominal"
        assert result[0]["severity"] == "ok"

    def test_single_bit_flatfield(self):
        """Test single bit: flatfield not available."""
        result = decode_quality(0x1)  # Bit 0
        assert len(result) == 1
        assert result[0]["bit"] == 0
        assert "Flatfield" in result[0]["description"]
        assert result[0]["severity"] == "minor"

    def test_single_bit_iss_open(self):
        """Test single bit: ISS loop open."""
        result = decode_quality(0x20000)  # Bit 17
        assert len(result) == 1
        assert result[0]["bit"] == 17
        assert "ISS" in result[0]["description"]
        assert result[0]["severity"] == "caution"

    def test_multiple_bits(self):
        """Test multiple bits set."""
        result = decode_quality(0x30000)  # Bits 16 and 17
        assert len(result) == 2
        bits = [r["bit"] for r in result]
        assert 16 in bits
        assert 17 in bits

    def test_eclipse_bit(self):
        """Test eclipse flag."""
        result = decode_quality(0x2000)  # Bit 13
        assert len(result) == 1
        assert result[0]["bit"] == 13
        assert "eclipse" in result[0]["description"].lower()
        assert result[0]["severity"] == "severe"

    def test_hmi_instrument(self):
        """Test HMI-specific bit definitions."""
        result = decode_quality(0x40000, instrument="HMI")  # Bit 18
        assert len(result) == 1
        assert "Motor" in result[0]["description"] or "encoder" in result[0]["description"].lower()

    def test_case_insensitive_instrument(self):
        """Test that instrument parameter is case-insensitive."""
        result_upper = decode_quality(0x1, instrument="AIA")
        result_lower = decode_quality(0x1, instrument="aia")
        assert result_upper == result_lower

    def test_image_not_available(self):
        """Test bit 31: image not available."""
        result = decode_quality(0x80000000)  # Bit 31
        assert len(result) == 1
        assert result[0]["bit"] == 31
        assert "not available" in result[0]["description"].lower()
        assert result[0]["severity"] == "severe"


class TestFormatQuality:
    """Tests for format_quality function."""

    def test_format_nominal(self):
        """Test formatting nominal quality."""
        formatted = format_quality(0)
        assert "QUALITY = 0x0" in formatted
        assert "nominal" in formatted.lower()

    def test_format_with_issues(self):
        """Test formatting quality with issues."""
        formatted = format_quality(0x30000)
        assert "QUALITY = 0x30000" in formatted
        assert "2 issue(s)" in formatted
        assert "Bit 16" in formatted
        assert "Bit 17" in formatted

    def test_format_verbose_true(self):
        """Test verbose formatting."""
        formatted = format_quality(0x20000, verbose=True)
        assert "ISS loop open" in formatted
        assert "caution" in formatted.lower()

    def test_format_verbose_false(self):
        """Test non-verbose formatting."""
        formatted = format_quality(0x20000, verbose=False)
        assert "Bit 17" in formatted
        assert "0x20000" in formatted
        # Description should not be included
        lines = formatted.split('\n')
        for line in lines:
            if "Bit 17" in line:
                assert "ISS" not in line


class TestIsQualityOk:
    """Tests for is_quality_ok function."""

    def test_nominal_is_ok(self):
        """Test that zero quality is OK."""
        assert is_quality_ok(0) is True

    def test_minor_issue_ok_in_normal_mode(self):
        """Test that minor issues are OK in non-strict mode."""
        assert is_quality_ok(0x1) is True  # Flatfield not available
        assert is_quality_ok(0x2) is True  # Orbit data not available
        assert is_quality_ok(0x1F) is True  # All minor bits

    def test_minor_issue_fails_in_strict_mode(self):
        """Test that minor issues fail in strict mode."""
        assert is_quality_ok(0x1, strict=True) is False
        assert is_quality_ok(0x1F, strict=True) is False

    def test_severe_issue_fails(self):
        """Test that severe issues always fail."""
        assert is_quality_ok(0x2000) is False  # Eclipse
        assert is_quality_ok(0x4000) is False  # No sun presence
        assert is_quality_ok(0x80000000) is False  # Not available

    def test_caution_issue_fails(self):
        """Test that caution issues fail in normal mode."""
        assert is_quality_ok(0x400) is False  # 5% missing
        assert is_quality_ok(0x20000) is False  # ISS open

    def test_ignore_bits(self):
        """Test ignoring specific bits."""
        # Eclipse (bit 13) normally fails
        assert is_quality_ok(0x2000) is False
        # But can be ignored
        assert is_quality_ok(0x2000, ignore_bits=[13]) is True

    def test_ignore_bits_strict_mode(self):
        """Test ignoring bits in strict mode."""
        # Flatfield (bit 0) fails in strict mode
        assert is_quality_ok(0x1, strict=True) is False
        # But can be ignored
        assert is_quality_ok(0x1, strict=True, ignore_bits=[0]) is True

    def test_ignore_multiple_bits(self):
        """Test ignoring multiple bits."""
        quality = 0x30000  # Bits 16 and 17
        assert is_quality_ok(quality) is False
        assert is_quality_ok(quality, ignore_bits=[17]) is True  # Still has bit 16 (info)


class TestGetQualitySummary:
    """Tests for get_quality_summary function."""

    def test_nominal_summary(self):
        """Test summary for nominal quality."""
        summary = get_quality_summary(0)
        assert summary["is_nominal"] is True
        assert summary["is_usable"] is True
        assert len(summary["issues"]) == 0
        assert len(summary["active_bits"]) == 0

    def test_summary_with_issues(self):
        """Test summary with issues."""
        summary = get_quality_summary(0x30000)
        assert summary["is_nominal"] is False
        assert summary["quality"] == 0x30000
        assert summary["quality_hex"] == "0x30000"
        assert len(summary["active_bits"]) == 2
        assert 16 in summary["active_bits"]
        assert 17 in summary["active_bits"]

    def test_summary_severity_counts(self):
        """Test severity counts in summary."""
        # Bit 16 = info (dark image)
        # Bit 17 = caution (ISS open)
        summary = get_quality_summary(0x30000)
        assert "info" in summary["severity_counts"]
        assert "caution" in summary["severity_counts"]
        assert summary["severity_counts"]["info"] == 1
        assert summary["severity_counts"]["caution"] == 1

    def test_summary_is_usable(self):
        """Test is_usable field."""
        # Minor issues are usable
        summary = get_quality_summary(0x1)
        assert summary["is_usable"] is True

        # Severe issues are not usable
        summary = get_quality_summary(0x2000)  # Eclipse
        assert summary["is_usable"] is False

    def test_summary_hmi_instrument(self):
        """Test summary with HMI instrument."""
        summary = get_quality_summary(0x40000, instrument="HMI")
        assert len(summary["issues"]) == 1
        assert "Motor" in summary["issues"][0]["description"] or "encoder" in summary["issues"][0]["description"].lower()
