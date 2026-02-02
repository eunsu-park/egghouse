"""
SDO quality flag interpretation utilities.

Provides functions to decode and interpret the QUALITY keyword from
SDO/AIA and SDO/HMI FITS headers. The QUALITY keyword is a 32-bit
integer where each bit indicates a specific data quality issue or
non-nominal operating condition.

References:
    - JSOC documentation: http://jsoc.stanford.edu/jsocwiki/Lev1qualBits
    - AIA quality bits: https://github.com/LM-SAL/aiapy
    - SDO User Guide section 7.7.6
"""

from typing import Dict, List, Union, Optional


# AIA Quality Flag Bit Definitions
# Source: aiapy and JSOC documentation
AIA_QUALITY_BITS: Dict[int, str] = {
    0: "Flatfield data not available",
    1: "Orbit data not available",
    2: "Ancillary science data not available",
    3: "Master pointing data not available",
    4: "Limb-fit data not available",
    5: "Reserved",
    6: "Reserved",
    7: "Reserved",
    8: "MISSVALS > 0 (some pixels missing)",
    9: "MISSVALS > 1% of TOTVALS",
    10: "MISSVALS > 5% of TOTVALS",
    11: "MISSVALS > 25% of TOTVALS",
    12: "Spacecraft not in science pointing mode",
    13: "Spacecraft eclipse flag set",
    14: "Spacecraft sun presence flag not set",
    15: "Spacecraft safe mode flag set",
    16: "Dark image",
    17: "ISS loop open",
    18: "Calibration image",
    19: "Reserved",
    20: "Focus out of range",
    21: "Register flag set",
    22: "Reserved",
    23: "Reserved",
    24: "Reserved",
    25: "Reserved",
    26: "Reserved",
    27: "Reserved",
    28: "Reserved",
    29: "Reserved",
    30: "Quicklook image",
    31: "Image not available",
}

# HMI Quality Flag Bit Definitions
# Source: JSOC quallev0.h and HMI documentation
# HMI includes all common bits plus additional instrument-specific bits
HMI_QUALITY_BITS: Dict[int, str] = {
    # Common bits (same as AIA)
    0: "Flatfield data not available",
    1: "Orbit data not available",
    2: "Ancillary science data not available",
    3: "Master pointing data not available",
    4: "Limb-fit data not available",
    5: "Reserved",
    6: "Reserved",
    7: "Reserved",
    8: "MISSVALS > 0 (some pixels missing)",
    9: "MISSVALS > 1% of TOTVALS",
    10: "MISSVALS > 5% of TOTVALS",
    11: "MISSVALS > 25% of TOTVALS",
    12: "Spacecraft not in science pointing mode",
    13: "Spacecraft eclipse flag set",
    14: "Spacecraft sun presence flag not set",
    15: "Spacecraft safe mode flag set (or camera anomaly)",
    16: "Dark image",
    17: "ISS loop open",
    18: "HMI Focus/Cal Motor 1 encoder error",
    19: "HMI Focus/Cal Motor 2 encoder error",
    20: "HMI Polarization Motor 1 encoder error",
    21: "HMI Polarization Motor 2 encoder error",
    22: "HMI Polarization Motor 3 encoder error",
    23: "HMI Wavelength Motor 1 encoder error",
    24: "HMI Wavelength Motor 2 encoder error",
    25: "Reserved",
    26: "Reserved",
    27: "Reserved",
    28: "Reserved",
    29: "Reserved",
    30: "Quicklook image",
    31: "Image not available",
}

# QUALLEV0 bits (Level 0 processing quality)
# From JSOC imgdecode flags
QUALLEV0_BITS: Dict[int, str] = {
    0: "Overflow flag set",
    1: "Header error flag set",
    2: "Compression error in image",
    3: "Last pixel error",
    12: "Misconfiguration (likely manual)",
    13: "Instrument anomaly",
    15: "Camera anomaly",
}

# Quality severity levels for user guidance
QUALITY_SEVERITY: Dict[int, str] = {
    0: "minor",      # Flatfield
    1: "minor",      # Orbit data
    2: "minor",      # Ancillary
    3: "minor",      # Master pointing
    4: "minor",      # Limb-fit
    8: "warning",    # Some missing pixels
    9: "warning",    # 1% missing
    10: "caution",   # 5% missing
    11: "severe",    # 25% missing
    12: "caution",   # Not science pointing
    13: "severe",    # Eclipse
    14: "severe",    # No sun presence
    15: "severe",    # Safe mode
    16: "info",      # Dark image
    17: "caution",   # ISS open
    18: "info",      # Calibration
    20: "caution",   # Focus out of range
    21: "info",      # Register flag
    30: "info",      # Quicklook
    31: "severe",    # Not available
}


def decode_quality(
    quality: int,
    instrument: str = "AIA"
) -> List[Dict[str, Union[int, str]]]:
    """
    Decode SDO QUALITY flag into a list of active quality issues.

    The QUALITY keyword in SDO FITS headers is a 32-bit integer where
    each bit indicates a specific data quality issue. This function
    decodes the flag and returns human-readable descriptions.

    Args:
        quality: The QUALITY value from FITS header (32-bit integer).
        instrument: Instrument name, either "AIA" or "HMI".
            Determines which bit definitions to use.

    Returns:
        List of dictionaries, each containing:
            - bit: The bit number (0-31)
            - hex: Hexadecimal representation of the bit
            - description: Human-readable description of the issue
            - severity: Severity level (info, minor, warning, caution, severe)

        Returns [{"bit": -1, "description": "nominal"}] if quality == 0.

    Example:
        >>> decode_quality(0)
        [{'bit': -1, 'hex': '0x0', 'description': 'nominal', 'severity': 'ok'}]

        >>> decode_quality(0x20000)  # ISS loop open
        [{'bit': 17, 'hex': '0x20000', 'description': 'ISS loop open', 'severity': 'caution'}]

        >>> decode_quality(0x30000, instrument="HMI")  # Multiple flags
        [{'bit': 16, 'hex': '0x10000', 'description': 'Dark image', 'severity': 'info'},
         {'bit': 17, 'hex': '0x20000', 'description': 'ISS loop open', 'severity': 'caution'}]
    """
    if quality == 0:
        return [{"bit": -1, "hex": "0x0", "description": "nominal", "severity": "ok"}]

    # Select appropriate bit definitions
    instrument = instrument.upper()
    if instrument == "HMI":
        bit_definitions = HMI_QUALITY_BITS
    else:
        bit_definitions = AIA_QUALITY_BITS

    results = []
    for bit in range(32):
        if quality & (1 << bit):
            hex_val = hex(1 << bit)
            desc = bit_definitions.get(bit, f"Unknown bit {bit}")
            severity = QUALITY_SEVERITY.get(bit, "unknown")
            results.append({
                "bit": bit,
                "hex": hex_val,
                "description": desc,
                "severity": severity,
            })

    return results


def format_quality(
    quality: int,
    instrument: str = "AIA",
    verbose: bool = True
) -> str:
    """
    Format SDO QUALITY flag as a human-readable string.

    Provides a formatted string representation of the quality flag,
    suitable for printing or logging.

    Args:
        quality: The QUALITY value from FITS header.
        instrument: Instrument name, either "AIA" or "HMI".
        verbose: If True, include detailed descriptions.
            If False, only show bit numbers and hex values.

    Returns:
        Formatted string describing the quality issues.

    Example:
        >>> print(format_quality(0))
        QUALITY = 0x0 (0)
        Status: nominal

        >>> print(format_quality(0x30000, instrument="HMI"))
        QUALITY = 0x30000 (196608)
        Status: 2 issue(s) detected
          [Bit 16] 0x10000: Dark image (info)
          [Bit 17] 0x20000: ISS loop open (caution)
    """
    decoded = decode_quality(quality, instrument)
    lines = [f"QUALITY = {hex(quality)} ({quality})"]

    if decoded[0]["bit"] == -1:
        lines.append("Status: nominal")
    else:
        lines.append(f"Status: {len(decoded)} issue(s) detected")
        for item in decoded:
            if verbose:
                lines.append(
                    f"  [Bit {item['bit']:2d}] {item['hex']}: "
                    f"{item['description']} ({item['severity']})"
                )
            else:
                lines.append(f"  [Bit {item['bit']:2d}] {item['hex']}")

    return "\n".join(lines)


def is_quality_ok(
    quality: int,
    strict: bool = False,
    ignore_bits: Optional[List[int]] = None
) -> bool:
    """
    Check if SDO QUALITY flag indicates usable data.

    Determines whether the data quality is acceptable for analysis.
    In non-strict mode, minor issues (bits 0-4) are ignored.

    Args:
        quality: The QUALITY value from FITS header.
        strict: If True, any non-zero quality fails the check.
            If False, only severe issues fail the check.
        ignore_bits: Optional list of bit numbers to ignore in the check.

    Returns:
        True if quality is acceptable, False otherwise.

    Example:
        >>> is_quality_ok(0)
        True

        >>> is_quality_ok(0x1)  # Flatfield not available (minor)
        True

        >>> is_quality_ok(0x1, strict=True)
        False

        >>> is_quality_ok(0x2000)  # Eclipse (severe)
        False
    """
    if quality == 0:
        return True

    if strict:
        if ignore_bits:
            mask = quality
            for bit in ignore_bits:
                mask &= ~(1 << bit)
            return mask == 0
        return False

    # Non-strict mode: check for severe issues
    # Ignore bits 0-4 (minor calibration issues) and info bits
    severe_mask = 0
    for bit in range(32):
        severity = QUALITY_SEVERITY.get(bit, "unknown")
        if severity in ("severe", "caution"):
            severe_mask |= (1 << bit)

    if ignore_bits:
        for bit in ignore_bits:
            severe_mask &= ~(1 << bit)

    return (quality & severe_mask) == 0


def get_quality_summary(quality: int, instrument: str = "AIA") -> Dict:
    """
    Get a summary dictionary of quality flag information.

    Provides a structured summary suitable for programmatic use,
    including counts by severity level.

    Args:
        quality: The QUALITY value from FITS header.
        instrument: Instrument name, either "AIA" or "HMI".

    Returns:
        Dictionary containing:
            - quality: Original quality value
            - quality_hex: Hex representation
            - is_nominal: Boolean indicating nominal operation
            - is_usable: Boolean indicating data is likely usable
            - issues: List of decoded issues
            - severity_counts: Dict of counts by severity level
            - active_bits: List of active bit numbers

    Example:
        >>> summary = get_quality_summary(0x30000)
        >>> summary["is_nominal"]
        False
        >>> summary["severity_counts"]
        {'info': 1, 'caution': 1}
    """
    decoded = decode_quality(quality, instrument)
    is_nominal = (decoded[0]["bit"] == -1)

    severity_counts = {}
    active_bits = []

    if not is_nominal:
        for item in decoded:
            sev = item["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            active_bits.append(item["bit"])

    return {
        "quality": quality,
        "quality_hex": hex(quality),
        "is_nominal": is_nominal,
        "is_usable": is_quality_ok(quality),
        "issues": decoded if not is_nominal else [],
        "severity_counts": severity_counts,
        "active_bits": active_bits,
    }


def print_all_quality_bits(instrument: str = "AIA") -> None:
    """
    Print all defined quality bit meanings for reference.

    Useful for understanding what each bit in the QUALITY flag means.

    Args:
        instrument: Instrument name, either "AIA" or "HMI".

    Example:
        >>> print_all_quality_bits("AIA")
        AIA QUALITY Bit Definitions
        ===========================
        Bit  0 (0x00000001): Flatfield data not available
        Bit  1 (0x00000002): Orbit data not available
        ...
    """
    instrument = instrument.upper()
    bits = HMI_QUALITY_BITS if instrument == "HMI" else AIA_QUALITY_BITS

    print(f"{instrument} QUALITY Bit Definitions")
    print("=" * (len(instrument) + 24))

    for bit in range(32):
        hex_val = f"0x{(1 << bit):08X}"
        desc = bits.get(bit, "Reserved")
        severity = QUALITY_SEVERITY.get(bit, "-")
        print(f"Bit {bit:2d} ({hex_val}): {desc} [{severity}]")
