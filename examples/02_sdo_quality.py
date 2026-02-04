#!/usr/bin/env python
"""
SDO Quality Flag Interpretation Example
=======================================

Demonstrates how to interpret SDO QUALITY flags from FITS headers.
No external data files required - uses example quality values.

Run:
    python examples/02_sdo_quality.py
"""

from egghouse.sdo import (
    decode_quality,
    format_quality,
    is_quality_ok,
    get_quality_summary,
)


def main():
    print("=" * 60)
    print("egghouse - SDO Quality Flag Interpretation")
    print("=" * 60)

    # Example quality values commonly encountered
    examples = [
        (0, "Nominal operation"),
        (0x1, "Flatfield data not available"),
        (0x100, "Some missing pixels"),
        (0x2000, "Spacecraft eclipse"),
        (0x20000, "ISS loop open"),
        (0x30000, "Dark image + ISS open"),
        (0x80000000, "Image not available"),
    ]

    # 1. Decode individual quality flags
    print("\n1. Decoding Quality Flags")
    print("-" * 40)

    for quality, description in examples:
        print(f"\nQuality = {hex(quality)} ({description}):")
        decoded = decode_quality(quality)
        for item in decoded:
            if item["bit"] == -1:
                print(f"   -> nominal (no issues)")
            else:
                print(f"   -> Bit {item['bit']}: {item['description']} [{item['severity']}]")

    # 2. Check if quality is acceptable
    print("\n\n2. Quality Acceptance Check")
    print("-" * 40)
    print(f"{'Quality':<15} {'Normal Mode':<12} {'Strict Mode':<12}")
    print("-" * 40)

    for quality, description in examples:
        ok_normal = is_quality_ok(quality)
        ok_strict = is_quality_ok(quality, strict=True)
        print(f"{hex(quality):<15} {str(ok_normal):<12} {str(ok_strict):<12}")

    # 3. Formatted output
    print("\n\n3. Formatted Quality Output")
    print("-" * 40)
    print(format_quality(0x30000))

    # 4. Summary dictionary (for programmatic use)
    print("\n\n4. Quality Summary Dictionary")
    print("-" * 40)
    summary = get_quality_summary(0x30000)
    print(f"Quality:        {summary['quality_hex']}")
    print(f"Is nominal:     {summary['is_nominal']}")
    print(f"Is usable:      {summary['is_usable']}")
    print(f"Active bits:    {summary['active_bits']}")
    print(f"Severity counts: {summary['severity_counts']}")

    # 5. Practical usage pattern
    print("\n\n5. Practical Usage Pattern")
    print("-" * 40)
    print("""
# In your analysis code:
from egghouse.sdo import is_quality_ok

# When reading FITS files:
quality = header['QUALITY']  # From FITS header

if is_quality_ok(quality):
    # Process the data
    pass
elif is_quality_ok(quality, strict=False):
    # Data has minor issues but still usable
    print("Warning: Minor quality issues")
else:
    # Skip this data
    print("Error: Severe quality issues, skipping")
""")

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
