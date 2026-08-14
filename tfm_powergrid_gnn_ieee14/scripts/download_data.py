from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
expected = ROOT / "data/raw/ieee14_fault/IEEE 14 bus test system data.xlsx"

print("Primary dataset: IEEE 14 bus test systems row data")
print("Figshare DOI: 10.6084/m9.figshare.30590399")
print("Expected local path:", expected)

if expected.exists():
    print("\n✓ Dataset already present. No download is required.")
    print("Next: python scripts/inspect_data.py")
else:
    print("\nThe dataset is distributed through Figshare, not KaggleHub.")
    print("Download the Excel file from the DOI page and place it at the path above.")
    raise SystemExit(1)
