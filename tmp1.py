"""Merge the yearly RDU GHCNh CSV files into one CSV.

Run:
    python3 tmp1.py
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_FILE = BASE_DIR / "rdu_ghcnh_2015-01-01_to_2026-09-16.csv"


def merge_yearly_csvs(
    raw_dir=RAW_DIR,
    output_file=OUTPUT_FILE,
    start_year=2015,
    end_year=2026,
):
    """Merge one CSV from each yearly folder, retaining a single header.

    The expected layout is ``raw/2015/*.csv`` through ``raw/2026/*.csv``.
    Files are copied in ascending year order without loading the full dataset
    into memory. All input CSVs must have the same header.

    Returns the path to the merged CSV.
    """
    raw_dir = Path(raw_dir)
    output_file = Path(output_file)

    input_files = []
    for year in range(start_year, end_year + 1):
        matches = sorted((raw_dir / str(year)).glob("*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one CSV in {raw_dir / str(year)}, "
                f"found {len(matches)}"
            )
        input_files.append(matches[0])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(f".{output_file.name}.tmp")
    expected_header = None
    total_rows = 0

    try:
        with temporary_file.open("wb") as destination:
            for input_file in input_files:
                with input_file.open("rb") as source:
                    header = source.readline()
                    if not header:
                        raise ValueError(f"CSV is empty: {input_file}")

                    if expected_header is None:
                        expected_header = header
                        destination.write(header)
                    elif header != expected_header:
                        raise ValueError(
                            f"CSV header does not match the first file: {input_file}"
                        )

                    rows = 0
                    for line in source:
                        destination.write(line)
                        rows += 1
                    total_rows += rows
                    print(f"merged {input_file.relative_to(BASE_DIR)} ({rows} rows)")

        temporary_file.replace(output_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()

    print(f"saved {output_file.name} ({total_rows} rows)")
    return output_file


if __name__ == "__main__":
    merge_yearly_csvs()
