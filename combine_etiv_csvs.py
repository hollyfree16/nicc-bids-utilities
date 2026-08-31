#!/usr/bin/env python3
"""
Combine multiple <session>_etiv.csv files (e.g. one per pipeline run, each
sitting in a date-stamped folder) into a single CSV with a run-date column.

Each input CSV is expected to have the header written by
format_talairach_etiv_batch.py:
    subject,session,etiv

The run date for each input is taken from --dates (if given, one per
--input, in order) or else inferred from the input file's parent folder
name, which must be an 8-digit YYYYMMDD (e.g. .../20260726/ses-001_etiv.csv).

Output CSV columns: subject,session,etiv,run-date
    - run-date is written as YYYY-MM-DD

If the same (subject, session) appears in more than one input file, all
occurrences are kept by default and a warning is printed to stderr -- pass
--dedupe-keep-latest to instead keep only the row from the latest run-date.

Usage
-----
    python combine_etiv_csvs.py \\
        --input 20260726/ses-001_etiv.csv 20260831/ses-001_etiv.csv \\
        --output combined/ses-001_etiv.csv

    # explicit dates instead of inferring from folder names
    python combine_etiv_csvs.py \\
        --input runA/ses-001_etiv.csv runB/ses-001_etiv.csv \\
        --dates 20260726 20260831 \\
        --output combined/ses-001_etiv.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DATE_RE = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$")


def format_date(raw: str) -> str:
    m = DATE_RE.match(raw.strip())
    if not m:
        sys.exit(f"ERROR: could not parse date {raw!r} as YYYYMMDD")
    return f"{m.group('y')}-{m.group('m')}-{m.group('d')}"


def infer_date(csv_path: Path) -> str:
    folder_name = csv_path.resolve().parent.name
    if not DATE_RE.match(folder_name):
        sys.exit(
            f"ERROR: parent folder of {csv_path} is {folder_name!r}, not an "
            "8-digit YYYYMMDD date -- pass --dates explicitly instead"
        )
    return format_date(folder_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, type=Path,
                     help="Two or more <session>_etiv.csv files to combine")
    ap.add_argument("--dates", nargs="+",
                     help="Run date for each --input, in order, as YYYYMMDD "
                          "(default: inferred from each input's parent folder name)")
    ap.add_argument("--output", required=True, type=Path, help="Combined output CSV path")
    ap.add_argument("--dedupe-keep-latest", action="store_true",
                     help="If a (subject, session) appears in multiple inputs, keep only "
                          "the row from the latest run-date instead of keeping all rows")
    args = ap.parse_args()

    if args.dates and len(args.dates) != len(args.input):
        sys.exit("ERROR: --dates must have exactly one entry per --input, in the same order")

    for f in args.input:
        if not f.is_file():
            sys.exit(f"ERROR: input CSV not found: {f}")

    run_dates = [format_date(d) for d in args.dates] if args.dates else [infer_date(f) for f in args.input]

    # rows keyed by (subject, session) -> list of (run_date, etiv) in file order
    rows_by_key = {}
    ordered_keys = []

    for csv_path, run_date in zip(args.input, run_dates):
        with open(csv_path, newline="") as fin:
            reader = csv.DictReader(fin)
            if reader.fieldnames != ["subject", "session", "etiv"]:
                sys.exit(f"ERROR: {csv_path} does not have header 'subject,session,etiv' "
                         f"(got {reader.fieldnames})")
            for row in reader:
                key = (row["subject"], row["session"])
                if key in rows_by_key:
                    print(f"[WARN] {key[0]},{key[1]} appears in more than one input "
                          f"(already have {rows_by_key[key][0]}, also in {run_date} from {csv_path})",
                          file=sys.stderr)
                    if args.dedupe_keep_latest:
                        if run_date > rows_by_key[key][0]:
                            rows_by_key[key] = (run_date, row["etiv"])
                        continue
                    else:
                        # keep both: give the duplicate a distinguishable key
                        key = (row["subject"], row["session"], run_date, csv_path)
                        ordered_keys.append(key)
                        rows_by_key[key] = (run_date, row["etiv"])
                    continue
                ordered_keys.append(key)
                rows_by_key[key] = (run_date, row["etiv"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["subject", "session", "etiv", "run-date"])
        for key in ordered_keys:
            run_date, etiv = rows_by_key[key]
            writer.writerow([key[0], key[1], etiv, run_date])

    print(f"Wrote {len(ordered_keys)} row(s) -> {args.output}")


if __name__ == "__main__":
    main()
