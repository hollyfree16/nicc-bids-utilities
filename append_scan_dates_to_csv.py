#!/usr/bin/env python3
"""
add_scan_dates.py

Reads an input CSV with rows like:
    UWp99568,ses-001,1526858
(subject-id, session-id, etiv)

For each row:
    - derives the site as the leading letters of subject-id (e.g. "UW" from "UWp99568")
    - builds the session directory: <root>/<site>/sub-<subject-id>/<session-id>
    - finds the DICOM scan date the same way generate_scan_dates_tsv.py does
      (first usable StudyDate/SeriesDate/AcquisitionDate/ContentDate tag,
      checked in filename order, lazily, without listing entire session trees)

Writes an output CSV with columns: subject-id, session-id, etiv, scan-date
    - scan-date is YYYY-MM-DD if found
    - NO_DATE if the session dir exists but no date could be read
    - MISSING_DIR if the session directory doesn't exist on disk at all

Usage
-----
    python add_scan_dates.py \\
        --input etivs.csv \\
        --root /autofs/space/nicc_006/data/LETBI/raw/mri \\
        --output etivs_with_dates.csv
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    sys.exit("ERROR: pydicom is required. Install with: pip install pydicom --break-system-packages")


# Leading UPPERCASE letters of subject-id = site, e.g. "UW" from "UWp99568".
# Note: matching all leading letters (e.g. [A-Za-z]+) is wrong here -- the
# "p" in "UWp99568" is a letter too and would get swallowed into the site
# code, producing a bogus "UWp" site and MISSING_DIR for every row.
SITE_RE = re.compile(r'^([A-Z]+)')

# Checked in this order; first usable one wins.
DATE_TAGS = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]


def format_date(raw) -> str | None:
    """DICOM dates are YYYYMMDD. Return YYYY-MM-DD, or None if not a usable date."""
    if raw is None:
        return None
    raw = str(raw).strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def iter_candidate_files(session_dir: Path):
    """Lazily yield files under session_dir without ever building a full listing.

    Layout is one level deep: session_dir/<series>/*.dcm. Top-level files
    (rare) are yielded first, then series subfolders one at a time.
    """
    try:
        with os.scandir(session_dir) as it:
            entries = list(it)  # session dir itself has few entries -- cheap
    except OSError:
        return

    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    for f in files:
        yield Path(f.path)

    for d in dirs:
        try:
            with os.scandir(d.path) as it2:
                for f2 in it2:
                    if f2.is_file():
                        yield Path(f2.path)
        except OSError:
            continue


def find_scan_date(session_dir: Path, max_files: int) -> str | None:
    """Try up to max_files candidate files (lazily discovered) until one
    yields a usable date tag. Returns None if none found."""
    checked = 0
    for f in iter_candidate_files(session_dir):
        if checked >= max_files:
            break
        if f.suffix.lower() not in ("", ".dcm"):
            continue  # skip obviously-non-DICOM files
        checked += 1
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except (InvalidDicomError, Exception):
            continue
        for tag in DATE_TAGS:
            val = getattr(ds, tag, None)
            formatted = format_date(val)
            if formatted:
                return formatted
    return None


def derive_site(subject_id: str) -> str | None:
    m = SITE_RE.match(subject_id)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                     help="Input CSV with columns: subject-id, session-id, etiv (no header assumed)")
    ap.add_argument("--root", required=True, type=Path,
                     help="raw/mri root containing site subfolders (IU, UW, MSSM, MGH, ...)")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--max-files-per-session", type=int, default=5,
                     help="Stop checking a session's files after this many attempts (default: 5)")
    ap.add_argument("--has-header", action="store_true",
                     help="Set this if the input CSV has a header row to skip")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"ERROR: root not found: {args.root}")
    if not args.input.is_file():
        sys.exit(f"ERROR: input CSV not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, newline="") as fin, open(args.output, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(["subject-id", "session-id", "etiv", "scan-date"])
        fout.flush()

        rows = list(reader)
        if args.has_header and rows:
            rows = rows[1:]

        total = len(rows)
        for i, row in enumerate(rows, start=1):
            if not row or len(row) < 3:
                print(f"[WARN] skipping malformed row: {row}", file=sys.stderr)
                continue
            subject_id, session_id, etiv = row[0].strip(), row[1].strip(), row[2].strip()

            site = derive_site(subject_id)
            if not site:
                print(f"[WARN] could not derive site from subject-id '{subject_id}', skipping", file=sys.stderr)
                writer.writerow([subject_id, session_id, etiv, "NO_DATE"])
                fout.flush()
                continue

            session_dir = args.root / site / f"sub-{subject_id}" / session_id

            if not session_dir.is_dir():
                scan_date = "MISSING_DIR"
            else:
                date = find_scan_date(session_dir, args.max_files_per_session)
                scan_date = date if date else "NO_DATE"

            print(f"[SCAN] {subject_id} {session_id} -> {scan_date}", file=sys.stderr)
            writer.writerow([subject_id, session_id, etiv, scan_date])
            fout.flush()  # visible to `tail -f`, safe if interrupted

            if i % 25 == 0 or i == total:
                print(f"[PROGRESS] {i}/{total} rows written", file=sys.stderr)

    print(f"\nWrote {total} row(s) -> {args.output}")


if __name__ == "__main__":
    main()