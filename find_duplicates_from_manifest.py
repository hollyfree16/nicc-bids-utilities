#!/usr/bin/env python3
"""
find_duplicates_from_manifest.py

Find potential duplicate DICOM instances/series using the _staging_manifest.csv
files nicc-bids-prep's dicom/dcm2dir already produces - instead of re-scanning
and re-parsing raw DICOM files.

Why this works: dcm2dir stages every raw source DICOM file it finds into a
deterministic path

    <SeriesDescription>_<SeriesNumber>/<InstanceNumber:06d>_<short-SOPUID>.dcm

and writes one manifest row per SOURCE file, including an "exists" row
(without re-copying) whenever a later source file maps to a destination path
that was already written by an earlier one. That "exists" bookkeeping is
already, for free, an exact record of duplicate source DICOM instances - no
DICOM parsing or extra disk I/O required, just reading the CSV.

CAVEAT: a manifest is a snapshot from whenever dcm2dir was last run for that
subject/session. If raw DICOM files were added/changed afterward, re-run
dcm2dir (or at least its scan step) before trusting this report.

Three checks per subject/session manifest:

  1. duplicate_source_file - more than one src_path maps to the same dst_path
                              (i.e. same SOPInstanceUID + InstanceNumber - the
                              pipeline's own dedup key). Strongest signal:
                              these are literal duplicate copies of the same
                              DICOM instance in your raw sourcedata.
  2. exact_byte_duplicate  - more than one DISTINCT dst_path shares the same
                              md5 (needs the manifest to have been written
                              without --skip-md5). Catches identical image
                              content filed under a different SOPInstanceUID/
                              series - e.g. a whole series re-run/re-exported.
  3. duplicate_series      - more than one SeriesInstanceUID shares the same
                              (SeriesDescription, SeriesNumber) within a
                              session. Loosest check - can include legitimate
                              repeats (e.g. rescanned after motion) - review
                              manually.

Usage:
    python find_duplicates_from_manifest.py /path/to/staged/output/root

    # only certain subjects/sessions (matched against the SubjectID/SessionID
    # values recorded IN each manifest, not directory names)
    python find_duplicates_from_manifest.py /path/to/staged/output/root \\
        --subject "sub-MGHL2p*" --session "ses-001"

Output:
    - Console: "Scanning <manifest path> ..." then a per-subject/session
      summary of duplicate groups found, printed as each manifest is read
    - CSV report (default: dicom_duplicate_report.csv)
"""

import argparse
import csv
import fnmatch
import sys
from collections import defaultdict
from pathlib import Path


def find_manifests(root: Path):
    return sorted(root.rglob("_staging_manifest.csv"))


def load_manifest(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_duplicate_source_file(rows):
    """Same dst_path written by more than one src_path."""
    groups = defaultdict(list)
    for r in rows:
        dst = r.get("dst_path") or ""
        if dst:
            groups[dst].append(r)
    results = []
    for dst_path, group_rows in groups.items():
        if len(group_rows) > 1:
            results.append({
                "type": "duplicate_source_file",
                "key": dst_path,
                "files": [r.get("src_path", "") for r in group_rows],
            })
    return results


def check_exact_byte_duplicate(rows):
    """Same md5 (of the staged/decoded output file) under different dst_path."""
    md5_to_dst = defaultdict(set)
    md5_to_rows = defaultdict(list)
    for r in rows:
        md5 = (r.get("md5") or "").strip()
        dst = r.get("dst_path") or ""
        if md5 and dst:
            md5_to_dst[md5].add(dst)
            md5_to_rows[md5].append(r)
    results = []
    for md5, dst_paths in md5_to_dst.items():
        if len(dst_paths) > 1:
            files = sorted({r.get("src_path", "") for r in md5_to_rows[md5]})
            results.append({"type": "exact_byte_duplicate", "key": md5, "files": files})
    return results


def check_duplicate_series(rows):
    """Same (SeriesDescription, SeriesNumber) under different SeriesInstanceUID."""
    sig_to_uids = defaultdict(set)
    uid_to_files = defaultdict(set)
    for r in rows:
        uid = r.get("SeriesInstanceUID") or ""
        if not uid:
            continue
        sig = (r.get("SeriesDescription") or "", r.get("SeriesNumber") or "")
        sig_to_uids[sig].add(uid)
        src = r.get("src_path") or ""
        if src:
            uid_to_files[uid].add(src)
    results = []
    for sig, uids in sig_to_uids.items():
        if len(uids) > 1:
            files = sorted({f for uid in uids for f in uid_to_files[uid]})
            results.append({
                "type": "duplicate_series",
                "key": f"SeriesDescription={sig[0]} | SeriesNumber={sig[1]}",
                "files": files,
            })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find potential duplicate DICOMs using existing nicc-bids-prep "
                     "_staging_manifest.csv files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("root_dir", type=str,
                         help="Directory to recursively search for _staging_manifest.csv files "
                              "(e.g. the dcm2dir --output-dir root)")
    parser.add_argument("--subject", type=str, default=None,
                         help="Only include manifests whose SubjectID column matches this glob, "
                              "e.g. 'sub-MGHL2p*'")
    parser.add_argument("--session", type=str, default=None,
                         help="Only include manifests whose SessionID column matches this glob, "
                              "e.g. 'ses-001'")
    parser.add_argument("--out", type=str, default="dicom_duplicate_report.csv",
                         help="Output CSV path (default: dicom_duplicate_report.csv)")
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory")

    manifests = find_manifests(root)
    if not manifests:
        sys.exit(f"No _staging_manifest.csv files found under {root}")

    print(f"Found {len(manifests)} manifest file(s) under {root}\n")

    all_results = []
    n_included = 0
    for manifest_path in manifests:
        rows = load_manifest(manifest_path)
        if not rows:
            continue

        subject_id = rows[0].get("SubjectID", "") or ""
        session_id = rows[0].get("SessionID", "") or ""

        if args.subject and not fnmatch.fnmatch(subject_id, args.subject):
            continue
        if args.session and not fnmatch.fnmatch(session_id, args.session):
            continue
        n_included += 1

        print(f"Scanning {manifest_path} (SubjectID={subject_id}, SessionID={session_id or 'N/A'}) ...")
        print(f"  {len(rows)} manifest rows")

        error_rows = [r for r in rows if r.get("action") == "error"]
        if error_rows:
            print(f"  [WARN] {len(error_rows)} row(s) recorded action=error during staging "
                  f"(source file couldn't be read/placed) - excluded from duplicate checks, "
                  f"worth a manual look")

        results = []
        results.extend(check_duplicate_source_file(rows))
        results.extend(check_exact_byte_duplicate(rows))
        results.extend(check_duplicate_series(rows))

        if not results:
            print("  No potential duplicates found.\n")
        else:
            print(f"  Found {len(results)} potential duplicate group(s):")
            for r in results:
                print(f"    [{r['type']}] {r['key']}")
                for f in r["files"]:
                    print(f"        - {f}")
            print()

        for r in results:
            r["subject"] = subject_id
            r["session"] = session_id
            r["manifest"] = str(manifest_path)
        all_results.extend(results)

    if n_included == 0:
        sys.exit("No manifests matched the given --subject/--session filters.")

    print(f"{'=' * 60}\nTOTAL: {len(all_results)} potential duplicate group(s) "
          f"across {n_included} subject/session manifest(s).")

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "session", "duplicate_type", "signature_key", "files", "manifest"])
        for r in all_results:
            writer.writerow([r["subject"], r["session"], r["type"], r["key"],
                              " ;; ".join(r["files"]), r["manifest"]])
    print(f"Report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
