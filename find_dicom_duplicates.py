#!/usr/bin/env python3
"""
find_dicom_duplicates.py

Scan a directory of raw DICOM files (e.g. a BIDS sourcedata/ folder) for
potential duplicate scans. Files are identified as DICOM by content, not by
extension, since raw exports are frequently missing or have inconsistent
extensions (.dcm, .ima, none, etc):

  - fast path:  128-byte preamble + "DICM" magic (DICOM Part 10 - the
                standard since 1993, and what virtually all scanner/PACS
                exports produce)
  - slow path (--allow-headerless): for older or bare "DICOM stream" files
                that were never given a Part 10 preamble (some legacy
                console/PACS exports), attempt a raw parse of every
                remaining candidate file and keep it if it decodes into a
                plausible dataset. Off by default because it has to open
                and attempt to parse every non-DICOM file too, which is
                much slower on a large tree.

Four independent checks are run per study, from strictest to loosest:

  1. exact_file_duplicate   - files are byte-for-byte identical (MD5 of raw file)
  2. duplicate_sop_instance - the same SOPInstanceUID appears in more than one
                               file. This UID should uniquely identify a single
                               DICOM image - a repeat means the same instance
                               was saved/copied more than once.
  3. identical_pixel_data   - pixel data + key geometry are identical even
                               though the file bytes/UIDs differ (e.g. a
                               re-exported or re-anonymized copy of the same
                               image). Requires pydicom.
  4. duplicate_series       - two series in the same study share
                               SeriesDescription/EchoTime/RepetitionTime/
                               FlipAngle/instance count but have different
                               SeriesInstanceUID - a whole series may have
                               been duplicated or re-run. This is the loosest
                               check and can include legitimate repeats (e.g.
                               a scan re-run after motion) - review manually.

Subjects are discovered as the immediate subdirectories of root_dir matching
--subject (default 'sub-*'); each is scanned and reported on independently,
so duplicate checks only compare files within the same subject.

Usage:
    python find_dicom_duplicates.py /path/to/sourcedata

    # only scan subject directories matching a glob
    python find_dicom_duplicates.py /path/to/sourcedata --subject "sub-MGHL2p*"

    # further restrict to files matching a glob against the full path, e.g. one session
    python find_dicom_duplicates.py /path/to/sourcedata --subject "sub-MGHL2p*" --pattern "*ses-01*"

    # skip the (slower) pixel-data content check
    python find_dicom_duplicates.py /path/to/sourcedata --no-pixel-hash

    # also catch older/legacy DICOM files with no Part 10 preamble (slower)
    python find_dicom_duplicates.py /path/to/sourcedata --allow-headerless

Output:
    - Console: "Scanning <subject path> ..." followed by a per-subject summary
      of any duplicate groups found, printed as each subject finishes
    - CSV report (default: dicom_duplicate_report.csv), one row per duplicate
      group across all subjects
"""

import argparse
import csv
import fnmatch
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pydicom
    HAVE_PYDICOM = True
except ImportError:
    HAVE_PYDICOM = False


KEY_SERIES_FIELDS = [
    "SeriesDescription",
    "ProtocolName",
    "SeriesNumber",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "AcquisitionNumber",
]

# extensions that can never be DICOM - skip without even opening these
SKIP_SUFFIXES = {
    ".json", ".txt", ".csv", ".xml", ".html", ".htm", ".md",
    ".yml", ".yaml", ".log", ".tsv", ".pdf", ".zip", ".gz", ".tar",
}
SKIP_NAMES = {"DICOMDIR", ".DS_Store"}


def find_candidate_files(root: Path, pattern: str = None):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in SKIP_NAMES or p.name.startswith("."):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        if pattern and not fnmatch.fnmatch(str(p), pattern):
            continue
        yield p


def has_dicom_magic(path: Path) -> bool:
    """Check for the Part-10 preamble + 'DICM' magic, regardless of extension."""
    try:
        with open(path, "rb") as f:
            header = f.read(132)
        return len(header) == 132 and header[128:132] == b"DICM"
    except OSError:
        return False


def read_dicom_header(path: Path):
    """force=True lets this also parse legacy files with no Part 10 preamble;
    it's a no-op for well-formed files that do have one."""
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def looks_like_dicom(ds) -> bool:
    # force=True will happily "parse" arbitrary non-DICOM bytes into a
    # near-empty Dataset - require real identifying tags before trusting it
    if ds is None or not hasattr(ds, "SOPInstanceUID"):
        return False
    return hasattr(ds, "Modality") or hasattr(ds, "SOPClassUID") or hasattr(ds, "PixelData")


def file_md5(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def pixel_data_hash(path: Path):
    """Hash of pixel data + key geometry - catches identical images saved
    under different filenames/UIDs (e.g. re-exported or re-anonymized)."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        if not hasattr(ds, "PixelData"):
            return None
        h = hashlib.md5()
        h.update(ds.PixelData)
        h.update(str(getattr(ds, "ImagePositionPatient", "")).encode())
        h.update(str(getattr(ds, "PixelSpacing", "")).encode())
        h.update(str(getattr(ds, "Rows", "")).encode())
        h.update(str(getattr(ds, "Columns", "")).encode())
        return h.hexdigest()
    except Exception:
        return None


def series_signature(ds):
    parts = [f"StudyInstanceUID={getattr(ds, 'StudyInstanceUID', None)}"]
    for field in KEY_SERIES_FIELDS:
        parts.append(f"{field}={getattr(ds, field, None)}")
    return tuple(parts)


def already_covered(files, results, types):
    fset = set(str(x) for x in files)
    return any(fset == set(r["files"]) for r in results if r["type"] in types)


def find_subject_dirs(root: Path, subject_pattern: str):
    return sorted(p for p in root.glob(subject_pattern) if p.is_dir())


def scan_for_dicom(scan_dir: Path, pattern: str, allow_headerless: bool):
    """Walk scan_dir and return (dicom_files, n_checked, n_magic_hits, n_headerless_hits)."""
    dicom_files = []
    n_checked = 0
    n_magic_hits = 0
    n_headerless_hits = 0
    for path in find_candidate_files(scan_dir, pattern):
        n_checked += 1
        if has_dicom_magic(path):
            n_magic_hits += 1
            ds = read_dicom_header(path)
            if looks_like_dicom(ds):
                dicom_files.append((path, ds))
        elif allow_headerless:
            ds = read_dicom_header(path)
            if looks_like_dicom(ds):
                n_headerless_hits += 1
                dicom_files.append((path, ds))
        if n_checked % 5000 == 0:
            print(f"    ...checked {n_checked} files, {len(dicom_files)} DICOM so far", file=sys.stderr)
    return dicom_files, n_checked, n_magic_hits, n_headerless_hits


def run_duplicate_checks(dicom_files, no_pixel_hash: bool):
    """dicom_files is a list of (path, pydicom Dataset). Returns a list of
    {"type", "key", "files"} duplicate-group dicts."""
    results = []

    # 1. exact file duplicate
    hash_groups = defaultdict(list)
    for path, ds in dicom_files:
        try:
            hash_groups[file_md5(path)].append(path)
        except OSError as e:
            print(f"    [WARN] could not hash {path}: {e}", file=sys.stderr)
    for h, files in hash_groups.items():
        if len(files) > 1:
            results.append({"type": "exact_file_duplicate", "key": h, "files": [str(x) for x in files]})

    # 2. duplicate SOPInstanceUID
    sop_groups = defaultdict(list)
    for path, ds in dicom_files:
        sop_groups[str(ds.SOPInstanceUID)].append(path)
    for uid, files in sop_groups.items():
        if len(files) > 1 and not already_covered(files, results, {"exact_file_duplicate"}):
            results.append({"type": "duplicate_sop_instance", "key": uid, "files": [str(x) for x in files]})

    # 3. identical pixel data
    if not no_pixel_hash:
        pixel_groups = defaultdict(list)
        for path, ds in dicom_files:
            ph = pixel_data_hash(path)
            if ph:
                pixel_groups[ph].append(path)
        for h, files in pixel_groups.items():
            if len(files) > 1 and not already_covered(
                files, results, {"exact_file_duplicate", "duplicate_sop_instance"}
            ):
                results.append({"type": "identical_pixel_data", "key": h, "files": [str(x) for x in files]})

    # 4. duplicate series (same study + acquisition params, different SeriesInstanceUID)
    series_index = defaultdict(set)   # signature -> set of SeriesInstanceUID
    series_files = defaultdict(list)  # SeriesInstanceUID -> files
    for path, ds in dicom_files:
        series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        series_files[series_uid].append(path)
        series_index[series_signature(ds)].add(series_uid)

    for sig, series_uids in series_index.items():
        if len(series_uids) > 1:
            files = sorted(set(f for uid in series_uids for f in series_files[uid]))
            results.append({"type": "duplicate_series", "key": " | ".join(sig), "files": [str(x) for x in files]})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find potential duplicate DICOM files/series in a raw sourcedata directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("root_dir", type=str, help="Path to scan (e.g. sourcedata/)")
    parser.add_argument("--subject", type=str, default=None,
                         help="Subject glob matched against immediate subdirectories of "
                              "root_dir, e.g. 'sub-MGHL2p*' (default: 'sub-*', falling back to "
                              "treating root_dir itself as a single subject if nothing matches)")
    parser.add_argument("--pattern", type=str, default=None,
                         help="Additionally restrict to files matching this glob against the "
                              "full path within each subject, e.g. '*ses-01*'")
    parser.add_argument("--no-pixel-hash", action="store_true",
                         help="Skip pixel-data hashing (faster, less thorough)")
    parser.add_argument("--allow-headerless", action="store_true",
                         help="Also attempt to parse files with no Part 10 preamble/'DICM' "
                              "magic (older/legacy exports). Slower: every non-matching "
                              "candidate file gets a parse attempt.")
    parser.add_argument("--out", type=str, default="dicom_duplicate_report.csv",
                         help="Output CSV path (default: dicom_duplicate_report.csv)")
    args = parser.parse_args()

    if not HAVE_PYDICOM:
        sys.exit("ERROR: pydicom is required for this script. Install with: pip install pydicom")

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory")

    subject_pattern = args.subject or "sub-*"
    subj_dirs = find_subject_dirs(root, subject_pattern)
    if not subj_dirs:
        if args.subject is None:
            subj_dirs = [root]  # no sub-* layout - treat root itself as one subject
        else:
            sys.exit(f"No subject directories matched '{args.subject}' under {root}")

    print(f"Found {len(subj_dirs)} subject director{'y' if len(subj_dirs) == 1 else 'ies'} "
          f"matching '{subject_pattern}'\n")

    all_results = []
    any_dicom_found = False
    for subj_dir in subj_dirs:
        print(f"Scanning {subj_dir} ...")
        dicom_files, n_checked, n_magic, n_headerless = scan_for_dicom(
            subj_dir, args.pattern, args.allow_headerless
        )
        print(f"  checked {n_checked} candidate files ({n_magic} DICOM magic-byte hits"
              + (f", {n_headerless} headerless" if args.allow_headerless else "")
              + f"), {len(dicom_files)} parsed as valid DICOM instances")

        if not dicom_files:
            print("  No DICOM files found for this subject.\n")
            continue
        any_dicom_found = True

        results = run_duplicate_checks(dicom_files, args.no_pixel_hash)
        if not results:
            print("  No potential duplicates found.\n")
        else:
            print(f"  Found {len(results)} potential duplicate group(s):")
            for r in results:
                print(f"    [{r['type']}]")
                for f in r["files"]:
                    print(f"        - {f}")
            print()

        for r in results:
            r["subject"] = subj_dir.name
        all_results.extend(results)

    if not any_dicom_found:
        if args.allow_headerless:
            sys.exit("No DICOM files found under the given path/subject/pattern.")
        sys.exit(
            "No DICOM files found via the Part 10 magic-byte check in any subject. If some of "
            "your files predate that standard or had the preamble stripped, re-run with "
            "--allow-headerless."
        )

    print(f"{'=' * 60}\nTOTAL: {len(all_results)} potential duplicate group(s) "
          f"across {len(subj_dirs)} subject(s).")

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "duplicate_type", "signature_key", "files"])
        for r in all_results:
            writer.writerow([r["subject"], r["type"], r["key"], " ;; ".join(r["files"])])
    print(f"Report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
