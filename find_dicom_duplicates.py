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

Usage:
    python find_dicom_duplicates.py /path/to/sourcedata

    # only scan paths matching a glob (matched against the full path)
    python find_dicom_duplicates.py /path/to/sourcedata --pattern "*sub-MGHL2p*"

    # skip the (slower) pixel-data content check
    python find_dicom_duplicates.py /path/to/sourcedata --no-pixel-hash

    # also catch older/legacy DICOM files with no Part 10 preamble (slower)
    python find_dicom_duplicates.py /path/to/sourcedata --allow-headerless

Output:
    - Console summary of all duplicate groups found
    - CSV report (default: dicom_duplicate_report.csv)
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


def main():
    parser = argparse.ArgumentParser(
        description="Find potential duplicate DICOM files/series in a raw sourcedata directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("root_dir", type=str, help="Path to scan (e.g. sourcedata/)")
    parser.add_argument("--pattern", type=str, default=None,
                         help="Only consider paths matching this glob against the full path, "
                              "e.g. '*sub-MGHL2p*'")
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

    print(f"Scanning {root} for DICOM files (identified by content, not extension)...")

    dicom_files = []  # list of (path, dataset)
    n_checked = 0
    n_magic_hits = 0
    n_headerless_hits = 0
    for path in find_candidate_files(root, args.pattern):
        n_checked += 1
        if has_dicom_magic(path):
            n_magic_hits += 1
            ds = read_dicom_header(path)
            if looks_like_dicom(ds):
                dicom_files.append((path, ds))
        elif args.allow_headerless:
            ds = read_dicom_header(path)
            if looks_like_dicom(ds):
                n_headerless_hits += 1
                dicom_files.append((path, ds))
        if n_checked % 5000 == 0:
            print(f"  ...checked {n_checked} files, {len(dicom_files)} DICOM so far", file=sys.stderr)

    print(f"Checked {n_checked} candidate files ({n_magic_hits} had DICOM magic bytes, "
          f"{n_headerless_hits} recovered as headerless legacy DICOM), "
          f"{len(dicom_files)} parsed as valid DICOM instances.")
    if not dicom_files:
        if args.allow_headerless:
            sys.exit("No DICOM files found under the given path/pattern.")
        sys.exit(
            "No DICOM files found via the Part 10 magic-byte check. If some of your files "
            "predate that standard or had the preamble stripped, re-run with --allow-headerless."
        )

    results = []

    # 1. exact file duplicate
    hash_groups = defaultdict(list)
    for path, ds in dicom_files:
        try:
            hash_groups[file_md5(path)].append(path)
        except OSError as e:
            print(f"  [WARN] could not hash {path}: {e}", file=sys.stderr)
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
    if not args.no_pixel_hash:
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

    if not results:
        print("\nNo potential duplicates found.")
    else:
        print(f"\nFound {len(results)} potential duplicate group(s):\n")
        for r in results:
            print(f"[{r['type']}]")
            for f in r["files"]:
                print(f"    - {f}")
            print()

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["duplicate_type", "signature_key", "files"])
        for r in results:
            writer.writerow([r["type"], r["key"], " ;; ".join(r["files"])])
    print(f"Report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
