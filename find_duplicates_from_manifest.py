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

Four checks per subject/session manifest:

  1. duplicate_source_file    - more than one src_path maps to the same
                                 dst_path (i.e. same SOPInstanceUID +
                                 InstanceNumber - the pipeline's own dedup
                                 key). Literal duplicate copies of the same
                                 DICOM instance in your raw sourcedata.
  2. duplicate_instance_number - the same (SeriesInstanceUID, InstanceNumber)
                                 maps to MORE THAN ONE SOPInstanceUID. This
                                 catches re-transmitted/re-exported source
                                 files that got a fresh SOPInstanceUID each
                                 time (e.g. a double PACS send) - since each
                                 copy then lands at a different dst_path,
                                 check 1 alone can't see it. In practice this
                                 is often the biggest source of inflated file
                                 counts.
  3. exact_byte_duplicate     - more than one DISTINCT dst_path shares the
                                 same md5 (needs the manifest to have been
                                 written without --skip-md5). Catches
                                 identical image content filed under a
                                 different SOPInstanceUID/series - e.g. a
                                 whole series re-run/re-exported.
  4. duplicate_series         - more than one SeriesInstanceUID shares the
                                 same (SeriesDescription, SeriesNumber) within
                                 a session. Loosest check - can include
                                 legitimate repeats (e.g. rescanned after
                                 motion) - review manually.

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

Optional dedupe move plan (--move-duplicates):
    For duplicate_instance_number and exact_byte_duplicate groups only (the
    two checks where every row in a group has a real file sitting in the
    staged tree), keeps the file with the lexicographically first src_path
    and relocates the rest out of --source-root into --dupe-root, mirroring
    their relative path. E.g. with
        --source-root /autofs/space/nicc_006/data/LETBI/raw/mri/MGH
        --dupe-root   /autofs/space/nicc_006/data/LETBI/raw/mri/dupe/MGH
    a duplicate at
        .../raw/mri/MGH/sub-MGHL2p001/ses-001/EMOTION_1/<file>.dcm
    moves to
        .../raw/mri/dupe/MGH/sub-MGHL2p001/ses-001/EMOTION_1/<file>.dcm

    Use --dry-run first to print the plan without moving anything:
        python find_duplicates_from_manifest.py /path/to/staged/output/root \\
            --move-duplicates --dry-run \\
            --source-root /autofs/.../raw/mri/MGH \\
            --dupe-root /autofs/.../raw/mri/dupe/MGH
    Drop --dry-run to actually perform the moves. A CSV log of the plan
    (default: dicom_dedupe_move_log.csv) is always written either way.
"""

import argparse
import csv
import fnmatch
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def find_manifests(root: Path):
    return sorted(root.rglob("_staging_manifest.csv"))


def load_manifest(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Each group_* function returns [(key, group_rows), ...] - the raw manifest
# rows behind each duplicate group, so callers can build either a human
# report (using src_path) or a move plan (using dst_path).

def group_duplicate_source_file(rows):
    """Same dst_path written by more than one src_path."""
    groups = defaultdict(list)
    for r in rows:
        dst = r.get("dst_path") or ""
        if dst:
            groups[dst].append(r)
    return [(dst, g) for dst, g in groups.items() if len(g) > 1]


def group_exact_byte_duplicate(rows):
    """Same md5 (of the staged/decoded output file) under different dst_path."""
    md5_to_rows = defaultdict(list)
    for r in rows:
        md5 = (r.get("md5") or "").strip()
        dst = r.get("dst_path") or ""
        if md5 and dst:
            md5_to_rows[md5].append(r)
    out = []
    for md5, group_rows in md5_to_rows.items():
        if len({r["dst_path"] for r in group_rows}) > 1:
            out.append((md5, group_rows))
    return out


def group_duplicate_instance_number(rows):
    """Same (SeriesInstanceUID, InstanceNumber) mapped to more than one
    SOPInstanceUID. Catches re-transmitted/re-exported source files that got
    a fresh SOPInstanceUID on each transmission (e.g. a double PACS send) -
    this defeats dcm2dir's own SOPInstanceUID-based dedup (each duplicate
    lands at a different dst_path since the filename is derived from
    SOPInstanceUID), so duplicate_source_file alone won't catch it."""
    groups = defaultdict(lambda: defaultdict(list))  # SeriesInstanceUID -> InstanceNumber -> rows
    for r in rows:
        series_uid = r.get("SeriesInstanceUID") or ""
        instnum = r.get("InstanceNumber") or ""
        if series_uid and instnum:
            groups[series_uid][instnum].append(r)
    out = []
    for series_uid, inst_map in groups.items():
        for instnum, group_rows in inst_map.items():
            if len({r.get("SOPInstanceUID", "") for r in group_rows}) > 1:
                key = f"SeriesInstanceUID={series_uid} | InstanceNumber={instnum}"
                out.append((key, group_rows))
    return out


def group_duplicate_series(rows):
    """Same (SeriesDescription, SeriesNumber) under different SeriesInstanceUID."""
    sig_to_rows = defaultdict(list)
    for r in rows:
        uid = r.get("SeriesInstanceUID") or ""
        if uid:
            sig = (r.get("SeriesDescription") or "", r.get("SeriesNumber") or "")
            sig_to_rows[sig].append(r)
    out = []
    for sig, group_rows in sig_to_rows.items():
        if len({r["SeriesInstanceUID"] for r in group_rows}) > 1:
            key = f"SeriesDescription={sig[0]} | SeriesNumber={sig[1]}"
            out.append((key, group_rows))
    return out


CHECKS = [
    ("duplicate_source_file", group_duplicate_source_file),
    ("duplicate_instance_number", group_duplicate_instance_number),
    ("exact_byte_duplicate", group_exact_byte_duplicate),
    ("duplicate_series", group_duplicate_series),
]

# Checks where every row in a group has its own real file sitting in the
# staged tree (dst_path), so "keep one, move the rest" is meaningful.
# duplicate_source_file only ever has ONE physical dst file per group (the
# rest were never written - the pipeline's "exists" idempotency check
# already caught them), and duplicate_series is too loose to auto-move
# (can include legitimate re-scans).
MOVE_ELIGIBLE_CHECKS = {"duplicate_instance_number", "exact_byte_duplicate"}


def rows_to_result(check_type, key, group_rows):
    files = sorted({r.get("src_path", "") for r in group_rows if r.get("src_path")})
    return {"type": check_type, "key": key, "files": files}


def build_move_plan(rows, source_root: Path, dupe_root: Path):
    """For each MOVE_ELIGIBLE_CHECKS group, keep the row with the
    lexicographically first src_path and plan to move every other row's
    dst_path into dupe_root, mirroring its position under source_root."""
    plan = []
    planned_srcs = set()
    for check_type, group_fn in ((t, fn) for t, fn in CHECKS if t in MOVE_ELIGIBLE_CHECKS):
        for key, group_rows in group_fn(rows):
            valid = [r for r in group_rows if r.get("dst_path")]
            if len(valid) < 2:
                continue
            valid.sort(key=lambda r: r.get("src_path", ""))
            keeper, movers = valid[0], valid[1:]
            for mover in movers:
                dst = mover["dst_path"]
                if dst in planned_srcs:
                    continue  # already planned by an earlier/overlapping check
                planned_srcs.add(dst)
                dst_path = Path(dst)
                try:
                    rel = dst_path.relative_to(source_root)
                except ValueError:
                    plan.append({
                        "check_type": check_type, "key": key, "keep": keeper["dst_path"],
                        "move_from": dst, "move_to": "",
                        "status": "error_not_under_source_root",
                    })
                    continue
                plan.append({
                    "check_type": check_type, "key": key, "keep": keeper["dst_path"],
                    "move_from": dst, "move_to": str(dupe_root / rel),
                    "status": "planned",
                })
    return plan


def run_move_plan(plan, dry_run: bool):
    """Mutates each entry's "status" in place to reflect what actually happened
    (or would happen, for a dry run)."""
    if not plan:
        print("\nNo duplicates matched --move-duplicates criteria - nothing to move.")
        return

    label = "DRY-RUN" if dry_run else "MOVE"
    print(f"\n{'=' * 60}\n{label} PLAN ({len(plan)} file(s)):")
    for p in plan:
        if p["status"] == "error_not_under_source_root":
            print(f"  [SKIP] {p['move_from']} is not under --source-root - can't remap")
            continue

        src = Path(p["move_from"])
        dst = Path(p["move_to"])
        print(f"  {src}\n    -> {dst}\n    (keeping {p['keep']})")

        if dry_run:
            continue

        if not src.exists():
            print("    [SKIP] source file no longer exists (already moved?)")
            p["status"] = "skipped_missing_source"
            continue
        if dst.exists():
            print("    [SKIP] destination already exists - not overwriting")
            p["status"] = "skipped_dest_exists"
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        p["status"] = "moved"


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
    parser.add_argument("--move-duplicates", action="store_true",
                         help="Also build a move plan: for duplicate_instance_number and "
                              "exact_byte_duplicate groups only, keep one file and relocate the "
                              "rest out of the staged tree into --dupe-root, mirroring their "
                              "path under --source-root. Requires --source-root and --dupe-root.")
    parser.add_argument("--source-root", type=str, default=None,
                         help="Root prefix to strip from each duplicate's dst_path before "
                              "remapping under --dupe-root, e.g. /autofs/.../raw/mri/MGH")
    parser.add_argument("--dupe-root", type=str, default=None,
                         help="Destination root duplicates get moved into, mirroring the "
                              "structure under --source-root, e.g. /autofs/.../raw/mri/dupe/MGH")
    parser.add_argument("--dry-run", action="store_true",
                         help="With --move-duplicates, print/log the move plan without moving "
                              "any files")
    parser.add_argument("--move-log", type=str, default="dicom_dedupe_move_log.csv",
                         help="CSV log of the move plan (default: dicom_dedupe_move_log.csv)")
    args = parser.parse_args()

    if args.move_duplicates and (not args.source_root or not args.dupe_root):
        sys.exit("ERROR: --move-duplicates requires both --source-root and --dupe-root")

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory")

    manifests = find_manifests(root)
    if not manifests:
        sys.exit(f"No _staging_manifest.csv files found under {root}")

    print(f"Found {len(manifests)} manifest file(s) under {root}\n")

    source_root = Path(args.source_root).resolve() if args.source_root else None
    dupe_root = Path(args.dupe_root).resolve() if args.dupe_root else None

    all_results = []
    all_move_plan = []
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
        for check_type, group_fn in CHECKS:
            for key, group_rows in group_fn(rows):
                results.append(rows_to_result(check_type, key, group_rows))

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

        if args.move_duplicates:
            plan = build_move_plan(rows, source_root, dupe_root)
            for p in plan:
                p["subject"] = subject_id
                p["session"] = session_id
            all_move_plan.extend(plan)

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

    if args.move_duplicates:
        run_move_plan(all_move_plan, dry_run=args.dry_run)

        log_path = Path(args.move_log)
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "session", "check_type", "signature_key",
                              "keep", "move_from", "move_to", "status"])
            for p in all_move_plan:
                writer.writerow([p["subject"], p["session"], p["check_type"], p["key"],
                                  p["keep"], p["move_from"], p["move_to"], p["status"]])
        print(f"\nMove log written to: {log_path.resolve()}")

        n_planned = sum(1 for p in all_move_plan if p["status"] in ("planned", "moved"))
        n_errors = sum(1 for p in all_move_plan if p["status"].startswith("error"))
        verb = "Would move" if args.dry_run else "Moved"
        print(f"{verb} {n_planned} duplicate file(s)"
              + (f", {n_errors} error(s) (see log)" if n_errors else "")
              + ".")


if __name__ == "__main__":
    main()
