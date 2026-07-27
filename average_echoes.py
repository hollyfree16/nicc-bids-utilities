#!/usr/bin/env python3
"""
Average multi-echo T1w images across a BIDS dataset using FreeSurfer's mri_average.

Walks sub-*/ses-*/anat/ for files matching:
    sub-<subject>_ses-<session>_echo-<n>_T1w.nii.gz

and, for each subject/session found, runs:
    mri_average -noconform -rms <echo files...> sub-<subject>_ses-<session>_T1w.nii.gz

The number of echoes is auto-detected per session (not assumed to be 4), and
sessions whose combined output already exists are skipped by default.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ECHO_RE = re.compile(
    r"^(?P<prefix>sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+))_echo-(?P<echo>\d+)_T1w\.nii\.gz$"
)


def normalize_session(session: str) -> str:
    """Turn a session flag value into a BIDS session label (e.g. "1" -> "ses-001").

    Accepts "1", "001", or "ses-001" and returns the "ses-XXX" form. Numeric
    values are zero-padded to 3 digits; non-numeric values are passed through
    as-is (with a "ses-" prefix added if missing).
    """
    session = session[len("ses-"):] if session.startswith("ses-") else session
    if session.isdigit():
        session = session.zfill(3)
    return f"ses-{session}"


def find_echo_groups(bids_root: Path, session: str = None):
    """Group multi-echo T1w files by (subject, session).

    Returns a dict mapping prefix ("sub-X_ses-Y") -> list of (echo_num, path),
    sorted by echo number.
    """
    session_glob = normalize_session(session) if session else "ses-*"

    groups = defaultdict(list)
    for anat_dir in sorted(bids_root.glob(f"sub-*/{session_glob}/anat")):
        for f in anat_dir.iterdir():
            m = ECHO_RE.match(f.name)
            if m:
                groups[m.group("prefix")].append((int(m.group("echo")), f))

    for prefix in groups:
        groups[prefix].sort(key=lambda x: x[0])

    return groups


def run_mri_average(prefix: str, echo_files, anat_dir: Path, overwrite: bool, dry_run: bool, log_dir: Path = None):
    out_path = anat_dir / f"{prefix}_T1w.nii.gz"

    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} already exists")
        return

    cmd = ["mri_average", "-noconform", "-rms"] + [str(p) for p in echo_files] + [str(out_path)]

    log_path = log_dir / f"{prefix}_mri-average.log" if log_dir else None

    print(f"[run]  {' '.join(cmd)}" + (f"  (log: {log_path})" if log_path else ""))
    if dry_run:
        return

    if log_path:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[error] mri_average failed for {prefix} (exit {result.returncode})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids_root", type=Path, help="Path to the BIDS dataset root")
    parser.add_argument("--session", help="Restrict to a single session, e.g. --session 1 or --session ses-001")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory to save mri_average logs under (written to <log-dir>/mri_average/<subject>_<session>_mri-average.log)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run and overwrite existing averaged outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands that would run without executing them")
    args = parser.parse_args()

    bids_root = args.bids_root.resolve()
    if not bids_root.is_dir():
        parser.error(f"{bids_root} is not a directory")

    groups = find_echo_groups(bids_root, args.session)
    if not groups:
        scope = f" for {normalize_session(args.session)}" if args.session else ""
        print(f"No multi-echo T1w files found under {bids_root}{scope}")
        return

    log_dir = (args.log_dir.resolve() / "mri_average") if args.log_dir else None

    for prefix, echo_files in sorted(groups.items()):
        anat_dir = echo_files[0][1].parent
        paths = [p for _, p in echo_files]
        run_mri_average(prefix, paths, anat_dir, args.overwrite, args.dry_run, log_dir)


if __name__ == "__main__":
    main()
