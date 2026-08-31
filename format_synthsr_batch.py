#!/usr/bin/env python3
"""
Build mri_synthsr command queues for a BIDS dataset.

Walks sub-*/ses-*/anat/ for combined (non-echo) T1w files matching:
    sub-<subject>_ses-<session>_T1w.nii.gz
    sub-<subject>_ses-<session>_run-<n>_T1w.nii.gz

For each subject/session, the run-free T1w is preferred if present;
otherwise the lowest-numbered run-<n> T1w is used. At most one T1w is
queued per subject/session.

For each session found, writes a queue file containing one
mri_synthsr command per subject:

    mri_synthsr --i <T1w> --o <synthsr-root>/ses-<session>/sub-<subject>/sub-<subject>_ses-<session>_T1w_synthsr.nii.gz

Queue files are named "<session>_synthsr_queue.txt" and written to
--output-dir. Subjects whose output file already exists under
<synthsr-root>/ses-<session>/sub-<subject>/ are skipped (not written to the queue).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

T1W_RE = re.compile(r"^sub-[^_]+_ses-[^_]+(?:_run-(?P<run>\d+))?_T1w\.nii\.gz$")


def normalize_session(session: str) -> str:
    """Turn a session flag value into a BIDS session label (e.g. "1" -> "ses-001")."""
    session = session[len("ses-"):] if session.startswith("ses-") else session
    if session.isdigit():
        session = session.zfill(3)
    return f"ses-{session}"


def find_t1w_images(bids_root: Path, session: str = None):
    """Find combined (non-echo) T1w images, grouped by session.

    For each subject/session, the run-free T1w is preferred if present;
    otherwise the lowest-numbered run-<n> T1w is used.

    Returns a dict mapping session label ("ses-Y") -> list of (subject, path)
    tuples, sorted by subject.
    """
    session_glob = normalize_session(session) if session else "ses-*"

    images = defaultdict(list)
    for anat_dir in sorted(bids_root.glob(f"sub-*/{session_glob}/anat")):
        subject = anat_dir.parent.parent.name[len("sub-"):]
        ses_label = anat_dir.parent.name

        candidates = {}  # run number (or None for run-free) -> path
        for f in anat_dir.iterdir():
            m = T1W_RE.match(f.name)
            if m:
                run = int(m.group("run")) if m.group("run") else None
                candidates[run] = f
        if not candidates:
            continue

        if None in candidates:
            chosen = candidates[None]
        else:
            chosen_run = min(candidates)
            chosen = candidates[chosen_run]
            print(f"[info] sub-{subject} {ses_label}: no run-free T1w found, using run-{chosen_run:02d}")

        images[ses_label].append((subject, chosen))

    for ses_label in images:
        images[ses_label].sort(key=lambda x: x[0])

    return images


def build_command(subject, ses_label, t1w_path: Path, output_path: Path) -> str:
    return " ".join([
        "mri_synthsr",
        "--i", str(t1w_path),
        "--o", str(output_path),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids_root", type=Path, help="Path to the BIDS dataset root")
    parser.add_argument("--synthsr-root", type=Path, required=True,
                         help="Base SynthSR output directory, e.g. /.../derivatives/synthsr "
                              "(ses-<session>/sub-<subject>/ subdirectories are appended automatically)")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to write <session>_synthsr_queue.txt files to")
    parser.add_argument("--session", help="Restrict to a single session, e.g. --session 1 or --session ses-001")
    args = parser.parse_args()

    bids_root = args.bids_root.resolve()
    if not bids_root.is_dir():
        parser.error(f"{bids_root} is not a directory")

    synthsr_root = args.synthsr_root.resolve()
    output_dir = args.output_dir.resolve()

    images_by_session = find_t1w_images(bids_root, args.session)
    if not images_by_session:
        scope = f" for {normalize_session(args.session)}" if args.session else ""
        print(f"No T1w images found under {bids_root}{scope}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for ses_label, images in sorted(images_by_session.items()):
        queue_path = output_dir / f"{ses_label}_synthsr_queue.txt"
        lines = []

        for subject, t1w_path in images:
            subj_dir = synthsr_root / ses_label / f"sub-{subject}"
            output_path = subj_dir / f"sub-{subject}_{ses_label}_T1w_synthsr.nii.gz"
            if output_path.exists():
                print(f"[skip] {output_path} already exists")
                continue
            lines.append(build_command(subject, ses_label, t1w_path, output_path))

        if not lines:
            print(f"[queue] {ses_label}: nothing to queue")
            continue

        queue_path.write_text("\n".join(lines) + "\n")
        print(f"[queue] {queue_path} ({len(lines)} command(s))")


if __name__ == "__main__":
    main()
