#!/usr/bin/env python3
"""
Build run_recon-any command queues for a BIDS dataset.

Walks sub-*/ses-*/anat/ for combined (non-echo) T1w files matching:
    sub-<subject>_ses-<session>_T1w.nii.gz

and, for each session found, writes a queue file containing one
run_recon-any command per subject:

    run_recon-any -i <T1w> -subjid sub-<subject> -threads <threads> -side <side> \
        -sdir <sdir-root>/ses-<session>

Queue files are named "<session>_recon-any_queue.txt" and written to
--output-dir. Subjects whose directory already exists under
<sdir-root>/ses-<session>/ are skipped (not written to the queue).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

T1W_RE = re.compile(r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_T1w\.nii\.gz$")


def normalize_session(session: str) -> str:
    """Turn a session flag value into a BIDS session label (e.g. "1" -> "ses-001")."""
    session = session[len("ses-"):] if session.startswith("ses-") else session
    if session.isdigit():
        session = session.zfill(3)
    return f"ses-{session}"


def find_t1w_images(bids_root: Path, session: str = None):
    """Find combined (non-echo) T1w images, grouped by session.

    Returns a dict mapping session label ("ses-Y") -> list of (subject, path)
    tuples, sorted by subject.
    """
    session_glob = normalize_session(session) if session else "ses-*"

    images = defaultdict(list)
    for anat_dir in sorted(bids_root.glob(f"sub-*/{session_glob}/anat")):
        for f in anat_dir.iterdir():
            m = T1W_RE.match(f.name)
            if m:
                images[f"ses-{m.group('session')}"].append((m.group("subject"), f))

    for ses_label in images:
        images[ses_label].sort(key=lambda x: x[0])

    return images


def build_command(subject, ses_label, t1w_path: Path, sdir_root: Path, threads: int, side: str) -> str:
    subjid = f"sub-{subject}"
    sdir = sdir_root / ses_label

    return " ".join([
        "run_recon-any",
        "-i", str(t1w_path),
        "-subjid", subjid,
        "-threads", str(threads),
        "-side", side,
        "-sdir", str(sdir),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids_root", type=Path, help="Path to the BIDS dataset root")
    parser.add_argument("--sdir-root", type=Path, required=True,
                         help="Base FreeSurfer subjects directory, e.g. /.../derivatives/freesurfer_reconany_v8.2.0 "
                              "(a ses-<session> subdirectory is appended automatically)")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to write <session>_recon-any_queue.txt files to")
    parser.add_argument("--session", help="Restrict to a single session, e.g. --session 1 or --session ses-001")
    parser.add_argument("--threads", type=int, default=1, help="Threads to pass to run_recon-any (default: 1)")
    parser.add_argument("--side", default="both", help="Side to pass to run_recon-any (default: both)")
    args = parser.parse_args()

    bids_root = args.bids_root.resolve()
    if not bids_root.is_dir():
        parser.error(f"{bids_root} is not a directory")

    sdir_root = args.sdir_root.resolve()
    output_dir = args.output_dir.resolve()

    images_by_session = find_t1w_images(bids_root, args.session)
    if not images_by_session:
        scope = f" for {normalize_session(args.session)}" if args.session else ""
        print(f"No T1w images found under {bids_root}{scope}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for ses_label, images in sorted(images_by_session.items()):
        queue_path = output_dir / f"{ses_label}_recon-any_queue.txt"
        lines = []

        for subject, t1w_path in images:
            subj_dir = sdir_root / ses_label / f"sub-{subject}"
            if subj_dir.exists():
                print(f"[skip] {subj_dir} already exists")
                continue
            lines.append(build_command(subject, ses_label, t1w_path, sdir_root, args.threads, args.side))

        if not lines:
            print(f"[queue] {ses_label}: nothing to queue")
            continue

        queue_path.write_text("\n".join(lines) + "\n")
        print(f"[queue] {queue_path} ({len(lines)} command(s))")


if __name__ == "__main__":
    main()
