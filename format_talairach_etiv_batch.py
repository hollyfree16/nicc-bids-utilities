#!/usr/bin/env python3
"""
Build talairach registration + eTIV command queues for a BIDS dataset.

Walks sub-*/ses-*/anat/ for combined (non-echo) T1w files matching:
    sub-<subject>_ses-<session>_T1w.nii.gz

and, for each session found, writes a queue file containing one command
chain per subject that runs the talairach_avi pipeline directly (the same
steps -autorecon1 calls internally) and then computes eTIV:

    mri_convert --conform <T1w> orig.mgz
    mri_nu_correct.mni --i orig.mgz --o nu.mgz
    talairach_avi --i nu.mgz --xfm talairach.xfm
    mri_segstats --etiv-only --talxfm talairach.xfm

Per-subject working files (orig.mgz, nu.mgz, talairach.xfm) are written under
--output-dir/ses-<session>/sub-<subject>/. eTIV results are appended as rows
to <session>_etiv.csv in --queue-dir.

Queue files are named "<session>_talairach_etiv_queue.txt" and written to
--queue-dir. Subjects whose talairach.xfm already exists under
--output-dir/ses-<session>/sub-<subject>/ are skipped (not written to the
queue).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

T1W_RE = re.compile(r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_T1w\.nii\.gz$")

CSV_HEADER = "subject,session,etiv\n"


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


def build_command(subject, t1w_path: Path, subj_dir: Path, csv_path: Path) -> str:
    ses_label = subj_dir.parent.name

    steps = [
        f"mri_convert --conform {t1w_path} orig.mgz",
        "mri_nu_correct.mni --i orig.mgz --o nu.mgz",
        "talairach_avi --i nu.mgz --xfm talairach.xfm",
        f'echo "{subject},{ses_label},$(mri_segstats --etiv-only --talxfm talairach.xfm)" >> {csv_path}',
    ]

    return f"mkdir -p {subj_dir} && (cd {subj_dir} && " + " && ".join(steps) + ")"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids_root", type=Path, help="Path to the BIDS dataset root")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Base directory for per-subject working files "
                              "(orig.mgz, nu.mgz, talairach.xfm), e.g. /.../derivatives/talairach_etiv "
                              "(a ses-<session>/sub-<subject> subdirectory is created automatically)")
    parser.add_argument("--queue-dir", type=Path, required=True,
                         help="Directory to write <session>_talairach_etiv_queue.txt and "
                              "<session>_etiv.csv files to")
    parser.add_argument("--session", help="Restrict to a single session, e.g. --session 1 or --session ses-001")
    args = parser.parse_args()

    bids_root = args.bids_root.resolve()
    if not bids_root.is_dir():
        parser.error(f"{bids_root} is not a directory")

    output_dir = args.output_dir.resolve()
    queue_dir = args.queue_dir.resolve()

    images_by_session = find_t1w_images(bids_root, args.session)
    if not images_by_session:
        scope = f" for {normalize_session(args.session)}" if args.session else ""
        print(f"No T1w images found under {bids_root}{scope}")
        return

    queue_dir.mkdir(parents=True, exist_ok=True)

    for ses_label, images in sorted(images_by_session.items()):
        queue_path = queue_dir / f"{ses_label}_talairach_etiv_queue.txt"
        csv_path = queue_dir / f"{ses_label}_etiv.csv"

        if not csv_path.exists():
            csv_path.write_text(CSV_HEADER)

        lines = []
        for subject, t1w_path in images:
            subj_dir = output_dir / ses_label / f"sub-{subject}"
            xfm_path = subj_dir / "talairach.xfm"

            if xfm_path.exists():
                print(f"[skip] {xfm_path} already exists")
                continue

            lines.append(build_command(subject, t1w_path, subj_dir, csv_path))

        if not lines:
            print(f"[queue] {ses_label}: nothing to queue")
            continue

        queue_path.write_text("\n".join(lines) + "\n")
        print(f"[queue] {queue_path} ({len(lines)} command(s))")


if __name__ == "__main__":
    main()
