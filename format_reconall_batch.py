#!/usr/bin/env python3
"""
Build recon-all command queues from a SynthSR output directory or a BIDS root.

With --synthsr-root, walks ses-*/sub-*/ under a SynthSR output directory (as
produced by format_synthsr_batch.py) for synthesized T1w images matching:
    sub-<subject>_ses-<session>_T1w_synthsr.nii.gz

With --bids-root, walks sub-*/ses-*/anat/ under a BIDS dataset root (same
layout as average_echoes.py) for T1w images matching:
    sub-<subject>_ses-<session>[_<other entities>]_T1w.nii.gz
(echo-<n> files are excluded, so an averaged multi-echo output produced by
average_echoes.py is picked up correctly).

For each session found, writes a queue file containing one recon-all command
per subject:

    recon-all -i <T1w> -s sub-<subject> -sd <sd-root>/ses-<session> -all

Queue files are named "<session>_recon-all_queue.txt" and written to
--output-dir. Subjects whose directory already exists under
<sd-root>/ses-<session>/ are skipped (not written to the queue).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

SYNTHSR_RE = re.compile(r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_T1w_synthsr\.nii\.gz$")
BIDS_RE = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)"
    r"(?:_(?!echo-)[A-Za-z0-9]+-[^_]+)*_T1w\.nii\.gz$"
)


def normalize_session(session: str) -> str:
    """Turn a session flag value into a BIDS session label (e.g. "1" -> "ses-001")."""
    session = session[len("ses-"):] if session.startswith("ses-") else session
    if session.isdigit():
        session = session.zfill(3)
    return f"ses-{session}"


def find_synthsr_images(synthsr_root: Path, session: str = None):
    """Find SynthSR T1w images, grouped by session.

    Returns a dict mapping session label ("ses-Y") -> list of (subject, path)
    tuples, sorted by subject.
    """
    session_glob = normalize_session(session) if session else "ses-*"

    images = defaultdict(list)
    for subj_dir in sorted(synthsr_root.glob(f"{session_glob}/sub-*")):
        for f in subj_dir.iterdir():
            m = SYNTHSR_RE.match(f.name)
            if m:
                images[f"ses-{m.group('session')}"].append((m.group("subject"), f))

    for ses_label in images:
        images[ses_label].sort(key=lambda x: x[0])

    return images


def find_bids_images(bids_root: Path, session: str = None):
    """Find BIDS T1w images, grouped by session.

    Returns a dict mapping session label ("ses-Y") -> list of (subject, path)
    tuples, sorted by subject.
    """
    session_glob = normalize_session(session) if session else "ses-*"

    images = defaultdict(list)
    for anat_dir in sorted(bids_root.glob(f"sub-*/{session_glob}/anat")):
        for f in anat_dir.iterdir():
            m = BIDS_RE.match(f.name)
            if m:
                images[f"ses-{m.group('session')}"].append((m.group("subject"), f))

    for ses_label in images:
        images[ses_label].sort(key=lambda x: x[0])

    return images


def build_command(subject, ses_label, t1w_path: Path, sd_root: Path) -> str:
    subjid = f"sub-{subject}"
    sd = sd_root / ses_label

    return " ".join([
        "recon-all",
        "-i", str(t1w_path),
        "-s", subjid,
        "-sd", str(sd),
        "-all",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root_group = parser.add_mutually_exclusive_group(required=True)
    root_group.add_argument("--synthsr-root", type=Path,
                             help="Path to a SynthSR output directory, e.g. /.../derivatives/synthsr "
                                  "(expects ses-*/sub-*/sub-<s>_ses-<n>_T1w_synthsr.nii.gz)")
    root_group.add_argument("--bids-root", type=Path,
                             help="Path to a BIDS dataset root, e.g. /.../rawdata "
                                  "(expects sub-*/ses-*/anat/sub-<s>_ses-<n>_T1w.nii.gz)")
    parser.add_argument("--sd-root", type=Path, required=True,
                         help="Base FreeSurfer subjects directory, e.g. /.../derivatives/freesurfer_reconall_v8.2.0 "
                              "(a ses-<session> subdirectory is appended automatically)")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to write <session>_recon-all_queue.txt files to")
    parser.add_argument("--session", help="Restrict to a single session, e.g. --session 1 or --session ses-001")
    args = parser.parse_args()

    if args.synthsr_root:
        input_root = args.synthsr_root.resolve()
        source_label = "SynthSR"
        find_images = find_synthsr_images
    else:
        input_root = args.bids_root.resolve()
        source_label = "BIDS"
        find_images = find_bids_images

    if not input_root.is_dir():
        parser.error(f"{input_root} is not a directory")

    sd_root = args.sd_root.resolve()
    output_dir = args.output_dir.resolve()

    images_by_session = find_images(input_root, args.session)
    if not images_by_session:
        scope = f" for {normalize_session(args.session)}" if args.session else ""
        print(f"No {source_label} T1w images found under {input_root}{scope}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for ses_label, images in sorted(images_by_session.items()):
        queue_path = output_dir / f"{ses_label}_recon-all_queue.txt"
        lines = []

        for subject, t1w_path in images:
            subj_dir = sd_root / ses_label / f"sub-{subject}"
            if subj_dir.exists():
                print(f"[skip] {subj_dir} already exists")
                continue
            lines.append(build_command(subject, ses_label, t1w_path, sd_root))

        if not lines:
            print(f"[queue] {ses_label}: nothing to queue")
            continue

        queue_path.write_text("\n".join(lines) + "\n")
        print(f"[queue] {queue_path} ({len(lines)} command(s))")


if __name__ == "__main__":
    main()
