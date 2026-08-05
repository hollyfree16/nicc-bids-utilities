#!/usr/bin/env python3
"""
Build recon-all command queues from a SynthSR output directory.

Walks ses-*/sub-*/ under a SynthSR output directory (as produced by
format_synthsr_batch.py) for synthesized T1w images matching:
    sub-<subject>_ses-<session>_T1w_synthsr.nii.gz

and, for each session found, writes a queue file containing one
recon-all command per subject:

    recon-all -i <synthsr T1w> -s sub-<subject> -sd <sd-root>/ses-<session> -all

Queue files are named "<session>_recon-all_queue.txt" and written to
--output-dir. Subjects whose directory already exists under
<sd-root>/ses-<session>/ are skipped (not written to the queue).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

SYNTHSR_RE = re.compile(r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_T1w_synthsr\.nii\.gz$")


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
    parser.add_argument("synthsr_root", type=Path, help="Path to the SynthSR output directory, e.g. "
                                                          "/.../derivatives/synthsr")
    parser.add_argument("--sd-root", type=Path, required=True,
                         help="Base FreeSurfer subjects directory, e.g. /.../derivatives/freesurfer_reconall_v8.2.0 "
                              "(a ses-<session> subdirectory is appended automatically)")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to write <session>_recon-all_queue.txt files to")
    parser.add_argument("--session", help="Restrict to a single session, e.g. --session 1 or --session ses-001")
    args = parser.parse_args()

    synthsr_root = args.synthsr_root.resolve()
    if not synthsr_root.is_dir():
        parser.error(f"{synthsr_root} is not a directory")

    sd_root = args.sd_root.resolve()
    output_dir = args.output_dir.resolve()

    images_by_session = find_synthsr_images(synthsr_root, args.session)
    if not images_by_session:
        scope = f" for {normalize_session(args.session)}" if args.session else ""
        print(f"No SynthSR T1w images found under {synthsr_root}{scope}")
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
