#!/usr/bin/env python3
"""
Open a T1w volume with a label volume overlaid in freeview, one subject at a time.

For each subject, opens:
    $BIDS_DIR/<subject>/ses-<session>/anat/*T1w.nii.gz   (bottom layer)
    $MODEL_DIR/ses-<session>/<subject>.nii.gz            (top layer, colormap=lut, outline=1)

freeview blocks until the viewer window is closed, so subjects are reviewed
sequentially: close the window to advance to the next subject.
"""

import argparse
import subprocess
from pathlib import Path


def find_t1w(bids_dir: Path, subject: str, session: str) -> Path | None:
    anat_dir = bids_dir / subject / f"ses-{session}" / "anat"
    matches = sorted(anat_dir.glob("*T1w.nii.gz"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[info] {subject}: multiple T1w matches under {anat_dir}, using {matches[0].name}")
    return matches[0]


def load_subjects(args) -> list[str]:
    if args.subjects_file:
        text = args.subjects_file.read_text()
        return [line.strip() for line in text.splitlines() if line.strip()]
    return args.subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bids-dir", type=Path, required=True, help="BIDS dataset root")
    parser.add_argument("--model-dir", type=Path, required=True,
                         help="Root of label volumes, e.g. /.../model "
                              "(expects <model-dir>/ses-<session>/<subject>.nii.gz)")
    parser.add_argument("--session", required=True,
                         help="Session label (without the 'ses-' prefix), used to build "
                              "ses-<session> under both --bids-dir and --model-dir")
    subj_group = parser.add_mutually_exclusive_group(required=True)
    subj_group.add_argument("--subjects", nargs="+", help="One or more subject IDs (e.g. sub-MGHL2p001)")
    subj_group.add_argument("--subjects-file", type=Path, help="Text file with one subject ID per line")
    args = parser.parse_args()

    bids_dir = args.bids_dir.resolve()
    model_dir = args.model_dir.resolve()

    if not bids_dir.is_dir():
        parser.error(f"{bids_dir} is not a directory")
    if not model_dir.is_dir():
        parser.error(f"{model_dir} is not a directory")

    subjects = load_subjects(args)
    if not subjects:
        print("No subjects to review")
        return

    for subject in subjects:
        t1w_path = find_t1w(bids_dir, subject, args.session)
        if t1w_path is None:
            print(f"[skip] {subject}: no T1w found under {bids_dir / subject / f'ses-{args.session}' / 'anat'}")
            continue

        label_path = model_dir / f"ses-{args.session}" / f"{subject}.nii.gz"
        if not label_path.is_file():
            print(f"[skip] {subject}: no label volume found at {label_path}")
            continue

        print(f"[freeview] {subject} (ses-{args.session}): {t1w_path.name} + {label_path.name}")
        subprocess.run([
            "freeview",
            "-v", str(t1w_path), f"{label_path}:colormap=lut:outline=1",
        ])


if __name__ == "__main__":
    main()
