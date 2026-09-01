#!/usr/bin/env python3
"""
Generate axial T1w QC mosaics for a BIDS dataset.

Walks sub-*/ses-*/anat/ for combined (non-echo) T1w files matching:
    sub-<subject>_ses-<session>_T1w.nii.gz
    sub-<subject>_ses-<session>_run-<n>_T1w.nii.gz

For each subject/session, the run-free T1w is preferred if present;
otherwise the lowest-numbered run-<n> T1w is used. At most one T1w is
used per subject/session (same selection logic as format_synthsr_batch.py).

For each T1w found, the volume is reoriented to canonical (RAS+) so that
axis 2 is the axial (superior-inferior) axis, the center axial slice is
taken as the geometric middle of that axis, and slices are sampled across
[center - --offset, center + --offset] (e.g. center=45, offset=20 ->
range 25-65): either --num-slices slices evenly spaced across that range
(default 40), or, if --stride is given, every stride-th slice instead
(e.g. --stride 2 for every other slice). Slices are tiled into a single
PNG mosaic:

    <output-dir>/ses-<session>/sub-<subject>_ses-<session>_T1w_mosaic.png

Subjects whose mosaic PNG already exists are skipped.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"ERROR: missing dependency ({e}). "
              "Install with: pip install numpy nibabel matplotlib --break-system-packages")

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


def pick_slices(center: int, offset: int, n_total: int, num_slices: int = None, stride: int = None) -> list:
    """Indices across [center-offset, center+offset], clipped to valid range and
    de-duplicated (rounding can collide neighbors).

    If stride is given, takes every stride-th slice (e.g. stride=2 -> every other
    slice). Otherwise takes num_slices indices evenly spaced across the range.
    """
    lo, hi = center - offset, center + offset
    if stride:
        raw = np.arange(lo, hi + 1, stride)
    else:
        raw = np.linspace(lo, hi, num_slices)
    idx = np.unique(np.round(raw).astype(int))
    idx = idx[(idx >= 0) & (idx < n_total)]
    return idx.tolist()


def build_mosaic(t1w_path: Path, out_path: Path, offset: int, num_slices: int, stride: int, cols: int) -> int:
    """Builds the mosaic PNG at out_path. Returns the number of slices used."""
    img = nib.load(str(t1w_path))
    canon = nib.as_closest_canonical(img)  # axis 2 becomes the axial (S-I) axis
    data = canon.get_fdata()

    n_axial = data.shape[2]
    center = n_axial // 2
    slice_idxs = pick_slices(center, offset, n_axial, num_slices=num_slices, stride=stride)
    if not slice_idxs:
        print(f"[warn] {t1w_path}: no valid axial slices in range, skipping")
        return 0

    n = len(slice_idxs)
    rows = -(-n // cols)  # ceil division

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), facecolor="black")
    axes = np.atleast_2d(axes)

    vmax = np.percentile(data, 99.5) or 1.0
    for ax_idx in range(rows * cols):
        r, c = divmod(ax_idx, cols)
        ax = axes[r][c]
        ax.set_facecolor("black")
        ax.axis("off")
        if ax_idx < n:
            sl = slice_idxs[ax_idx]
            slice_data = np.rot90(data[:, :, sl])
            ax.imshow(slice_data, cmap="gray", vmin=0, vmax=vmax)
            ax.text(0.02, 0.02, str(sl), color="yellow", fontsize=6,
                    transform=ax.transAxes, va="bottom", ha="left")

    fig.suptitle(out_path.stem, color="white", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.02, hspace=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bids_root", type=Path, help="Path to the BIDS dataset root")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Base directory for mosaic PNGs, e.g. /.../derivatives/t1_mosaics "
                              "(a ses-<session>/ subdirectory is created automatically)")
    parser.add_argument("--session", help="Restrict to a single session, e.g. --session 1 or --session ses-001")
    parser.add_argument("--offset", type=int, default=20,
                         help="Slices span [center-offset, center+offset] (default: 20)")
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument("--num-slices", type=int, default=40,
                              help="Number of slices evenly spaced across that range (default: 40). "
                                   "Ignored if --stride is given.")
    count_group.add_argument("--stride", type=int,
                              help="Take every stride-th slice across the range instead of a fixed "
                                   "count, e.g. --stride 2 for every other slice, --stride 3 for "
                                   "every 3rd slice. Overrides --num-slices.")
    parser.add_argument("--cols", type=int, default=8,
                         help="Number of mosaic grid columns (default: 8)")
    args = parser.parse_args()

    bids_root = args.bids_root.resolve()
    if not bids_root.is_dir():
        parser.error(f"{bids_root} is not a directory")

    output_dir = args.output_dir.resolve()

    images_by_session = find_t1w_images(bids_root, args.session)
    if not images_by_session:
        scope = f" for {normalize_session(args.session)}" if args.session else ""
        print(f"No T1w images found under {bids_root}{scope}")
        return

    total_built = 0
    for ses_label, images in sorted(images_by_session.items()):
        for subject, t1w_path in images:
            out_path = output_dir / ses_label / f"sub-{subject}_{ses_label}_T1w_mosaic.png"
            if out_path.exists():
                print(f"[skip] {out_path} already exists")
                continue

            n = build_mosaic(t1w_path, out_path, args.offset, args.num_slices, args.stride, args.cols)
            if n:
                print(f"[mosaic] {out_path} ({n} slices)")
                total_built += 1

    print(f"\nBuilt {total_built} mosaic(s)")


if __name__ == "__main__":
    main()
