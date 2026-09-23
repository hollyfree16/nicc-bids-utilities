#!/usr/bin/env python3
"""
Compute lesion-corrected Yeo-7 network volumes for FreeSurfer subjects.

For each subject that has already completed `recon-all`, this pipeline:

  1. Projects the Schaefer2018 1000-parcel / Yeo-7-network cortical
     parcellation from fsaverage onto the subject's native surface
     (`mri_surf2surf`).
  2. Builds a subject-native VOLUMETRIC Schaefer parcellation using the
     cortical ribbon (`mri_aparc2aseg`). This volume -- not the surface
     stats produced by `mris_anatomical_stats` -- is the single reference
     space for every volume/overlap calculation below.
  3. Aligns the subject's lesion segmentation to that same volumetric grid
     (using an explicit, user-supplied transform, or an exact-match /
     user-asserted native-header path -- never a silent guess).
  4. Restricts the lesion to the FreeSurfer brainmask.
  5. Computes, per Yeo-7 network (bilateral, LH, RH) and per Schaefer
     parcel: raw volume, lesion overlap volume, percent lesioned, and a
     lesion-corrected ("lesion-excluded") volume = raw - overlap.

See README_yeo7_lesion_corrected_volumes.md for the full methodology,
required lesion-space assumptions, and output schema.

Example
-------
    python extract_yeo7_lesion_corrected_volumes.py \\
        --subjects-dir /path/to/derivatives/freesurfer/ses-001 \\
        --subjects-file subjects.txt \\
        --atlas-dir /path/to/Schaefer2018/FreeSurfer5.3/fsaverage/label \\
        --lesion-dir /path/to/lesions \\
        --session 001 \\
        --output-dir ./yeo7_lesion_results
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REQUIRED_FS_COMMANDS = ["mri_surf2surf", "mri_aparc2aseg", "mri_vol2vol", "mri_info"]

ATLAS_BASENAME = "Schaefer2018_1000Parcels_7Networks_order"

REQUIRED_SUBJECT_FILES = [
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
    "mri/aseg.mgz",
]

# Schaefer/Yeo-7 network token (as embedded in parcel names, e.g.
# "7Networks_LH_SalVentAttn_Med_1") -> canonical network name.
NETWORK_TOKEN_MAP = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "DorsalAttention",
    "SalVentAttn": "SalienceVentralAttention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}
NETWORK_ORDER = [
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "SalienceVentralAttention",
    "Limbic",
    "Control",
    "Default",
]
NETWORK_INDEX = {net: i + 1 for i, net in enumerate(NETWORK_ORDER)}  # for QC label volumes

GEOM_ATOL = 1e-3
VOLUME_TOLERANCE_VOXELS = 1e-6  # for corrected + overlap == raw sanity check


log = logging.getLogger("yeo7_lesion")


class SubjectStageError(Exception):
    """Raised to abort processing of a single subject at a named stage."""

    def __init__(self, stage: str, reason: str, stderr: str = ""):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.stderr = stderr


# --------------------------------------------------------------------------
# Environment / input validation
# --------------------------------------------------------------------------

def check_fs_environment() -> None:
    missing = [c for c in REQUIRED_FS_COMMANDS if shutil.which(c) is None]
    if missing:
        raise EnvironmentError(
            "Missing required FreeSurfer commands on PATH: "
            + ", ".join(missing)
            + ". Source FreeSurfer (e.g. `source $FREESURFER_HOME/SetUpFreeSurfer.sh`) first."
        )
    if nib is None:
        raise EnvironmentError("nibabel is required (`pip install nibabel numpy pandas`).")


def validate_atlas_dir(atlas_dir: Path) -> tuple[Path, Path]:
    lh = atlas_dir / f"lh.{ATLAS_BASENAME}.annot"
    rh = atlas_dir / f"rh.{ATLAS_BASENAME}.annot"
    missing = [str(p) for p in (lh, rh) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing Schaefer atlas annotation file(s): " + "; ".join(missing))
    return lh, rh


def load_subjects(subjects_file: Path) -> list[str]:
    subjects = []
    for line in subjects_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        subjects.append(line)
    return subjects


# --------------------------------------------------------------------------
# Per-subject path bookkeeping
# --------------------------------------------------------------------------

@dataclass
class SubjectPaths:
    subject: str
    subj_dir: Path
    sphere_lh: Path
    sphere_rh: Path
    white_lh: Path
    white_rh: Path
    pial_lh: Path
    pial_rh: Path
    brainmask: Path
    aseg: Path
    annot_lh: Path
    annot_rh: Path
    schaefer_vol: Path
    intermediate_dir: Path
    qc_dir: Path
    log_file: Path


def build_subject_paths(subject: str, subjects_dir: Path, output_dir: Path, brainmask_name: str) -> SubjectPaths:
    sdir = subjects_dir / subject
    return SubjectPaths(
        subject=subject,
        subj_dir=sdir,
        sphere_lh=sdir / "surf" / "lh.sphere.reg",
        sphere_rh=sdir / "surf" / "rh.sphere.reg",
        white_lh=sdir / "surf" / "lh.white",
        white_rh=sdir / "surf" / "rh.white",
        pial_lh=sdir / "surf" / "lh.pial",
        pial_rh=sdir / "surf" / "rh.pial",
        brainmask=sdir / "mri" / brainmask_name,
        aseg=sdir / "mri" / "aseg.mgz",
        annot_lh=sdir / "label" / f"lh.{ATLAS_BASENAME}.annot",
        annot_rh=sdir / "label" / f"rh.{ATLAS_BASENAME}.annot",
        schaefer_vol=sdir / "mri" / f"{ATLAS_BASENAME}.mgz",
        intermediate_dir=output_dir / "intermediate" / subject,
        qc_dir=output_dir / "qc" / subject,
        log_file=output_dir / "logs" / f"{subject}.log",
    )


def validate_subject_inputs(paths: SubjectPaths, lesion_path: Path) -> None:
    missing = []
    for rel in REQUIRED_SUBJECT_FILES:
        p = paths.subj_dir / rel
        if not p.is_file():
            missing.append(str(p))
    if not paths.brainmask.is_file():
        missing.append(str(paths.brainmask))
    if not lesion_path.is_file():
        missing.append(str(lesion_path))
    if missing:
        raise SubjectStageError("input validation", "Missing required file(s): " + "; ".join(missing))


# --------------------------------------------------------------------------
# Subprocess helper
# --------------------------------------------------------------------------

def run_fs(cmd: list[str], env: dict, log_file: Optional[Path], stage: str) -> subprocess.CompletedProcess:
    log.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as fh:
            fh.write(f"$ {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n\n")
    if proc.returncode != 0:
        raise SubjectStageError(
            stage,
            f"`{cmd[0]}` exited with code {proc.returncode}",
            stderr=(proc.stderr or "")[-4000:],
        )
    return proc


# --------------------------------------------------------------------------
# Step: project Schaefer atlas onto subject surface
# --------------------------------------------------------------------------

def project_schaefer_annotations(paths: SubjectPaths, atlas_lh: Path, atlas_rh: Path, subject: str,
                                  overwrite: bool, env: dict) -> None:
    for hemi, atlas_annot, out_annot in (("lh", atlas_lh, paths.annot_lh), ("rh", atlas_rh, paths.annot_rh)):
        if out_annot.is_file() and not overwrite:
            log.info("[%s] %s: %s already exists, skipping projection (use --overwrite to regenerate)",
                      subject, hemi, out_annot.name)
            continue
        out_annot.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", subject,
            "--hemi", hemi,
            "--sval-annot", str(atlas_annot),
            "--tval", str(out_annot),
        ]
        run_fs(cmd, env, paths.log_file, "Schaefer surface projection")
        if not out_annot.is_file():
            raise SubjectStageError("Schaefer surface projection", f"{out_annot} was not created")


# --------------------------------------------------------------------------
# Step: build subject-native volumetric Schaefer segmentation
# --------------------------------------------------------------------------

def build_schaefer_volume(paths: SubjectPaths, subject: str, overwrite: bool, env: dict) -> None:
    if paths.schaefer_vol.is_file() and not overwrite:
        log.info("[%s] %s already exists, skipping (use --overwrite to regenerate)",
                  subject, paths.schaefer_vol.name)
        return
    cmd = [
        "mri_aparc2aseg",
        "--s", subject,
        "--annot", ATLAS_BASENAME,
        "--new-ribbon",
        "--o", str(paths.schaefer_vol),
    ]
    run_fs(cmd, env, paths.log_file, "Schaefer volume creation")
    if not paths.schaefer_vol.is_file():
        raise SubjectStageError("Schaefer volume creation", f"{paths.schaefer_vol} was not created")


# --------------------------------------------------------------------------
# Step: parse annotation -> parcel -> network LUT
# --------------------------------------------------------------------------

def parse_annot_networks(annot_path: Path, hemi_token: str) -> dict[int, tuple[str, Optional[str]]]:
    """Return {ctab_index: (parcel_name, network_or_None)} for one hemisphere annot."""
    labels, ctab, names = nib.freesurfer.read_annot(str(annot_path))
    mapping: dict[int, tuple[str, Optional[str]]] = {}
    for idx, raw_name in enumerate(names):
        name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
        tokens = name.split("_")
        network = None
        if len(tokens) >= 3 and tokens[1].upper() == hemi_token.upper():
            network = NETWORK_TOKEN_MAP.get(tokens[2])
        mapping[idx] = (name, network)
    return mapping


def build_label_lut(annot_lh: Path, annot_rh: Path) -> dict[int, dict]:
    """
    Map volumetric Schaefer label IDs (as produced by mri_aparc2aseg) to
    parcel name / hemisphere / Yeo network.

    mri_aparc2aseg follows the standard FreeSurfer convention of encoding
    cortical labels as 1000 + <color-table index> for the left hemisphere
    and 2000 + <color-table index> for the right hemisphere, where the
    color-table index is the 0-based position of the parcel in the
    annotation's embedded color table (the same order `nibabel` returns
    via `read_annot`'s `names`). This is validated against the actual
    volumetric segmentation in `validate_label_lut`.
    """
    lut: dict[int, dict] = {}
    for hemi_token, offset, annot_path in (("LH", 1000, annot_lh), ("RH", 2000, annot_rh)):
        mapping = parse_annot_networks(annot_path, hemi_token)
        for idx, (name, network) in mapping.items():
            if network is None:
                continue
            label_id = offset + idx
            lut[label_id] = {"parcel": name, "hemisphere": hemi_token, "network": network}
    return lut


def validate_label_lut(lut: dict[int, dict], schaefer_data: np.ndarray) -> tuple[set[int], list[str]]:
    warnings: list[str] = []
    present_ids = set(np.unique(schaefer_data).astype(np.int64).tolist())
    expected_ids = set(lut.keys())
    matched = expected_ids & present_ids
    if not matched:
        raise SubjectStageError(
            "LUT parsing",
            "None of the expected Schaefer label IDs were found in the volumetric segmentation "
            "(label-ID mapping is likely wrong or mri_aparc2aseg failed silently).",
        )
    coverage = len(matched) / len(expected_ids)
    if coverage < 0.9:
        warnings.append(
            f"Only {coverage:.1%} of expected Schaefer parcel labels ({len(matched)}/{len(expected_ids)}) "
            "were found in the volumetric segmentation."
        )
    return matched, warnings


# --------------------------------------------------------------------------
# Step: lesion geometry / alignment
# --------------------------------------------------------------------------

def geometries_match(img_a, img_b, atol: float = GEOM_ATOL) -> bool:
    return img_a.shape[:3] == img_b.shape[:3] and np.allclose(
        np.asarray(img_a.affine), np.asarray(img_b.affine), atol=atol
    )


def prepare_lesion_volume(subject: str, lesion_path: Path, schaefer_path: Path, out_path: Path,
                           lesion_xfm_pattern: Optional[str], lesion_xfm_type: str,
                           lesion_native_header: bool, env: dict, log_file: Path,
                           warnings: list[str]) -> Path:
    lesion_img = nib.load(str(lesion_path))
    target_img = nib.load(str(schaefer_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if geometries_match(lesion_img, target_img):
        nib.save(nib.Nifti1Image(np.asarray(lesion_img.dataobj), target_img.affine, target_img.header),
                  str(out_path))
        return out_path

    if lesion_xfm_pattern:
        xfm_path = Path(lesion_xfm_pattern.format(subject=subject))
        if not xfm_path.is_file():
            raise SubjectStageError("lesion registration", f"Transform file not found: {xfm_path}")
        cmd = [
            "mri_vol2vol",
            "--mov", str(lesion_path),
            "--targ", str(schaefer_path),
            "--o", str(out_path),
            "--interp", "nearest",
        ]
        cmd += ["--lta", str(xfm_path)] if lesion_xfm_type == "lta" else ["--fsl", str(xfm_path)]
        run_fs(cmd, env, log_file, "lesion registration")
    elif lesion_native_header:
        cmd = [
            "mri_vol2vol",
            "--mov", str(lesion_path),
            "--targ", str(schaefer_path),
            "--o", str(out_path),
            "--regheader",
            "--interp", "nearest",
        ]
        run_fs(cmd, env, log_file, "lesion registration")
        warnings.append(
            "Lesion resampled via --regheader (user-asserted native-header alignment with the "
            "recon-all T1) -- visually QC the overlay before trusting these numbers."
        )
    else:
        raise SubjectStageError(
            "lesion registration",
            "Lesion geometry (shape/affine) does not match the subject's Schaefer/FreeSurfer volume, "
            "and no --lesion-xfm-pattern or --lesion-native-header was supplied. Refusing to guess a "
            "registration between potentially unrelated coordinate systems (e.g. MNI vs. native).",
        )

    resampled_img = nib.load(str(out_path))
    if not geometries_match(resampled_img, target_img):
        raise SubjectStageError(
            "lesion registration",
            f"Resampled lesion grid still does not match the Schaefer volume grid "
            f"(shape {resampled_img.shape[:3]} vs {target_img.shape[:3]}).",
        )
    return out_path


def prepare_brainmask_volume(paths: SubjectPaths, schaefer_path: Path, env: dict,
                              warnings: list[str]) -> np.ndarray:
    brain_img = nib.load(str(paths.brainmask))
    target_img = nib.load(str(schaefer_path))
    if geometries_match(brain_img, target_img):
        return np.asarray(brain_img.dataobj)

    out_path = paths.intermediate_dir / "brainmask_in_fs_space.nii.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mri_vol2vol",
        "--mov", str(paths.brainmask),
        "--targ", str(schaefer_path),
        "--o", str(out_path),
        "--regheader",
        "--interp", "nearest",
    ]
    run_fs(cmd, env, paths.log_file, "brainmask restriction")
    warnings.append("brainmask.mgz did not share the Schaefer volume's grid; resampled via --regheader "
                     "(same-subject, same-pipeline output -- header-based alignment is safe here).")
    resampled = nib.load(str(out_path))
    if not geometries_match(resampled, target_img):
        raise SubjectStageError("brainmask restriction",
                                 "Resampled brainmask grid still does not match the Schaefer volume grid.")
    return np.asarray(resampled.dataobj)


# --------------------------------------------------------------------------
# Core per-subject computation
# --------------------------------------------------------------------------

def process_subject(subject: str, cfg: dict) -> dict:
    subjects_dir = Path(cfg["subjects_dir"])
    output_dir = Path(cfg["output_dir"])
    atlas_lh = Path(cfg["atlas_lh"])
    atlas_rh = Path(cfg["atlas_rh"])
    lesion_path = Path(cfg["lesion_path"])
    overwrite = cfg["overwrite"]
    lesion_threshold = cfg["lesion_threshold"]
    env = cfg["env"]

    paths = build_subject_paths(subject, subjects_dir, output_dir, cfg["brainmask_name"])
    warnings: list[str] = []

    try:
        validate_subject_inputs(paths, lesion_path)

        project_schaefer_annotations(paths, atlas_lh, atlas_rh, subject, overwrite, env)
        build_schaefer_volume(paths, subject, overwrite, env)

        schaefer_img = nib.load(str(paths.schaefer_vol))
        schaefer_data = np.rint(np.asarray(schaefer_img.dataobj)).astype(np.int32)
        zooms = schaefer_img.header.get_zooms()[:3]
        voxvol = float(zooms[0] * zooms[1] * zooms[2])

        lut = build_label_lut(paths.annot_lh, paths.annot_rh)
        matched_ids, lut_warnings = validate_label_lut(lut, schaefer_data)
        warnings.extend(lut_warnings)

        intermediate_lesion_path = paths.intermediate_dir / "lesion_in_fs_space.nii.gz"
        prepare_lesion_volume(
            subject, lesion_path, paths.schaefer_vol, intermediate_lesion_path,
            cfg["lesion_xfm_pattern"], cfg["lesion_xfm_type"], cfg["lesion_native_header"],
            env, paths.log_file, warnings,
        )
        lesion_img = nib.load(str(intermediate_lesion_path))
        lesion_data = np.asarray(lesion_img.dataobj, dtype=np.float64)
        lesion_binary = lesion_data > lesion_threshold
        lesion_voxels_before_mask = int(lesion_binary.sum())

        brain_data = prepare_brainmask_volume(paths, paths.schaefer_vol, env, warnings)
        brain_binary = brain_data > 0
        lesion_brain = lesion_binary & brain_binary
        lesion_voxels_after_mask = int(lesion_brain.sum())

        if lesion_voxels_before_mask > 0 and lesion_voxels_after_mask == 0:
            warnings.append("All lesion voxels fall outside the FreeSurfer brainmask.")
        elif lesion_voxels_after_mask == 0:
            warnings.append("No lesion voxels found (before or after brainmask restriction).")

        total_lesion_mm3 = lesion_voxels_after_mask * voxvol

        # -- single pass over Schaefer parcels: per-parcel + per-network stats,
        #    plus QC label volumes (network id, lesion/network overlap id). --
        network_id_vol = np.zeros(schaefer_data.shape, dtype=np.int16)
        lesion_overlap_vol = np.zeros(schaefer_data.shape, dtype=np.int16)
        stats = {hemi: {net: {"raw": 0, "overlap": 0} for net in NETWORK_ORDER} for hemi in ("LH", "RH")}
        parcel_rows = []

        for label_id, info in lut.items():
            parcel_mask = schaefer_data == label_id
            raw_count = int(parcel_mask.sum())
            overlap_count = 0
            if raw_count > 0:
                overlap_mask = parcel_mask & lesion_brain
                overlap_count = int(overlap_mask.sum())
                stats[info["hemisphere"]][info["network"]]["raw"] += raw_count
                stats[info["hemisphere"]][info["network"]]["overlap"] += overlap_count
                network_id_vol[parcel_mask] = NETWORK_INDEX[info["network"]]
                if overlap_count > 0:
                    lesion_overlap_vol[overlap_mask] = NETWORK_INDEX[info["network"]]

            parcel_rows.append({
                "subject": subject,
                "hemisphere": info["hemisphere"],
                "parcel": info["parcel"],
                "network": info["network"],
                "raw_parcel_volume_mm3": raw_count * voxvol,
                "lesion_overlap_mm3": overlap_count * voxvol,
                "lesion_percent": (100.0 * overlap_count / raw_count) if raw_count > 0 else 0.0,
                "corrected_parcel_volume_mm3": (raw_count - overlap_count) * voxvol,
                "touched": overlap_count > 0,
            })

        # sanity check: overlap can never exceed raw (guaranteed by construction, checked anyway)
        for hemi in ("LH", "RH"):
            for net in NETWORK_ORDER:
                s = stats[hemi][net]
                if s["overlap"] > s["raw"]:
                    raise SubjectStageError(
                        "overlap calculation",
                        f"{hemi} {net}: lesion overlap ({s['overlap']}) exceeds raw network voxel count "
                        f"({s['raw']}); this indicates a bug in mask construction.",
                    )

        long_rows = []
        subject_row = {"subject": subject, "voxel_volume_mm3": voxvol}
        lesion_within_yeo_mm3 = 0.0

        for net in NETWORK_ORDER:
            per_hemi = {}
            for hemi in ("LH", "RH"):
                raw = stats[hemi][net]["raw"]
                overlap = stats[hemi][net]["overlap"]
                raw_mm3 = raw * voxvol
                overlap_mm3 = overlap * voxvol
                corrected_mm3 = raw_mm3 - overlap_mm3
                pct = (100.0 * overlap_mm3 / raw_mm3) if raw_mm3 > 0 else 0.0
                per_hemi[hemi] = (raw_mm3, overlap_mm3, corrected_mm3, pct, overlap > 0)
                subject_row[f"{hemi}_{net}_raw_mm3"] = raw_mm3
                subject_row[f"{hemi}_{net}_lesion_overlap_mm3"] = overlap_mm3
                subject_row[f"{hemi}_{net}_lesion_pct"] = pct
                subject_row[f"{hemi}_{net}_corrected_mm3"] = corrected_mm3
                long_rows.append({
                    "subject": subject, "hemisphere": hemi, "network": net,
                    "raw_volume_mm3": raw_mm3, "lesion_overlap_mm3": overlap_mm3,
                    "lesion_percent": pct, "corrected_volume_mm3": corrected_mm3,
                    "touched": overlap > 0,
                })

            bilateral_raw_mm3 = per_hemi["LH"][0] + per_hemi["RH"][0]
            bilateral_overlap_mm3 = per_hemi["LH"][1] + per_hemi["RH"][1]
            bilateral_corrected_mm3 = bilateral_raw_mm3 - bilateral_overlap_mm3
            bilateral_pct = (100.0 * bilateral_overlap_mm3 / bilateral_raw_mm3) if bilateral_raw_mm3 > 0 else 0.0
            bilateral_touched = bilateral_overlap_mm3 > 0

            if abs((bilateral_corrected_mm3 + bilateral_overlap_mm3) - bilateral_raw_mm3) > VOLUME_TOLERANCE_VOXELS * voxvol:
                raise SubjectStageError("overlap calculation",
                                         f"{net}: corrected + overlap != raw beyond numerical tolerance.")

            subject_row[f"{net}_raw_mm3"] = bilateral_raw_mm3
            subject_row[f"{net}_lesion_overlap_mm3"] = bilateral_overlap_mm3
            subject_row[f"{net}_lesion_pct"] = bilateral_pct
            subject_row[f"{net}_corrected_mm3"] = bilateral_corrected_mm3
            subject_row[f"{net}_touched"] = bilateral_touched

            long_rows.append({
                "subject": subject, "hemisphere": "Bilateral", "network": net,
                "raw_volume_mm3": bilateral_raw_mm3, "lesion_overlap_mm3": bilateral_overlap_mm3,
                "lesion_percent": bilateral_pct, "corrected_volume_mm3": bilateral_corrected_mm3,
                "touched": bilateral_touched,
            })
            lesion_within_yeo_mm3 += bilateral_overlap_mm3

        lesion_outside_yeo_mm3 = total_lesion_mm3 - lesion_within_yeo_mm3
        pct_lesion_within_yeo = (100.0 * lesion_within_yeo_mm3 / total_lesion_mm3) if total_lesion_mm3 > 0 else 0.0

        touched_networks = [net for net in NETWORK_ORDER if subject_row[f"{net}_touched"]]
        if lesion_voxels_after_mask > 0 and not touched_networks:
            warnings.append("Lesion is entirely outside Yeo cortical networks (likely white matter/"
                             "subcortical) -- not treated as a processing failure.")

        subject_row.update({
            "TotalLesion_mm3": total_lesion_mm3,
            "TotalLesion_mL": total_lesion_mm3 / 1000.0,
            "LesionWithinYeoCortex_mm3": lesion_within_yeo_mm3,
            "LesionOutsideYeoCortex_mm3": lesion_outside_yeo_mm3,
            "PercentLesionWithinYeoCortex": pct_lesion_within_yeo,
            "Num_Yeo_Networks_Touched": len(touched_networks),
            "Networks_Touched": ";".join(touched_networks),
        })
        # reorder: identifiers/globals first
        ordered_row = {k: subject_row[k] for k in [
            "subject", "voxel_volume_mm3", "TotalLesion_mm3", "TotalLesion_mL",
            "LesionWithinYeoCortex_mm3", "LesionOutsideYeoCortex_mm3", "PercentLesionWithinYeoCortex",
            "Num_Yeo_Networks_Touched", "Networks_Touched",
        ]}
        for net in NETWORK_ORDER:
            for suffix in ("raw_mm3", "lesion_overlap_mm3", "lesion_pct", "corrected_mm3", "touched"):
                ordered_row[f"{net}_{suffix}"] = subject_row[f"{net}_{suffix}"]
        for hemi in ("LH", "RH"):
            for net in NETWORK_ORDER:
                for suffix in ("raw_mm3", "lesion_overlap_mm3", "lesion_pct", "corrected_mm3"):
                    ordered_row[f"{hemi}_{net}_{suffix}"] = subject_row[f"{hemi}_{net}_{suffix}"]

        # -- QC volumes --
        paths.qc_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.schaefer_vol, paths.qc_dir / paths.schaefer_vol.name)
        nib.save(nib.Nifti1Image(lesion_brain.astype(np.uint8), schaefer_img.affine, schaefer_img.header),
                  str(paths.qc_dir / "lesion_brain_binary.nii.gz"))
        nib.save(nib.Nifti1Image(network_id_vol, schaefer_img.affine, schaefer_img.header),
                  str(paths.qc_dir / "network_id_volume.nii.gz"))
        nib.save(nib.Nifti1Image(lesion_overlap_vol, schaefer_img.affine, schaefer_img.header),
                  str(paths.qc_dir / "lesion_overlap_by_network.nii.gz"))

        qc_row = {
            "subject": subject,
            "lesion_file_found": True,
            "lesion_aligned": True,
            "lesion_voxels_before_mask": lesion_voxels_before_mask,
            "lesion_voxels_after_mask": lesion_voxels_after_mask,
            "schaefer_parcels_found": len(matched_ids),
            "schaefer_parcels_expected": len(lut),
            "raw_yeo_cortical_volume_mm3": sum(ordered_row[f"{n}_raw_mm3"] for n in NETWORK_ORDER),
            "lesion_within_yeo_cortex_mm3": lesion_within_yeo_mm3,
            "lesion_outside_yeo_cortex_mm3": lesion_outside_yeo_mm3,
            "num_networks_touched": len(touched_networks),
            "warnings": "; ".join(warnings),
        }

        return {"status": "ok", "subject": subject, "row": ordered_row, "long_rows": long_rows,
                "parcel_rows": parcel_rows, "qc_row": qc_row, "warnings": warnings}

    except SubjectStageError as exc:
        log.error("[%s] FAILED at stage '%s': %s", subject, exc.stage, exc.reason)
        return {
            "status": "failed", "subject": subject, "stage": exc.stage,
            "reason": exc.reason, "stderr": exc.stderr,
            "qc_row": {
                "subject": subject, "lesion_file_found": lesion_path.is_file(), "lesion_aligned": False,
                "lesion_voxels_before_mask": None, "lesion_voxels_after_mask": None,
                "schaefer_parcels_found": None, "schaefer_parcels_expected": None,
                "raw_yeo_cortical_volume_mm3": None, "lesion_within_yeo_cortex_mm3": None,
                "lesion_outside_yeo_cortex_mm3": None, "num_networks_touched": None,
                "warnings": f"FAILED at {exc.stage}: {exc.reason}",
            },
        }
    except Exception as exc:  # noqa: BLE001 - keep batch alive on unexpected errors too
        log.exception("[%s] unexpected error", subject)
        return {
            "status": "failed", "subject": subject, "stage": "unexpected error",
            "reason": str(exc), "stderr": "",
            "qc_row": {
                "subject": subject, "lesion_file_found": lesion_path.is_file(), "lesion_aligned": False,
                "lesion_voxels_before_mask": None, "lesion_voxels_after_mask": None,
                "schaefer_parcels_found": None, "schaefer_parcels_expected": None,
                "raw_yeo_cortical_volume_mm3": None, "lesion_within_yeo_cortex_mm3": None,
                "lesion_outside_yeo_cortex_mm3": None, "num_networks_touched": None,
                "warnings": f"FAILED (unexpected error): {exc}",
            },
        }


# --------------------------------------------------------------------------
# Lesion path resolution
# --------------------------------------------------------------------------

def resolve_lesion_path(lesion_dir: Path, pattern: str, subject: str, session: Optional[str]) -> Path:
    return lesion_dir / pattern.format(subject=subject, session=session or "")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects-dir", type=Path, required=True, help="FreeSurfer SUBJECTS_DIR")
    parser.add_argument("--subjects-file", type=Path, required=True,
                         help="Text file with one subject ID per line (blank lines and lines starting "
                              "with '#' are ignored)")
    parser.add_argument("--atlas-dir", type=Path, required=True,
                         help="Directory containing lh/rh.Schaefer2018_1000Parcels_7Networks_order.annot")
    parser.add_argument("--lesion-dir", type=Path, required=True, help="Directory containing lesion segmentations")
    parser.add_argument("--lesion-pattern", default=None,
                         help="Filename pattern relative to --lesion-dir, containing '{subject}' and "
                              "optionally '{session}'. Default: 'ses-{session}/{subject}.nii.gz' if "
                              "--session is given (matching freeview_qc_batch.py's layout), otherwise "
                              "'{subject}_lesion.nii.gz'.")
    parser.add_argument("--session", default=None,
                         help="Session label without the 'ses-' prefix, e.g. '001'. Used to resolve the "
                              "default --lesion-pattern and to substitute '{session}' in a custom pattern.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--overwrite", action="store_true",
                         help="Regenerate subject annotations / Schaefer volume even if they already exist")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of subjects to process in parallel")
    parser.add_argument("--lesion-threshold", type=float, default=0.5,
                         help="Lesion probability/intensity threshold; voxels > threshold are lesion (default 0.5)")
    parser.add_argument("--brainmask", default="brainmask.mgz",
                         help="Brainmask filename within each subject's mri/ directory (default brainmask.mgz)")
    parser.add_argument("--lesion-xfm-pattern", default=None,
                         help="Path pattern (containing '{subject}') to a per-subject registration transform "
                              "(LTA or FSL matrix) mapping the lesion image to FreeSurfer anatomical space. "
                              "Required if lesion geometry does not already exactly match the subject's "
                              "Schaefer volume grid, unless --lesion-native-header is used instead.")
    parser.add_argument("--lesion-xfm-type", choices=["lta", "fsl"], default="lta",
                         help="Transform format for --lesion-xfm-pattern (default lta)")
    parser.add_argument("--lesion-native-header", action="store_true",
                         help="Assert that the lesion image, despite having a different grid than the "
                              "Schaefer volume, was derived from the exact T1 used for recon-all and is "
                              "therefore safe to resample with a header-based (--regheader) registration. "
                              "Only use this when you have verified that assumption; it is never applied "
                              "automatically.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        check_fs_environment()
    except EnvironmentError as exc:
        log.error(str(exc))
        return 1

    subjects_dir = args.subjects_dir.resolve()
    output_dir = args.output_dir.resolve()
    atlas_dir = args.atlas_dir.resolve()
    lesion_dir = args.lesion_dir.resolve()

    if not subjects_dir.is_dir():
        log.error("--subjects-dir %s is not a directory", subjects_dir)
        return 1
    if not lesion_dir.is_dir():
        log.error("--lesion-dir %s is not a directory", lesion_dir)
        return 1

    try:
        atlas_lh, atlas_rh = validate_atlas_dir(atlas_dir)
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    subjects = load_subjects(args.subjects_file)
    if not subjects:
        log.error("No subjects found in %s", args.subjects_file)
        return 1

    if args.lesion_pattern:
        pattern = args.lesion_pattern
    elif args.session:
        pattern = "ses-{session}/{subject}.nii.gz"
    else:
        pattern = "{subject}_lesion.nii.gz"

    for sub in ("intermediate", "qc", "logs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(subjects_dir)

    base_cfg = dict(
        subjects_dir=str(subjects_dir),
        output_dir=str(output_dir),
        atlas_lh=str(atlas_lh),
        atlas_rh=str(atlas_rh),
        overwrite=args.overwrite,
        lesion_threshold=args.lesion_threshold,
        brainmask_name=args.brainmask,
        lesion_xfm_pattern=args.lesion_xfm_pattern,
        lesion_xfm_type=args.lesion_xfm_type,
        lesion_native_header=args.lesion_native_header,
        env=env,
    )

    jobs = []
    for subject in subjects:
        lesion_path = resolve_lesion_path(lesion_dir, pattern, subject, args.session)
        cfg = dict(base_cfg)
        cfg["lesion_path"] = str(lesion_path)
        jobs.append((subject, cfg))

    results = []
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(process_subject, subject, cfg): subject for subject, cfg in jobs}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for subject, cfg in jobs:
            log.info("Processing %s", subject)
            results.append(process_subject(subject, cfg))

    results_by_subject = {r["subject"]: r for r in results}
    ordered_results = [results_by_subject[s] for s in subjects]

    subject_rows = [r["row"] for r in ordered_results if r["status"] == "ok"]
    long_rows = [row for r in ordered_results if r["status"] == "ok" for row in r["long_rows"]]
    parcel_rows = [row for r in ordered_results if r["status"] == "ok" for row in r["parcel_rows"]]
    qc_rows = [r["qc_row"] for r in ordered_results]
    failed_rows = [
        {"subject": r["subject"], "stage": r["stage"], "reason": r["reason"], "stderr": r.get("stderr", "")}
        for r in ordered_results if r["status"] == "failed"
    ]

    if subject_rows:
        pd.DataFrame(subject_rows).to_csv(output_dir / "Yeo7_lesion_corrected_volumes.csv", index=False)
    if long_rows:
        pd.DataFrame(long_rows).to_csv(output_dir / "Yeo7_lesion_network_long.csv", index=False)
    if parcel_rows:
        pd.DataFrame(parcel_rows).to_csv(output_dir / "Schaefer1000_lesion_overlap.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output_dir / "qc_summary.csv", index=False)
    pd.DataFrame(failed_rows, columns=["subject", "stage", "reason", "stderr"]).to_csv(
        output_dir / "failed_subjects.csv", index=False
    )

    log.info("Done: %d succeeded, %d failed (see %s)",
              len(subject_rows), len(failed_rows), output_dir / "failed_subjects.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
