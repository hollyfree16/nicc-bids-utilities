# Yeo-7 lesion-corrected network volumes

`extract_yeo7_lesion_corrected_volumes.py` computes, for each FreeSurfer
subject that has already completed `recon-all`, the raw and lesion-corrected
volume of each Yeo-7 resting-state network, using the Schaefer2018
1000-parcel / 7-network cortical parcellation and a subject-specific lesion
segmentation.

## What "lesion-corrected" means here

The script reports, per network:

```
V_corrected(network) = V_raw(network) - V(lesion ∩ network)
```

where `V_raw(network)` is the volume of the subject-native, surface-derived,
volumetrically-resampled Schaefer/Yeo-7 cortical ribbon assigned to that
network, and `V(lesion ∩ network)` is the volume of that same network's
voxels that are also flagged as lesion (after restricting the lesion to the
FreeSurfer brainmask).

This is a **lesion-excluded network volume**: the volume of cortical ribbon
assigned to a Yeo network at the time of scanning, minus the portion
occupied by the lesion segmentation. **It is not an estimate of what the
network's volume would have been before the lesion occurred** — it makes no
assumption about tissue that may have been displaced, atrophied, or
reorganized. It is a straightforward voxel-count subtraction performed
entirely within one subject-native volumetric space.

All raw and corrected volumes come from the same volumetric Schaefer
segmentation (built with `mri_aparc2aseg`), not from `mris_anatomical_stats`
surface `GrayVol`. Those two measurements are produced by different methods
(surface integration vs. voxel counting) and are not interchangeable; mixing
them would silently introduce error, so the script never does.

## Pipeline

1. **Validate environment**: `mri_surf2surf`, `mri_aparc2aseg`, `mri_vol2vol`,
   `mri_info` must be on `PATH`; `SUBJECTS_DIR` is set to `--subjects-dir`.
2. **Validate inputs** per subject (surfaces, `aseg.mgz`, brainmask, lesion
   file). A subject that fails validation is recorded in
   `failed_subjects.csv` and the batch continues.
3. **Project the Schaefer atlas** from `fsaverage` to each subject's surface
   with `mri_surf2surf`, writing
   `SUBJECTS_DIR/<subject>/label/{lh,rh}.Schaefer2018_1000Parcels_7Networks_order.annot`.
   Skipped if the annotation already exists, unless `--overwrite` is given.
4. **Build a subject-native volumetric Schaefer segmentation** with
   `mri_aparc2aseg --ribbon`, written to
   `SUBJECTS_DIR/<subject>/mri/Schaefer2018_1000Parcels_7Networks_order.mgz`.
   This is the single reference grid for every calculation below. Standard
   `recon-all` outputs (`aparc+aseg.mgz`, `aseg.mgz`, `brainmask.mgz`, etc.)
   are never written to.
5. **Align the lesion segmentation** to that grid (see "Lesion-space
   assumptions" below), writing
   `OUTPUT_DIR/intermediate/<subject>/lesion_in_fs_space.nii.gz`.
6. **Binarize** the lesion (`value > --lesion-threshold`, default `0.5`).
7. **Restrict to the FreeSurfer brainmask** (`brainmask > 0`): all reported
   lesion volumes use this brainmask-restricted lesion, not the raw lesion
   file. Voxel counts before and after masking are both reported in
   `qc_summary.csv`.
8. **Map volumetric label IDs back to parcels/networks** using the color
   table embedded in each subject's projected annotation (see "Label-ID
   mapping" below), and validate that the expected parcels are present.
9. **Compute, per Schaefer parcel and per Yeo-7 network** (bilateral, LH,
   RH): raw volume, lesion overlap volume, percent lesioned, and corrected
   volume, plus a "touched" flag and lesion voxels outside the cortical
   Yeo networks entirely (white matter / subcortical / non-cortex lesion).

## Lesion-space assumptions

The script **never silently assumes** a lesion segmentation is aligned to
FreeSurfer anatomical space. For every subject it compares the lesion
image's shape and affine to the subject's Schaefer volume:

- **Exact match** (same shape and affine within `1e-3`): used directly, no
  resampling.
- **Not an exact match**: you must supply one of:
  - `--lesion-xfm-pattern '/path/{subject}_to_fs.lta'` (or `--lesion-xfm-type fsl`
    for an FSL-style matrix) — a per-subject registration transform, applied
    with `mri_vol2vol --interp nearest`.
  - `--lesion-native-header` — an explicit assertion that the lesion image,
    despite a different grid, was derived from the exact T1 used for
    `recon-all` and is therefore safe to resample with a header-based
    (`--regheader`) registration. This is only applied when you pass this
    flag; a warning is logged for every subject resampled this way so you
    remember to QC it visually (`qc/<subject>/lesion_brain_binary.nii.gz`
    overlaid on the T1).
  - If neither is supplied and the grids don't match, that subject **fails**
    at the "lesion registration" stage with a clear reason in
    `failed_subjects.csv`, rather than being silently misaligned (e.g. a
    lesion mask that is actually in MNI space).

All resampling of the lesion (and, if needed, the brainmask) uses
**nearest-neighbor interpolation** — never trilinear — because these are
discrete segmentations.

The brainmask is a special case: since it comes from the same `recon-all`
run as the Schaefer volume, if its grid doesn't already match (rare), it is
resampled with `--regheader` automatically, and this is logged as a QC note,
not treated as a registration failure.

## Label-ID mapping (Schaefer parcel → Yeo network)

`mri_aparc2aseg` encodes cortical labels using the standard FreeSurfer
convention: volumetric label ID = `1000 + <color-table index>` for the left
hemisphere and `2000 + <color-table index>` for the right hemisphere, where
the color-table index is the 0-based position of the parcel in the
annotation's embedded color table (the same order returned by
`nibabel.freesurfer.read_annot`). The script builds this mapping per subject
from the subject's own projected annotation (not a hardcoded table), then
parses each parcel name (e.g. `7Networks_LH_SalVentAttn_Med_1`) to recover
its Yeo-7 network from the token immediately following the hemisphere:

| Token          | Network                    |
|----------------|-----------------------------|
| `Vis`          | Visual                      |
| `SomMot`       | Somatomotor                 |
| `DorsAttn`     | DorsalAttention              |
| `SalVentAttn`  | SalienceVentralAttention     |
| `Limbic`       | Limbic                       |
| `Cont`         | Control                      |
| `Default`      | Default                      |

After building the mapping, the script checks that the expected label IDs
actually appear in the volumetric segmentation (`schaefer_parcels_found` /
`schaefer_parcels_expected` in `qc_summary.csv`); if fewer than 90% are
found, a warning is recorded (not necessarily a failure — small/absent
parcels can be legitimate in an unusual anatomy).

## Inputs

| Flag | Description |
|---|---|
| `--subjects-dir` | FreeSurfer `SUBJECTS_DIR` |
| `--subjects-file` | One subject ID per line; blank lines and `#` comments ignored |
| `--atlas-dir` | Directory with `{lh,rh}.Schaefer2018_1000Parcels_7Networks_order.annot` |
| `--lesion-dir` | Root directory of lesion segmentations |
| `--lesion-pattern` | Filename pattern with `{subject}` (and optionally `{session}`). Default: `ses-{session}/{subject}.nii.gz` if `--session` is given — matching the `<lesion-dir>/ses-<session>/<subject>.nii.gz` layout used by `freeview_qc_batch.py`'s `--model-dir` — otherwise `{subject}_lesion.nii.gz` |
| `--session` | Session label without `ses-` prefix, e.g. `001` |
| `--output-dir` | Output directory |
| `--overwrite` | Regenerate subject annotations / Schaefer volume even if present |
| `--n-jobs` | Subjects processed in parallel (default 1) |
| `--lesion-threshold` | Binarization threshold, default `0.5` |
| `--brainmask` | Brainmask filename within `mri/`, default `brainmask.mgz` |
| `--lesion-xfm-pattern` / `--lesion-xfm-type` | Per-subject registration transform for lesion alignment |
| `--lesion-native-header` | Assert lesion shares the recon-all T1's native space; use `--regheader` |

## Outputs (under `--output-dir`)

- `Yeo7_lesion_corrected_volumes.csv` — one row per subject: global lesion
  quantities plus, for each of the 7 networks, `*_raw_mm3`,
  `*_lesion_overlap_mm3`, `*_lesion_pct`, `*_corrected_mm3`, `*_touched`
  (bilateral), and `LH_*` / `RH_*` hemisphere-specific volumes.
- `Yeo7_lesion_network_long.csv` — one row per subject × hemisphere
  (`LH`/`RH`/`Bilateral`) × network; easiest format for statistics.
- `Schaefer1000_lesion_overlap.csv` — one row per subject × Schaefer parcel:
  raw/overlap/corrected volume, percent lesioned, touched flag.
- `qc_summary.csv` — one row per subject: lesion voxel counts before/after
  brainmask restriction, parcel coverage, cortical vs. total lesion volume,
  networks touched, and any warnings.
- `failed_subjects.csv` — subject, failure stage, reason, and captured
  stderr for any subject that could not be processed.
- `intermediate/<subject>/lesion_in_fs_space.nii.gz` (and
  `brainmask_in_fs_space.nii.gz` if resampling was needed).
- `qc/<subject>/` — copy of the volumetric Schaefer segmentation, the
  brainmask-restricted binary lesion, a 7-valued network-ID volume, and a
  lesion/network overlap volume (network ID where lesion overlaps that
  network, 0 elsewhere) for inspection in `freeview`.
- `logs/<subject>.log` — captured stdout/stderr of every FreeSurfer command.

Network ID encoding used in the QC label volumes: `1=Visual, 2=Somatomotor,
3=DorsalAttention, 4=SalienceVentralAttention, 5=Limbic, 6=Control,
7=Default`.

## Example

```bash
python extract_yeo7_lesion_corrected_volumes.py \
    --subjects-dir /path/to/derivatives/freesurfer_reconall_v8.2.0/ses-001 \
    --subjects-file subjects.txt \
    --atlas-dir /path/to/Schaefer2018/FreeSurfer5.3/fsaverage/label \
    --lesion-dir /path/to/lesions \
    --session 001 \
    --output-dir ./yeo7_lesion_results \
    --n-jobs 4
```

This resolves each subject's lesion file at
`/path/to/lesions/ses-001/<subject>.nii.gz`, matching the directory layout
`freeview_qc_batch.py` uses for its `--model-dir` label volumes. To use a
flat, non-session directory with a different naming convention instead:

```bash
python extract_yeo7_lesion_corrected_volumes.py \
    --subjects-dir /path/to/subjects \
    --subjects-file subjects.txt \
    --atlas-dir /path/to/Schaefer2018/FreeSurfer5.3/fsaverage/label \
    --lesion-dir /path/to/lesions \
    --lesion-pattern '{subject}_lesion.nii.gz' \
    --output-dir ./yeo7_lesion_results
```

If a subject's lesion segmentation isn't already aligned to FreeSurfer
space, add `--lesion-xfm-pattern '/path/to/xfms/{subject}_lesion2fs.lta'`
(or `--lesion-native-header` if you've verified the lesion shares the
recon-all T1's native space) — otherwise that subject will fail at the
lesion-registration stage rather than being silently mis-registered.

## Requirements

- FreeSurfer (`mri_surf2surf`, `mri_aparc2aseg`, `mri_vol2vol`, `mri_info`
  on `PATH`, environment sourced)
- Python: `numpy`, `pandas`, `nibabel`
