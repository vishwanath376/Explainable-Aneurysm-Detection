# Explainable Modality-Aware Deep Learning for Intracranial Aneurysm Detection Using CTA and MRA

This is a research prototype, not a clinical diagnostic tool. The included starter pipeline is a **small CPU-compatible 3D patient/control classification pilot for TOF-MRA**. It is not, by itself, an aneurysm localization detector and it does not complete CTA-vs-MRA comparison or two-stage candidate verification.

## Honest status

- Research plan and literature materials: prepared.
- Dataset: must be downloaded from an authorized source by the user.
- Manifest builder: included.
- Small 3D classification baseline: included.
- 3D Grad-CAM example: included.
- Actual training metrics: not available until the user runs the code on real data.
- CTA/MRA comparison and candidate-level false-positive reduction: require additional data/annotations and experiments.

## 1. Python environment (Windows CMD)

From the repository root:

```bat
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the environment does not exist, create it with Python 3.11:

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The code defaults to CPU and does not require CUDA.

## 2. Dataset

Potential MRA pilot: [Lausanne TOF-MRA Aneurysm Cohort on OpenNeuro](https://openneuro.org/datasets/ds003949/versions/1.0.1). Read and follow the dataset's current terms and citation instructions.

A GitHub dataset clone may contain Git-annex pointer files instead of real volumes. Verify each downloaded `.nii.gz` file is a real NIfTI volume; files only a few hundred bytes are not valid scans. Do not upload raw medical scans or private signed URLs to GitHub.

Expected layout includes `participants.tsv` and subject/session folders with `*_angio.nii.gz` volumes. The exact downloaded layout must be inspected.

## 3. Build a manifest

```bat
.\.venv\Scripts\python.exe scripts\prepare_mra_manifest.py --root data\raw\openneuro_ds003949
```

The script creates `data/raw/openneuro_ds003949/manifest.csv`, labels participants as patient/control based on `participants.tsv`, and assigns subject-level train/validation/test splits. This is a **subject-level pilot classification task**, not a voxel-level aneurysm detector. Review the manifest before training.

## 4. Train a small 3D CPU baseline

```bat
.\.venv\Scripts\python.exe scripts\train_baseline.py --root data\raw\openneuro_ds003949 --manifest data\raw\openneuro_ds003949\manifest.csv --epochs 3
```

The script uses one volume per batch and small fixed-size inputs to limit memory use. CPU training may still be slow. The first run should be treated as a smoke test, not a reliable clinical experiment.

Outputs go to `results/checkpoints/`, `results/tables/`, and `results/test_metrics.json`.

## 5. Generate a Grad-CAM example

```bat
.\.venv\Scripts\python.exe scripts\explain_3d.py --root data\raw\openneuro_ds003949 --manifest data\raw\openneuro_ds003949\manifest.csv --checkpoint results\checkpoints\best_model.pt
```

The script saves a NumPy 3D heatmap and a central-slice overlay under `results/figures/`. Grad-CAM illustrates model influence; it does not prove correct aneurysm localization.

## Limitations and next steps

- The Lausanne cohort is MRA-only; it cannot support direct CTA-vs-MRA comparison.
- Patient/control labels do not equal voxel-level aneurysm masks.
- A true two-stage false-positive reduction study requires a defined candidate-generation method and valid spatial annotations.
- Very small pilot datasets produce unstable estimates. Report actual counts and results, and do not claim clinical performance.
- Split by subject, not by slice or series.
- Keep raw scans, processed volumes, checkpoints, and local results out of public GitHub.
