# Explainable Modality-Aware Deep Learning for Intracranial Aneurysm Detection Using CTA and MRA

Research codebase for an explainable, modality-aware intracranial aneurysm imaging project.

## Current implementation status

- A compact 3D CNN baseline and preprocessing pipeline are available for **MRA patient/control classification**.
- 3D Grad-CAM utilities are included for exploratory model interpretation. Grad-CAM is not validated aneurysm localization.
- Dataset validation, manifest preparation, geometry audits, evaluation, and subject-level evaluation utilities are included.
- A CTA–MRA fusion training script and paired-data manifest template are present, but **no real paired CTA–MRA cohort has been validated**. Multimodal fusion is therefore not presented as a completed or validated result.
- The Streamlit demo entry point is currently a placeholder.

This is a research prototype, not a clinical diagnostic device. Existing baseline results are weak and must not be interpreted as clinical performance. Reproduce and independently verify all results before citing them.

## Project structure

- `scripts/` — MRA preprocessing, training, evaluation, explainability, and audit scripts
- `training/`, `inference/`, `dataset_creation/` — supporting training, inference, and patch-generation code
- `show_results/` — evaluation and visualization utilities
- `data/smoke_test/` — synthetic smoke-test data only
- `data/templates/` — paired CTA–MRA manifest template
- `literature/research_gap.md` — research-gap notes

## Environment

Python 3.10–3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Check individual script arguments with `python scripts/<script_name>.py --help`.

## Data policy

Clinical scans, patient-level manifests, subject identifiers, private metadata, model checkpoints, and derived patient-level prediction files are intentionally excluded from this public repository. Obtain datasets from their authorized source and follow applicable data-use terms. Do not commit protected health information or restricted research data.

## Research limitations

The current workflow is MRA patient/control classification, not lesion-level aneurysm detection. CTA-only modelling, paired multimodal fusion, lesion-localization validation, external validation, and clinical evaluation remain future work.

## Citation and licensing

See `LICENSE` and the dataset's own license/terms. Cite the underlying dataset and relevant publications when using them. This repository does not grant rights to third-party datasets or publications.
