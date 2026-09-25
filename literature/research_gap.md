# Research Gap --- Explainable Modality-Aware Deep Learning for Intracranial Aneurysm Detection Using CTA and MRA

## Current literature status

The literature set now includes 9 reviewed papers in the working survey.
The two newly added 2026 papers materially change the novelty
assessment:

-   **Paper-11 (Wang et al., 2026)** demonstrates CTA deep learning with
    transfer learning and quantitatively evaluated Grad-CAM using
    IoU/Dice.
-   **Paper-12 (Huang et al., 2026)** demonstrates multimodal
    CTA/TOF-MRA aneurysm analysis with anatomy-guided self-supervised
    learning, domain-adaptive prompting, boundary-aware domain
    generalization, false-positive reduction, and calibration.

Therefore, the project should **not** claim that multimodal aneurysm
detection, domain adaptation, false-positive reduction, Grad-CAM, or
uncertainty/calibration is individually unprecedented.

## Strong defensible research gap

The defensible gap is the **lack of a single, reproducible and
systematically controlled evaluation framework that combines
modality-aware CTA/MRA learning, two-stage candidate verification,
small-aneurysm analysis, 3D spatial explainability, and cross-modality
evaluation under a common protocol**.

The project is therefore positioned as an **integrated methodology and
evaluation framework**, with the architectural contribution kept
lightweight enough for an M.Tech implementation.

## Core gaps

### 1. Small aneurysm detection

Small aneurysms remain difficult to detect. The CTA systematic review
reports substantially lower sensitivity for lesions below 3 mm, and the
MRA literature also shows lower sensitivity for very small lesions.

**Project response:** report sensitivity separately for `<3 mm`,
`3–7 mm`, and `>7 mm` when the actual dataset provides reliable size
annotations.

### 2. False-positive reduction

Prior systems use cascades, topology, anatomy-based post-processing, or
other refinement mechanisms.

**Project response:** use a lightweight two-stage pipeline: 1. Stage 1
--- generate candidate aneurysm regions. 2. Stage 2 --- classify/refine
candidates and suppress false positives.

The contribution is the integration of this refinement with the
modality-aware and explainability workflow, not the claim that two-stage
detection itself is new.

### 3. Explainability

Paper-11 shows that Grad-CAM can be quantitatively evaluated in CTA
aneurysm analysis. Therefore, Grad-CAM itself is not the novelty.

**Project response:** generate **3D Grad-CAM explanations for the final
verified candidates** and evaluate whether the activation region
corresponds to the aneurysm ROI when suitable annotations exist.

### 4. Cross-modality robustness

Paper-12 shows that CTA and TOF-MRA multimodal/domain-generalization
research is already active.

**Project response:** explicitly compare: - CTA-only baseline - MRA-only
baseline - modality-combined baseline - proposed modality-aware model

Use the same evaluation definitions across experiments.

### 5. Standardized evaluation

The reviewed literature varies in datasets, patient/lesion-level
reporting, metrics, false-positive definitions, external validation and
size-group reporting.

**Project response:** establish one evaluation protocol covering: -
sensitivity - specificity - precision/PPV - F1 - AUC - false positives
per scan - localization performance - size-wise sensitivity -
cross-modality performance

## Proposed novelty

### Main contribution

> **An integrated modality-aware 3D intracranial aneurysm detection
> framework that combines CTA/MRA-specific representation learning,
> lightweight two-stage candidate verification, 3D spatial
> explainability, and standardized cross-modality evaluation, with
> explicit analysis of small-aneurysm sensitivity and false-positive
> reduction.**

### Optional additional feature

Prediction calibration/uncertainty may be added as a reliability layer
after the core system is stable.

This should be described as an **optional extension**, because Paper-12
already demonstrates calibration and robustness analysis.

## Proposed system

``` text
CTA / TOF-MRA
      ↓
Modality-specific preprocessing
      ↓
3D feature extraction
      ↓
Modality-aware representation
      ↓
Stage 1: Candidate detection
      ↓
Stage 2: Candidate verification
      ↓
False-positive reduction
      ↓
Final aneurysm detection + localization
      ↓
3D Grad-CAM explanation
      ↓
Standardized evaluation
      ↓
Size-wise + cross-modality analysis
      ↓
Optional calibration / uncertainty
```

## Model comparison plan

### Model A --- CTA baseline

3D detector trained/evaluated on CTA.

### Model B --- MRA baseline

3D detector trained/evaluated on TOF-MRA.

### Model C --- Modality-combined baseline

CTA and MRA data combined without explicit modality-aware handling.

### Model D --- Proposed modality-aware model

Explicit modality information or modality-specific feature processing is
incorporated before/within a shared representation.

### Stage-2 verification

The candidate outputs of the detector are passed to a lightweight
second-stage classifier to reduce false positives.

### Explainability layer

3D Grad-CAM is generated for final predictions and, where annotations
permit, compared with expert ROIs using spatial overlap measures.

## What we should NOT claim

Do not claim that the following are individually new:

-   CTA + MRA aneurysm detection
-   two-stage aneurysm detection
-   Grad-CAM
-   false-positive reduction
-   domain adaptation
-   uncertainty/calibration
-   anatomy-aware learning
-   self-supervised learning

These are all represented in the current literature set to varying
degrees.

## Final research-gap statement

> **Although deep-learning methods for intracranial aneurysm detection
> have advanced substantially in CTA and MRA, recent studies
> increasingly address individual challenges such as multimodal
> learning, domain adaptation, false-positive reduction, and
> explainability. However, the literature remains heterogeneous in
> modality handling, candidate verification, spatial explanation,
> size-wise analysis, and evaluation protocols. This project therefore
> proposes a reproducible modality-aware 3D framework that
> systematically compares CTA-specific, MRA-specific, modality-combined,
> and modality-aware models; applies lightweight second-stage candidate
> verification; evaluates small-aneurysm sensitivity and false-positive
> rates; and provides 3D Grad-CAM explanations for detected candidates
> under a common evaluation protocol.**

## Scope control

Before implementation, verify from the actual dataset:

-   CTA and MRA study availability
-   reliable aneurysm labels
-   aneurysm-size availability
-   mapping of sizes to `<3 mm`, `3–7 mm`, `>7 mm`
-   localization/segmentation annotations
-   negative cases
-   duplicate/repeated studies
-   patient-level splitting requirements
-   sufficient data for cross-modality experiments
-   computational feasibility on the friend's laptop

## Current literature completion status

**Reviewed and available:** 9 papers.

**Still recommended before closing the literature survey:** 3 additional
MRA-focused papers already identified earlier: 1. Zhou et al. --- Joint
two-stage CNNs for intracranial aneurysm detection on 3D TOF-MRA. 2.
Claux et al. --- Two-stage regularized U-Net for TOF-MRI aneurysm
detection. 3. Faron et al. --- Deep-learning aneurysm detection from 3D
TOF-MRA compared with human readers.

After these three, stop downloading papers and move to
dataset/hardware/implementation.
