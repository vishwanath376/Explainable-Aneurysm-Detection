"""Create a patient/control manifest for the Lausanne TOF-MRA cohort.

This is a subject-level classification pilot, not voxel-level detection.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def participant_labels(path: Path) -> dict[str, int]:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"participant_id", "group"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}; found {list(df.columns)}")
    labels = {}
    for _, row in df.iterrows():
        sid = row["participant_id"].strip()
        group = row["group"].strip().lower()
        if not sid:
            continue
        if "control" in group or group in {"healthy", "healthy_control"}:
            labels[sid] = 0
        elif "patient" in group:
            labels[sid] = 1
    if not labels:
        raise ValueError("No patient/control labels recognized in participants.tsv; inspect its group column.")
    return labels


def split_subjects(labels: dict[str, int]) -> dict[str, str]:
    subjects = sorted(labels)
    y = [labels[s] for s in subjects]
    try:
        train, tmp = train_test_split(subjects, test_size=0.30, random_state=SEED, stratify=y)
        tmp_y = [labels[s] for s in tmp]
        val, test = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=tmp_y)
    except ValueError:
        # Tiny pilot fallback. Explicitly warn that these estimates will be unstable.
        shuffled = subjects[:]
        random.Random(SEED).shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(0.70 * n))
        n_val = max(1, int(0.15 * n)) if n >= 4 else 0
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train+n_val]
        test = shuffled[n_train+n_val:]
        if not test and len(train) > 1:
            test.append(train.pop())
        print("WARNING: too few subjects for stratified splits; using deterministic unstratified split.")
    result = {}
    result.update({s: "train" for s in train})
    result.update({s: "val" for s in val})
    result.update({s: "test" for s in test})
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    root = args.root.resolve()
    participants = root / "participants.tsv"
    if not participants.is_file():
        raise FileNotFoundError(f"Missing {participants}. Confirm that the dataset download completed and --root is correct.")
    labels = participant_labels(participants)
    scans = sorted(root.rglob("*_angio.nii.gz"))
    if not scans:
        raise FileNotFoundError(
            f"No *_angio.nii.gz volumes found under {root}. Check the dataset layout and ensure files are real volumes, not Git-annex pointers."
        )
    rows, skipped = [], 0
    for image in scans:
        subject = next((part for part in image.parts if part.startswith("sub-")), "")
        if subject not in labels:
            skipped += 1
            continue
        session = next((part for part in image.parts if part.startswith("ses-")), "")
        study = f"{subject}_{session}" if session else subject
        # A mask is recorded only when a matching lesion mask can be found.
        masks = [
            m for m in root.rglob("*.nii.gz")
            if "desc-Lesion" in m.name and "mask" in m.name.lower()
            and subject in m.as_posix() and study in m.name
        ]
        rows.append({
            "subject_id": subject,
            "study_id": study,
            "image_path": image.resolve().relative_to(root).as_posix(),
            "mask_path": masks[0].resolve().relative_to(root).as_posix() if masks else "",
            "modality": "MRA",
            "label": labels[subject],
        })
    if not rows:
        raise ValueError("No scans matched participants.tsv labels. Inspect filenames and metadata.")
    df = pd.DataFrame(rows).drop_duplicates("image_path").reset_index(drop=True)
    subject_labels = df.groupby("subject_id")["label"].first().to_dict()
    split_map = split_subjects({str(k): int(v) for k, v in subject_labels.items()})
    df["split"] = df["subject_id"].map(split_map)
    output = args.output.resolve() if args.output else root / "manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Manifest: {output}")
    print(f"Matched scans: {len(df)}; unique subjects: {df.subject_id.nunique()}; skipped unlabeled scans: {skipped}")
    print("Subject class counts:", {int(k): int(v) for k, v in df.drop_duplicates("subject_id")["label"].value_counts().items()})
    print("Subject split counts:", df.drop_duplicates("subject_id")["split"].value_counts().to_dict())
    print(f"Scans with matching lesion masks: {(df.mask_path != '').sum()} / {len(df)}")
    print("Review the manifest before training; small-sample metrics are exploratory only.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
