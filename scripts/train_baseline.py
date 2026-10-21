"""Train a small 3D patient/control classifier on real NIfTI volumes.

Required manifest columns: image_path,label,split; optional subject_id.
This is a pilot classification model, not a clinical aneurysm detector.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

SEED = 42
INPUT_SHAPE = (24, 32, 32)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def read_volume(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    if path.stat().st_size < 1024:
        raise ValueError(f"{path} is only {path.stat().st_size} bytes; likely a Git-annex pointer, not a real NIfTI volume.")
    vol = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {vol.shape}: {path}")
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi) if hi > lo else vol
    vol = (vol - float(vol.mean())) / (float(vol.std()) + 1e-6)
    x = torch.from_numpy(vol.copy())[None, None]
    # Fixed output shape avoids DataLoader errors when source volumes differ in size.
    return F.interpolate(x, size=INPUT_SHAPE, mode="trilinear", align_corners=False).squeeze(0)


class VolumeDataset(Dataset):
    def __init__(self, df, root):
        self.df = df.reset_index(drop=True); self.root = root
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = Path(str(row.image_path))
        if not path.is_absolute(): path = self.root / path
        return read_volume(path), torch.tensor(float(row.label)), str(row.get("subject_id", i))


class Small3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1, bias=False), nn.BatchNorm3d(8), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1, bias=False), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 24, 3, padding=1, bias=False), nn.BatchNorm3d(24), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(24, 1))
    def forward(self, x): return self.classifier(self.pool(self.features(x))).squeeze(1)


def evaluate(model, loader, device, criterion=None, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    ys, probs, losses = [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        if train_mode: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(y.detach().cpu().numpy().astype(int).tolist())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return (float(np.mean(losses)) if losses else None), ys, probs


def calculate_metrics(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
        "sensitivity": float(tp/(tp+fn)) if tp+fn else None,
        "specificity": float(tn/(tn+fp)) if tn+fp else None,
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else None,
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else None,
        "roc_auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.epochs < 1: raise ValueError("--epochs must be >= 1")
    if args.batch_size != 1:
        print("Note: batch size 1 is recommended for low-memory CPU training.")
    set_seed(args.seed)
    root = args.root.resolve(); manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}. Run scripts/prepare_mra_manifest.py first.")
    df = pd.read_csv(manifest).fillna("")
    required = {"image_path", "label", "split"}
    if not required.issubset(df.columns): raise ValueError(f"Manifest requires columns: {sorted(required)}")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df.label.unique()).issubset({0,1}): raise ValueError("label must be 0=control or 1=patient")
    if "subject_id" in df:
        ids = {s: set(df.loc[df.split == s, "subject_id"].astype(str)) for s in ("train","val","test")}
        if (ids["train"] & ids["val"]) or (ids["train"] & ids["test"]) or (ids["val"] & ids["test"]):
            raise ValueError("Patient leakage detected: subject appears in more than one split.")
    train_df, val_df, test_df = [df[df.split == s].copy() for s in ("train","val","test")]
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(f"Need non-empty train/val/test sets; got train={len(train_df)}, val={len(val_df)}, test={len(test_df)}. Dataset may be too small.")
    train_loader = DataLoader(VolumeDataset(train_df, root), batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(VolumeDataset(val_df, root), batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(VolumeDataset(test_df, root), batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Small3DCNN().to(device)
    npos = max(1, int(train_df.label.sum())); nneg = max(1, int(len(train_df)-train_df.label.sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nneg/npos], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = args.output; ckpt_dir = out / "checkpoints"; table_dir = out / "tables"
    ckpt_dir.mkdir(parents=True, exist_ok=True); table_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); history = []
    for epoch in range(1, args.epochs+1):
        tl, _, _ = evaluate(model, train_loader, device, criterion, optimizer)
        vl, vy, vp = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl})
        print(f"Epoch {epoch}/{args.epochs}: train_loss={tl:.4f}, val_loss={vl:.4f}")
        if vl is not None and vl < best_val:
            best_val = vl
            torch.save({"model_state_dict": model.state_dict(), "model_name":"Small3DCNN",
                        "input_shape":INPUT_SHAPE, "seed":args.seed, "epoch":epoch}, ckpt_dir/"best_model.pt")
    pd.DataFrame(history).to_csv(table_dir/"training_history.csv", index=False)
    checkpoint_path = ckpt_dir/"best_model.pt"
    if not checkpoint_path.exists(): raise RuntimeError("No checkpoint saved.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, ty, tp = evaluate(model, test_loader, device, criterion)
    report = {"task":"MRA patient/control pilot classification", "device":str(device), "test_loss":test_loss,
              "test_metrics":calculate_metrics(ty,tp), "warning":"Exploratory pilot only; not clinical performance."}
    (out/"test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pred_df = test_df.reset_index(drop=True).copy()
    pred_df["probability_patient"] = tp; pred_df["predicted_label"] = (np.asarray(tp)>=0.5).astype(int)
    pred_df.to_csv(table_dir/"test_predictions.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Metrics: {(out/'test_metrics.json').resolve()}")

if __name__ == "__main__": main()
