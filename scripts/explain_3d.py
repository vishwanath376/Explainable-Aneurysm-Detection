"""Generate a 3D Grad-CAM heatmap for the small 3D baseline."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from train_baseline import Small3DCNN, read_volume, INPUT_SHAPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("results/checkpoints/best_model.pt"))
    ap.add_argument("--subject-id", default=None)
    ap.add_argument("--output", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    root = args.root.resolve()
    df = pd.read_csv(args.manifest).fillna("")
    rows = df[df["split"] == "test"]
    if args.subject_id and "subject_id" in rows:
        rows = rows[rows["subject_id"].astype(str) == args.subject_id]
    if rows.empty: raise ValueError("No matching test subject in manifest.")
    row = rows.iloc[0]
    image = Path(str(row["image_path"]))
    if not image.is_absolute(): image = root / image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Small3DCNN().to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    activations, gradients = {}, {}
    layer = model.features[8]
    layer.register_forward_hook(lambda _m, _i, out: activations.__setitem__("value", out))
    layer.register_full_backward_hook(lambda _m, _gi, go: gradients.__setitem__("value", go[0]))
    x = read_volume(image).unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    logit = model(x)
    logit.sum().backward()
    act, grad = activations["value"], gradients["value"]
    weights = grad.mean(dim=(2,3,4), keepdim=True)
    cam = torch.relu((weights*act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=INPUT_SHAPE, mode="trilinear", align_corners=False)[0,0].detach().cpu().numpy()
    cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
    volume = x.detach().cpu().numpy()[0,0]
    prob = float(torch.sigmoid(logit).detach().cpu().item())
    args.output.mkdir(parents=True, exist_ok=True)
    sid = str(row.get("subject_id", "test_subject")).replace("/", "_")
    np.save(args.output/f"gradcam_{sid}.npy", cam)
    z = cam.shape[0]//2
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    axes[0].imshow(volume[z], cmap="gray"); axes[0].set_title(f"Input slice; P(patient)={prob:.3f}")
    axes[1].imshow(volume[z], cmap="gray"); axes[1].imshow(cam[z], cmap="jet", alpha=.45, vmin=0, vmax=1)
    axes[1].set_title("3D Grad-CAM overlay")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); out_img=args.output/f"gradcam_{sid}.png"; fig.savefig(out_img,dpi=160); plt.close(fig)
    print(f"Subject={sid}; true label={row['label']}; P(patient)={prob:.4f}")
    print(f"Saved heatmap: {args.output/f'gradcam_{sid}.npy'}")
    print(f"Saved overlay: {out_img}")
    print("Interpretation: Grad-CAM is not proof of correct aneurysm localization.")

if __name__ == "__main__": main()
