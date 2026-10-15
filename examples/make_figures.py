"""Figures for the README from a finished ``cvt reconstruct`` run.

    python examples/make_figures.py demo.mp4 work docs
"""
import struct
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colmap_video_toolkit.frames import measure_quality
from colmap_video_toolkit.model_io import read_camera_centers

video, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(exist_ok=True)
thr = 50.0


def read_points(model):
    pts = []
    with open(model / "points3D.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            vals = struct.unpack("<QdddBBBd", f.read(43))
            (tl,) = struct.unpack("<Q", f.read(8))
            f.seek(tl * 8, 1)
            pts.append(vals[1:7])
    return np.array(pts)


# 1. sharpness per frame
cap = cv2.VideoCapture(str(video))
sharp = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    sharp.append(measure_quality(f).sharpness)
sharp = np.array(sharp)
fig, ax = plt.subplots(figsize=(9, 2.6))
keep = sharp >= thr
ax.bar(np.flatnonzero(keep), sharp[keep], width=1, color="#4C72B0", label="kept")
ax.bar(np.flatnonzero(~keep), sharp[~keep], width=1, color="#C44E52", label="rejected (motion blur)")
ax.axhline(thr, color="k", ls="--", lw=1)
ax.set_xlabel("frame")
ax.set_ylabel("sharpness (var. of Laplacian)")
ax.set_ylim(0, sharp.max() * 1.4)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_title(f"Frame filter: {keep.sum()} of {len(sharp)} frames kept")
fig.tight_layout()
fig.savefig(out / "sharpness.png", dpi=150)
plt.close(fig)

# 2. camera path: coarse cameras, selected views, sparse points (top view)
coarse = work / "coarse" / "sparse"
coarse_model = max((p for p in coarse.iterdir() if (p / "images.bin").exists()), key=lambda p: len(read_camera_centers(p)[0]))
c_names, c_centers = read_camera_centers(coarse_model)
fine_model = max((p for p in (work / "fine" / "sparse").iterdir() if (p / "images.bin").exists()),
                 key=lambda p: len(read_camera_centers(p)[0]))
selected = {p.name for p in (work / "selected").iterdir()}
pts = read_points(coarse_model)
sel = np.array([n in selected for n in c_names])

# rotate into the ground plane: PCA of the points, normal = least-variance axis
centred = pts[:, :3] - pts[:, :3].mean(0)
_, _, vt = np.linalg.svd(centred, full_matrices=False)
proj = lambda x: (x - pts[:, :3].mean(0)) @ vt[:2].T
P, C = proj(pts[:, :3]), proj(c_centers)
lim = np.percentile(np.abs(P), 97, axis=0) * 1.3
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(P[:, 0], P[:, 1], s=0.3, c=pts[:, 3:6] / 255, alpha=0.6)
ax.plot(C[:, 0], C[:, 1], ".", color="0.6", ms=4, label=f"coarse cameras ({len(C)})")
ax.plot(C[sel, 0], C[sel, 1], "o", color="#C44E52", ms=5, label=f"selected views ({sel.sum()})")
ax.set_aspect("equal")
ax.legend(loc="lower left", fontsize=8)
ax.set_title(f"Top view: {len(pts)} sparse points, evenly spaced views")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(out / "camera_path.png", dpi=150)
plt.close(fig)
print("figures written to", out)
