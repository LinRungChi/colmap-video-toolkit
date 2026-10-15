"""Encode rendered frames into a video with simulated hand-shake blur on every 5th frame.

    blender -b -P examples/render_demo_scene.py -- orbit      # or: python examples/render_demo_scene.py orbit  (bpy wheel)
    python examples/make_demo_video.py orbit demo.mp4
    cvt reconstruct demo.mp4 work --fps 30 --views 40 --min-sharpness 50
    python examples/make_figures.py demo.mp4 work docs
"""
import sys
from pathlib import Path

import cv2
import numpy as np

src, dst = Path(sys.argv[1]), sys.argv[2]
files = sorted(src.glob("*.png"))
h, w = cv2.imread(str(files[0])).shape[:2]
writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
rng = np.random.default_rng(0)
for i, f in enumerate(files):
    img = cv2.imread(str(f))
    if i % 5 == 2:
        k = np.zeros((21, 21))
        k[10, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((10, 10), float(rng.uniform(0, 180)), 1), (21, 21))
        img = cv2.filter2D(img, -1, k / k.sum())
    writer.write(img)
writer.release()
print(f"{len(files)} frames -> {dst}")
