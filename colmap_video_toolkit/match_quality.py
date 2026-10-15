"""Diagnose a COLMAP database: which frames match well and which are isolated."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np

from .model_io import image_ids_to_pair_id, read_verified_matches


def per_image_stats(db_path: str | Path) -> list[dict]:
    """For each image: total verified inliers, number of matched neighbours, mean inliers per neighbour.

    Sorted from best to worst. Images with zero neighbours are isolated and
    will not be registered by the mapper.
    """
    names, pairs = read_verified_matches(db_path)
    total = {i: 0 for i in names}
    neigh = {i: 0 for i in names}
    for a, b, n in pairs:
        for i in (a, b):
            if i in total:
                total[i] += n
                neigh[i] += 1
    rows = [{"image": names[i], "inliers": total[i], "neighbours": neigh[i],
             "mean_inliers": round(total[i] / neigh[i], 1) if neigh[i] else 0.0} for i in names]
    return sorted(rows, key=lambda r: -r["inliers"])


def draw_pair(db_path: str | Path, image_dir: str | Path, id1: int, id2: int) -> np.ndarray:
    """Side-by-side image of the verified matches between two images."""
    con = sqlite3.connect(str(db_path))
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))

        def keypoints(i):
            data, cols = con.execute("SELECT data, cols FROM keypoints WHERE image_id=?", (i,)).fetchone()
            return np.frombuffer(data, np.float32).reshape(-1, cols)[:, :2]

        k1, k2 = keypoints(id1), keypoints(id2)
        row = con.execute("SELECT data FROM two_view_geometries WHERE pair_id=?",
                          (image_ids_to_pair_id(id1, id2),)).fetchone()
    finally:
        con.close()
    m = np.frombuffer(row[0], np.uint32).reshape(-1, 2)
    if id1 > id2:  # COLMAP stores matches in (smaller id, larger id) order
        m = m[:, ::-1]
    img1 = cv2.imread(str(Path(image_dir) / names[id1]))
    img2 = cv2.imread(str(Path(image_dir) / names[id2]))
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in k1]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in k2]
    dm = [cv2.DMatch(int(a), int(b), 0) for a, b in m]
    return cv2.drawMatches(img1, kp1, img2, kp2, dm, None, matchColor=(0, 255, 0),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def plot_inliers(rows: list[dict], out_png: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = [r["inliers"] for r in sorted(rows, key=lambda r: r["image"])]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    mean = float(np.mean(vals)) if vals else 0.0
    ax.bar(range(len(vals)), vals, color=["#4C72B0" if v >= mean else "#C44E52" for v in vals], width=1.0)
    ax.axhline(mean, color="black", ls="--", lw=1, label=f"mean {mean:.0f}")
    ax.set_xlabel("frame (in file order)")
    ax.set_ylabel("verified inliers")
    ax.set_title("Match strength per frame (red = below mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
