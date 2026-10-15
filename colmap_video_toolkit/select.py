"""Pick a well-spread subset of frames along the camera path.

A video gives hundreds of nearly identical neighbouring frames. Reconstruction
is faster and often better with fewer, evenly spaced views, so the camera
centres from a quick coarse reconstruction are projected onto the main axis of
the path (first principal component) and sampled at equal steps.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def order_along_path(centers: np.ndarray) -> np.ndarray:
    """Indices that sort camera centres along their main direction of travel."""
    c = centers - centers.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return np.argsort(c @ vt[0])


def select_along_path(centers: np.ndarray, k: int) -> np.ndarray:
    """Indices of ``k`` cameras evenly spaced along the path (all of them if k >= N)."""
    n = len(centers)
    order = order_along_path(centers)
    if k >= n:
        return order
    return order[np.linspace(0, n - 1, k).round().astype(int)]


def copy_selected(names: list[str], indices: np.ndarray, src_dir: str | Path, dst_dir: str | Path) -> int:
    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = 0
    for i in indices:
        src = Path(src_dir) / names[i]
        if src.exists():
            (dst / names[i]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / names[i])
            n += 1
    return n
