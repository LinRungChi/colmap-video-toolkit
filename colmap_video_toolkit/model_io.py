"""Minimal readers for COLMAP sparse models and databases (no pycolmap needed)."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np

MAX_IMAGE_ID = 2**31 - 1


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_camera_centers(model_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Image names and camera centres (N x 3, world frame) from ``images.bin``."""
    path = Path(model_dir) / "images.bin"
    names, centers = [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            props = struct.unpack("<idddddddi", f.read(64))
            q, t = np.array(props[1:5]), np.array(props[5:8])
            name = bytearray()
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.seek(n_pts * 24, 1)  # skip 2D points (x, y, point3D_id)
            names.append(name.decode("utf-8"))
            centers.append(-qvec_to_rotmat(q).T @ t)
    return names, np.array(centers).reshape(-1, 3)


def write_images_bin(model_dir: str | Path, names: list[str], qvecs: np.ndarray, tvecs: np.ndarray) -> None:
    """Write a minimal ``images.bin`` (no 2D points). Used by the tests."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(model_dir) / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, (name, q, t) in enumerate(zip(names, qvecs, tvecs), start=1):
            f.write(struct.pack("<idddddddi", i, *q, *t, 1))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    return pair_id // MAX_IMAGE_ID, pair_id % MAX_IMAGE_ID


def image_ids_to_pair_id(id1: int, id2: int) -> int:
    if id1 > id2:
        id1, id2 = id2, id1
    return id1 * MAX_IMAGE_ID + id2


def read_verified_matches(db_path: str | Path) -> tuple[dict[int, str], list[tuple[int, int, int]]]:
    """Image id -> name, and (id1, id2, n_inliers) for every geometrically verified pair."""
    con = sqlite3.connect(str(db_path))
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        pairs = [(*pair_id_to_image_ids(pid), rows)
                 for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0")]
    finally:
        con.close()
    return names, pairs
