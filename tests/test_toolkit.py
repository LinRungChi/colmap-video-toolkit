import sqlite3
import shutil

import cv2
import numpy as np
import pytest

from colmap_video_toolkit import frames
from colmap_video_toolkit.match_quality import per_image_stats
from colmap_video_toolkit.model_io import (image_ids_to_pair_id, pair_id_to_image_ids, qvec_to_rotmat,
                                           read_camera_centers, write_images_bin)
from colmap_video_toolkit.select import select_along_path


def _video(path, n=30, blur_every=3):
    rng = np.random.default_rng(0)
    base = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(n):
        f = np.roll(base, i, axis=1)
        if i % blur_every == 0:
            f = cv2.GaussianBlur(f, (21, 21), 8)
        w.write(f)
    w.release()


def test_extract_with_step_and_sharpness_filter(tmp_path):
    _video(tmp_path / "v.mp4")
    all_frames = frames.extract_frames(tmp_path / "v.mp4", tmp_path / "all")
    assert len(all_frames) == 30
    every3 = frames.extract_frames(tmp_path / "v.mp4", tmp_path / "s3", step=3)
    assert len(every3) == 10
    sharp = frames.extract_frames(tmp_path / "v.mp4", tmp_path / "sharp", keep=frames.FrameFilter(min_sharpness=500))
    assert len(sharp) == 20  # the 10 blurred frames are rejected


def test_quality_ignores_black_mask():
    img = np.zeros((50, 50), np.uint8)
    assert frames.measure_quality(img, ignore_black=True) is None
    img[10:40, 10:40] = 200
    q = frames.measure_quality(img, ignore_black=True)
    assert q.brightness == pytest.approx(200)


def test_images_to_video_roundtrip(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in [1, 2, 10]:  # natural sort: 1, 2, 10
        cv2.imwrite(str(d / f"img_{i}.png"), np.full((40, 60, 3), i * 20, np.uint8))
    assert frames.images_to_video(d, tmp_path / "out.mp4", fps=10) == 3
    assert len(frames.extract_frames(tmp_path / "out.mp4", tmp_path / "back")) == 3


def _quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def test_qvec_matches_axis_angle():
    R = qvec_to_rotmat(_quat([0, 0, 1], np.pi / 2))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_images_bin_roundtrip_gives_camera_centres(tmp_path):
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(5, 3))
    qs = np.array([_quat(rng.normal(size=3), rng.uniform(0, np.pi)) for _ in range(5)])
    ts = np.array([-qvec_to_rotmat(q) @ c for q, c in zip(qs, centers)])
    names = [f"f{i}.jpg" for i in range(5)]
    write_images_bin(tmp_path, names, qs, ts)
    got_names, got = read_camera_centers(tmp_path)
    assert got_names == names
    assert np.allclose(got, centers)


def test_selection_is_even_along_path():
    t = np.random.default_rng(2).permutation(np.linspace(0, 10, 101))
    centers = np.c_[t, 0.3 * np.sin(t), np.zeros_like(t)]
    idx = select_along_path(centers, 11)
    assert np.allclose(np.sort(t[idx]), np.arange(0, 11), atol=0.11)
    assert len(select_along_path(centers, 500)) == 101


def test_pair_id_roundtrip():
    assert pair_id_to_image_ids(image_ids_to_pair_id(7, 3)) == (3, 7)


def test_per_image_stats_from_database(tmp_path):
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
    con.executemany("INSERT INTO images VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c"), (4, "lonely")])
    for a, b, n in [(1, 2, 100), (2, 3, 50), (1, 3, 10)]:
        con.execute("INSERT INTO two_view_geometries VALUES (?, ?, 2, ?)",
                    (image_ids_to_pair_id(a, b), n, np.zeros((n, 2), np.uint32).tobytes()))
    con.commit()
    con.close()
    rows = {r["image"]: r for r in per_image_stats(db)}
    assert rows["b"]["inliers"] == 150 and rows["b"]["neighbours"] == 2
    assert rows["lonely"]["neighbours"] == 0
    assert per_image_stats(db)[0]["image"] == "b"


@pytest.mark.skipif(shutil.which("colmap") is None, reason="COLMAP not installed")
def test_colmap_wrapper_builds_arguments(monkeypatch):
    from colmap_video_toolkit.colmap import Colmap
    calls = []
    monkeypatch.setattr("subprocess.run", lambda args, **kw: calls.append(args))
    Colmap("colmap", verbose=False).run("mapper", database_path="db", Mapper__ba_refine_focal_length=True, skip=None)
    assert calls[0][1:] == ["mapper", "--database_path", "db", "--Mapper.ba_refine_focal_length", "1"]
