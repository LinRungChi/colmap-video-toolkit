"""Frame extraction and image-quality filtering.

Blurry or badly exposed frames are the most common reason a video
reconstruction fails, so frames can be filtered by sharpness (variance of the
Laplacian) and brightness while they are extracted.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class Quality:
    brightness: float  # mean grey level, 0-255
    sharpness: float   # variance of the Laplacian


def measure_quality(img: np.ndarray, ignore_black: bool = False) -> Quality | None:
    """Brightness and sharpness of a BGR or grey image.

    With ``ignore_black`` only pixels > 0 are used, which suits images where the
    background has already been masked out. Returns None for an all-black image.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if ignore_black:
        mask = gray > 0
        if not mask.any():
            return None
        return Quality(float(gray[mask].mean()), float(lap[mask].var()))
    return Quality(float(gray.mean()), float(lap.var()))


@dataclass
class FrameFilter:
    min_sharpness: float = 0.0
    max_sharpness: float = float("inf")
    min_brightness: float = 0.0
    max_brightness: float = 255.0

    def accepts(self, q: Quality) -> bool:
        return (self.min_sharpness <= q.sharpness <= self.max_sharpness
                and self.min_brightness <= q.brightness <= self.max_brightness)


def extract_frames(video: str | Path, out_dir: str | Path, fps: float | None = None, step: int = 1,
                   keep: FrameFilter | None = None, ext: str = ".jpg") -> list[Path]:
    """Save frames from ``video`` into ``out_dir``.

    ``fps`` resamples to a target frame rate (e.g. 5 frames per second);
    otherwise every ``step``-th frame is used. Frames rejected by ``keep`` are
    skipped. Returns the written paths.
    """
    video, out_dir = Path(video), Path(out_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    if fps:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, round(src_fps / fps))
    out_dir.mkdir(parents=True, exist_ok=True)

    written, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0 and (keep is None or keep.accepts(measure_quality(frame))):
            path = out_dir / f"{video.stem}_{idx:06d}{ext}"
            cv2.imwrite(str(path), frame)
            written.append(path)
        idx += 1
    cap.release()
    return written


def extract_folder(video_dir: str | Path, out_dir: str | Path, **kwargs) -> dict[str, int]:
    """Run :func:`extract_frames` on every video in a folder, one sub-folder per video."""
    counts = {}
    for v in sorted(p for p in Path(video_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS):
        counts[v.name] = len(extract_frames(v, Path(out_dir) / v.stem, **kwargs))
    return counts


def quality_report(image_dir: str | Path, csv_path: str | Path | None = None,
                   ignore_black: bool = False) -> list[dict]:
    """Brightness / sharpness of every image in a folder, optionally written to CSV.

    Useful for choosing the thresholds of :class:`FrameFilter`.
    """
    rows = []
    for p in sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS):
        img = cv2.imread(str(p))
        if img is None:
            continue
        q = measure_quality(img, ignore_black)
        rows.append({"file": p.name,
                     "brightness": None if q is None else round(q.brightness, 3),
                     "sharpness": None if q is None else round(q.sharpness, 3)})
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "brightness", "sharpness"])
            w.writeheader()
            w.writerows(rows)
    return rows


def images_to_video(image_dir: str | Path, out_path: str | Path, fps: float = 30) -> int:
    """Write the images of a folder (natural sort order) into an mp4. Returns frame count."""
    import re

    def natural(p: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

    files = sorted((p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural)
    if not files:
        raise FileNotFoundError(f"no images in {image_dir}")
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        n += 1
    writer.release()
    return n
