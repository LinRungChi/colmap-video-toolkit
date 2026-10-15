"""Video -> frames -> coarse SfM -> view selection -> fine SfM -> undistorted dataset.

Two-pass strategy: a cheap first reconstruction on many frames only serves to
find where the camera was; a fixed number of evenly spaced frames is then
reconstructed properly. The output folder (``images/`` + ``sparse/0``) can be
opened directly by 3D Gaussian Splatting trainers or Nerfstudio.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .colmap import Colmap
from .frames import FrameFilter, extract_frames
from .model_io import read_camera_centers
from .select import copy_selected, select_along_path


@dataclass
class PipelineConfig:
    fps: float = 5.0
    n_views: int = 100
    min_sharpness: float = 0.0
    camera_model: str = "SIMPLE_RADIAL"
    matcher: str = "sequential"        # coarse pass: frames are in video order
    fine_matcher: str = "exhaustive"   # fine pass: the selected set is small, so match every pair
    max_image_size: int = 2000
    dense: bool = False
    use_gpu: bool | None = None


def run(video: str | Path, out_dir: str | Path, colmap: Colmap, cfg: PipelineConfig) -> dict:
    video, out = Path(video), Path(out_dir)
    t0 = time.time()
    report: dict = {"video": video.name, "config": asdict(cfg)}

    frames_dir = out / "frames"
    frames = extract_frames(video, frames_dir, fps=cfg.fps, keep=FrameFilter(min_sharpness=cfg.min_sharpness))
    report["frames_extracted"] = len(frames)
    print(f"[1/4] {len(frames)} frames extracted")

    coarse = colmap.sparse(frames_dir, out / "coarse", fast=True, matcher=cfg.matcher,
                           camera_model=cfg.camera_model, use_gpu=cfg.use_gpu)
    names, centers = read_camera_centers(coarse)
    report["coarse_registered"] = len(names)
    print(f"[2/4] coarse model registered {len(names)}/{len(frames)} frames")

    idx = select_along_path(centers, cfg.n_views)
    selected_dir = out / "selected"
    report["selected"] = copy_selected(names, idx, frames_dir, selected_dir)
    print(f"[3/4] selected {report['selected']} frames along the camera path")

    fine = colmap.sparse(selected_dir, out / "fine", fast=False, matcher=cfg.fine_matcher,
                         camera_model=cfg.camera_model, use_gpu=cfg.use_gpu)
    report["fine_registered"] = len(read_camera_centers(fine)[0])
    dataset = colmap.undistort(selected_dir, fine, out / "dataset", cfg.max_image_size)
    print(f"[4/4] fine model registered {report['fine_registered']}/{report['selected']}; dataset in {dataset}")

    if cfg.dense:
        report["dense_ply"] = str(colmap.dense(dataset, out / "fused.ply", cfg.max_image_size))

    report["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
