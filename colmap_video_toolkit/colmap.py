"""Thin wrapper around the COLMAP command-line tool."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .model_io import read_camera_centers


class Colmap:
    def __init__(self, binary: str = "colmap", verbose: bool = True):
        resolved = shutil.which(binary) or (binary if Path(binary).exists() else None)
        if resolved is None:
            raise FileNotFoundError(f"COLMAP not found: {binary!r}. Install it or pass --colmap /path/to/colmap")
        self.binary = resolved
        self.verbose = verbose

    def run(self, command: str, **options) -> None:
        """``run("mapper", database_path=..., Mapper__ba_refine_principal_point=1)``.

        Keyword names map to COLMAP options: ``__`` becomes ``.``.
        """
        args = [self.binary, command]
        for key, value in options.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = int(value)
            args += [f"--{key.replace('__', '.')}", str(value)]
        if self.verbose:
            print("$", " ".join(args), flush=True)
        subprocess.run(args, check=True, stdout=None if self.verbose else subprocess.DEVNULL)

    def sparse(self, images: Path, workspace: Path, fast: bool = False, matcher: str = "sequential",
               camera_model: str = "SIMPLE_RADIAL", use_gpu: bool | None = None) -> Path:
        """Features, matching and mapping. Returns the folder of the largest model (``sparse/0``).

        ``fast`` uses fewer features and a smaller matching window; it is meant
        for a first pass whose only purpose is to recover camera positions.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        db = workspace / "database.db"
        if db.exists():
            db.unlink()
        sparse = workspace / "sparse"
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir()

        self.run("feature_extractor", database_path=db, image_path=images,
                 ImageReader__camera_model=camera_model, ImageReader__single_camera=1,
                 SiftExtraction__max_num_features=2048 if fast else 8192,
                 SiftExtraction__use_gpu=use_gpu)
        if matcher == "sequential":
            self.run("sequential_matcher", database_path=db, SequentialMatching__overlap=20,
                     SiftMatching__use_gpu=use_gpu)
        else:
            self.run("exhaustive_matcher", database_path=db, SiftMatching__use_gpu=use_gpu)
        # Neighbouring video frames have small baselines; COLMAP's default 16 degree
        # minimum for the initial pair often rejects every pair in the matching window.
        self.run("mapper", database_path=db, image_path=images, output_path=sparse,
                 Mapper__init_min_tri_angle=4 if matcher == "sequential" else None)
        models = [p for p in sparse.iterdir() if (p / "images.bin").exists()]
        if not models:
            raise RuntimeError("COLMAP mapper produced no model: too few matches between frames")
        best = max(models, key=lambda p: len(read_camera_centers(p)[0]))
        if not fast:
            self.run("bundle_adjuster", input_path=best, output_path=best,
                     BundleAdjustment__refine_principal_point=1)
        return best

    def undistort(self, images: Path, model: Path, out: Path, max_size: int = 2000) -> Path:
        """Undistorted ``images/`` + ``sparse/`` layout expected by 3D Gaussian Splatting and dense MVS."""
        self.run("image_undistorter", image_path=images, input_path=model, output_path=out,
                 output_type="COLMAP", max_image_size=max_size)
        return out

    def dense(self, undistorted: Path, out_ply: Path, max_size: int = 2000) -> Path:
        """PatchMatch stereo + fusion (needs a CUDA build of COLMAP)."""
        self.run("patch_match_stereo", workspace_path=undistorted, workspace_format="COLMAP",
                 PatchMatchStereo__max_image_size=max_size, PatchMatchStereo__geom_consistency=1)
        self.run("stereo_fusion", workspace_path=undistorted, workspace_format="COLMAP",
                 input_type="geometric", output_path=out_ply)
        return out_ply
