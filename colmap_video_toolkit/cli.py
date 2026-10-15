"""Command-line entry point: ``cvt <command> ...``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import frames, match_quality
from .colmap import Colmap
from .pipeline import PipelineConfig, run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cvt", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("frames", help="extract (and filter) frames from a video or a folder of videos")
    a.add_argument("video", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--fps", type=float)
    a.add_argument("--step", type=int, default=1)
    a.add_argument("--min-sharpness", type=float, default=0.0)
    a.add_argument("--min-brightness", type=float, default=0.0)
    a.add_argument("--max-brightness", type=float, default=255.0)

    q = sub.add_parser("quality", help="brightness/sharpness of every image in a folder -> CSV")
    q.add_argument("images", type=Path)
    q.add_argument("--csv", type=Path, default=Path("quality.csv"))
    q.add_argument("--ignore-black", action="store_true", help="ignore masked (black) pixels")

    r = sub.add_parser("reconstruct", help="video -> frames -> coarse SfM -> view selection -> fine SfM")
    r.add_argument("video", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--fps", type=float, default=5.0)
    r.add_argument("--views", type=int, default=100)
    r.add_argument("--min-sharpness", type=float, default=0.0)
    r.add_argument("--camera-model", default="SIMPLE_RADIAL")
    r.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential", help="coarse pass")
    r.add_argument("--fine-matcher", choices=["sequential", "exhaustive"], default="exhaustive")
    r.add_argument("--dense", action="store_true", help="also run PatchMatch stereo (CUDA COLMAP)")
    r.add_argument("--cpu", action="store_true", help="disable GPU SIFT")
    r.add_argument("--colmap", default="colmap", help="COLMAP executable (e.g. COLMAP.bat on Windows)")

    m = sub.add_parser("matches", help="per-frame match statistics of a COLMAP database")
    m.add_argument("database", type=Path)
    m.add_argument("--images", type=Path, help="image folder, to draw the best and worst pairs")
    m.add_argument("--out", type=Path, default=Path("match_report"))

    v = sub.add_parser("video", help="images in a folder -> mp4")
    v.add_argument("images", type=Path)
    v.add_argument("out", type=Path)
    v.add_argument("--fps", type=float, default=30)

    args = p.parse_args(argv)

    if args.cmd == "frames":
        keep = frames.FrameFilter(min_sharpness=args.min_sharpness, min_brightness=args.min_brightness,
                                  max_brightness=args.max_brightness)
        if args.video.is_dir():
            print(json.dumps(frames.extract_folder(args.video, args.out, fps=args.fps, step=args.step, keep=keep), indent=1))
        else:
            print(len(frames.extract_frames(args.video, args.out, fps=args.fps, step=args.step, keep=keep)), "frames")

    elif args.cmd == "quality":
        rows = frames.quality_report(args.images, args.csv, args.ignore_black)
        print(f"{len(rows)} images -> {args.csv}")

    elif args.cmd == "reconstruct":
        cfg = PipelineConfig(fps=args.fps, n_views=args.views, min_sharpness=args.min_sharpness,
                             camera_model=args.camera_model, matcher=args.matcher, fine_matcher=args.fine_matcher,
                             dense=args.dense,
                             use_gpu=False if args.cpu else None)
        print(json.dumps(run(args.video, args.out, Colmap(args.colmap), cfg), indent=1))

    elif args.cmd == "matches":
        import csv

        import cv2

        from .model_io import read_verified_matches
        args.out.mkdir(parents=True, exist_ok=True)
        rows = match_quality.per_image_stats(args.database)
        with open(args.out / "per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        match_quality.plot_inliers(rows, args.out / "inliers.png")
        isolated = [r["image"] for r in rows if r["neighbours"] == 0]
        print(f"{len(rows)} images, {len(isolated)} isolated; best: {rows[0]['image']}  worst: {rows[-1]['image']}")
        if args.images:
            _, pairs = read_verified_matches(args.database)
            pairs.sort(key=lambda t: -t[2])
            for tag, (a, b, n) in (("best", pairs[0]), ("weakest", pairs[-1])):
                cv2.imwrite(str(args.out / f"pair_{tag}_{n}.jpg"), match_quality.draw_pair(args.database, args.images, a, b))

    elif args.cmd == "video":
        print(frames.images_to_video(args.images, args.out, args.fps), "frames written")


if __name__ == "__main__":
    main()
