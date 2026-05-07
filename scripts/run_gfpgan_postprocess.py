from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facegan_studio.modules.gfpgan_postprocess import (
    run_gfpgan_postprocess,
    select_default_gfpgan_inputs,
    sync_gfpgan_report_assets,
)
from facegan_studio.modules.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GFPGAN as a post-processing enhancer for generated portraits.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input", action="append", default=[], help="Input image path. Can be repeated.")
    parser.add_argument("--input-dir", action="append", default=[], help="Directory containing images.")
    parser.add_argument("--default-inputs", action="store_true", help="Use recent FaceGAN Studio outputs and showcase assets.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Copy inputs without GFPGAN; useful for tests.")
    parser.add_argument("--sync-report", action="store_true", help="Copy result grid and metadata to report assets.")
    parser.add_argument(
        "--sync-handoff",
        action="store_true",
        help="Copy the comparison grid to docs/handoff_assets. Use only for non-private showcase images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.project_root)
    output_dir = Path(args.output_dir) if args.output_dir else paths.create_run_dir("gfpgan_postprocess")

    inputs = [Path(path) for path in args.input]
    for directory in args.input_dir:
        inputs.extend(sorted(Path(directory).rglob("*.png")))
        inputs.extend(sorted(Path(directory).rglob("*.jpg")))
        inputs.extend(sorted(Path(directory).rglob("*.jpeg")))
    if args.default_inputs:
        inputs.extend(select_default_gfpgan_inputs(paths.project_root, limit=args.limit))

    result = run_gfpgan_postprocess(
        inputs[: args.limit],
        output_dir,
        model_path=args.model_path,
        upscale=args.upscale,
        dry_run=args.dry_run,
    )
    report_dir = sync_gfpgan_report_assets(paths.project_root, result) if args.sync_report else None

    handoff_grid = None
    if args.sync_handoff:
        handoff_grid = paths.project_root / "docs" / "handoff_assets" / "gfpgan_comparison_grid.png"
        shutil.copy2(result.comparison_grid, handoff_grid)

    print(f"GFPGAN output dir: {result.output_dir}")
    print(f"GFPGAN comparison grid: {result.comparison_grid}")
    if handoff_grid:
        print(f"Stable handoff grid: {handoff_grid}")
    else:
        print("Stable handoff grid: skipped; pass --sync-handoff only for non-private showcase images.")
    if report_dir:
        print(f"Report assets: {report_dir}")


if __name__ == "__main__":
    main()
