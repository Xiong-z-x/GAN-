from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facegan_studio.modules.image_utils import collect_images
from facegan_studio.modules.paths import ProjectPaths
from facegan_studio.modules.pose_styler import generate_pose_styles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InstantID identity-preserving realistic portrait edits for the personal photo set."
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input", default=None, help="Identity source image. Defaults to the first image in data/raw/my_photos.")
    parser.add_argument("--count", type=int, default=4, help="Number of lightweight realistic edits to generate.")
    parser.add_argument("--identity-strength", choices=["standard", "strong"], default="strong")
    parser.add_argument("--edit-strength", type=float, default=0.35, help="Img2img denoise strength for non-original edits.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--sync-handoff", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.project_root)
    input_path = Path(args.input) if args.input else None
    if input_path is None:
        photos = collect_images(paths.project_root / "data" / "raw" / "my_photos", recursive=True, limit=4)
        if not photos:
            raise FileNotFoundError("未在 data/raw/my_photos/ 找到本人照片。")
        input_path = photos[0]

    result = generate_pose_styles(
        input_path,
        project_root=paths.project_root,
        count=args.count,
        identity_strength=args.identity_strength,
        timestamp=args.timestamp,
        edit_strength=args.edit_strength,
    )

    if args.sync_handoff:
        print("Skip handoff sync: InstantID personal-result grids may contain private face images.")

    print(f"InstantID output dir: {result.output_dir}")
    print(f"InstantID grid: {result.grid_path}")
    print(f"InstantID metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
