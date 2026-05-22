from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facegan_studio.modules.identity_accessories import run_identity_accessory_styles
from facegan_studio.modules.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-distortion identity accessory styles on four personal photos.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--sync-handoff", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.project_root)
    result = run_identity_accessory_styles(paths.project_root, timestamp=args.timestamp, sync_handoff=args.sync_handoff)
    print(f"Identity accessory output dir: {result.output_dir}")
    print(f"Identity accessory grid: {result.grid_path}")
    print(f"Identity accessory metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
