from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import run
import sys

from PIL import Image

from facegan_studio.modules.gallery import collect_showcase_assets
from facegan_studio.modules.id_photo import create_id_photo_variants
from facegan_studio.modules.image_utils import make_grid, save_rgb
from facegan_studio.modules.paths import ProjectPaths


class FaceGANStudioCoreTests(unittest.TestCase):
    def test_create_run_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ProjectPaths(tmp)
            run_dir = paths.create_run_dir("preview", "20260507_000000")
            report_dir = paths.create_report_run_dir("preview", "20260507_000000")

            self.assertTrue(run_dir.exists())
            self.assertTrue(report_dir.exists())
            self.assertEqual(run_dir.name, "20260507_000000")

    def test_grid_requires_images(self) -> None:
        with self.assertRaises(ValueError):
            make_grid([])

    def test_id_photo_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.png"
            image = Image.new("RGB", (600, 800), (190, 170, 150))
            save_rgb(image, input_path)

            result = create_id_photo_variants(
                input_path,
                root / "out",
                backgrounds=["white", "blue", "red"],
                size=(413, 626),
            )

            self.assertEqual(set(result.variants), {"white", "blue", "red"})
            self.assertTrue(result.grid_path.exists())
            self.assertTrue(result.metadata_path.exists())

    def test_showcase_uses_handoff_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_dir = root / "docs" / "handoff_assets"
            assets_dir.mkdir(parents=True)
            save_rgb(Image.new("RGB", (64, 64), "white"), assets_dir / "dcgan_evolution_grid.png")

            assets = collect_showcase_assets(root)

            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0].name, "DCGAN 演化展示")

    def test_style_transfer_inputs_supplement_with_generated_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            personal_dir = root / "personal"
            generated_dir = root / "generated"
            output_dir = root / "processed"
            personal_dir.mkdir()
            generated_dir.mkdir()

            save_rgb(Image.new("RGB", (80, 80), "red"), personal_dir / "personal.png")
            save_rgb(Image.new("RGB", (80, 80), "blue"), generated_dir / "generated_0.png")
            save_rgb(Image.new("RGB", (80, 80), "green"), generated_dir / "generated_1.png")

            result = run(
                [
                    sys.executable,
                    "scripts/prepare_style_transfer_inputs.py",
                    "--personal-dir",
                    str(personal_dir),
                    "--generated-dir",
                    str(generated_dir),
                    "--output-dir",
                    str(output_dir),
                    "--max-images",
                    "3",
                    "--clear",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(output_dir.glob("style_input_*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
