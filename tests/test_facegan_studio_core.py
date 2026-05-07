from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import run
import sys
import os

from PIL import Image

from facegan_studio.modules.gallery import collect_showcase_assets
from facegan_studio.modules.gfpgan_postprocess import build_gfpgan_comparison_grid, select_default_gfpgan_inputs
from facegan_studio.modules.identity_accessories import draw_glasses_overlay
from facegan_studio.modules.image_utils import copy_files, make_grid, save_rgb
from facegan_studio.modules.paths import ProjectPaths
from facegan_studio.modules.pose_styler import (
    NEGATIVE_PROMPT,
    POSE_PROMPTS,
    copy_pose_report_assets,
    patch_instantid_img2img_check_inputs,
)


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

    def test_id_photo_feature_removed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        app_source = (root / "facegan_studio" / "app.py").read_text(encoding="utf-8")

        self.assertNotIn("证件照", app_source)
        self.assertNotIn("run_id_photo", app_source)
        self.assertFalse((root / "facegan_studio" / "modules" / "id_photo.py").exists())

    def test_showcase_uses_handoff_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_dir = root / "docs" / "handoff_assets"
            assets_dir.mkdir(parents=True)
            save_rgb(Image.new("RGB", (64, 64), "white"), assets_dir / "dcgan_evolution_grid.png")

            assets = collect_showcase_assets(root)

            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0].name, "DCGAN 演化展示")

    def test_gfpgan_selects_latest_representative_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest_pose = root / "outputs" / "facegan_studio" / "pose_style" / "20260507_010000"
            older_pose = root / "outputs" / "facegan_studio" / "pose_style" / "20260506_010000"
            anime = root / "outputs" / "facegan_studio" / "anime" / "20260507_010000"
            for directory in [latest_pose, older_pose, anime]:
                directory.mkdir(parents=True)
            save_rgb(Image.new("RGB", (32, 32), "red"), latest_pose / "pose_style_00.png")
            save_rgb(Image.new("RGB", (32, 32), "blue"), older_pose / "pose_style_00.png")
            save_rgb(Image.new("RGB", (32, 32), "green"), anime / "anime_grid.png")

            selected = select_default_gfpgan_inputs(root, limit=4)

            self.assertIn(latest_pose / "pose_style_00.png", selected)
            self.assertNotIn(older_pose / "pose_style_00.png", selected)
            self.assertIn(anime / "anime_grid.png", selected)

    def test_gfpgan_comparison_grid_pairs_original_and_enhanced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.png"
            enhanced = root / "enhanced.png"
            grid = root / "grid.png"
            save_rgb(Image.new("RGB", (40, 40), "red"), original)
            save_rgb(Image.new("RGB", (40, 40), "blue"), enhanced)

            out = build_gfpgan_comparison_grid([(original, enhanced)], grid, cell_size=(32, 32))

            self.assertEqual(out, grid)
            self.assertTrue(grid.exists())

    def test_gfpgan_script_requires_explicit_handoff_sync(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_gfpgan_postprocess.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--sync-handoff", source)
        self.assertIn("if args.sync_handoff", source)
        self.assertIn("non-private showcase images", source)

    def test_instantid_prompts_are_identity_locked_realistic_edits(self) -> None:
        first_keys = [key for key, _prompt in POSE_PROMPTS[:4]]

        self.assertEqual(first_keys, ["original", "black_glasses", "metal_glasses", "business"])
        self.assertTrue(all("same person" in prompt and "same face" in prompt for _key, prompt in POSE_PROMPTS[:4]))
        self.assertIn("different person", NEGATIVE_PROMPT)
        self.assertIn("anime", NEGATIVE_PROMPT)

    def test_pose_report_assets_exclude_raw_identity_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            report_dir = root / "report"
            run_dir.mkdir()
            generated = run_dir / "pose_style_00_original.png"
            grid = run_dir / "pose_style_grid.png"
            reference = run_dir / "identity_reference.png"
            pose_reference = run_dir / "pose_reference_00.png"
            metadata = run_dir / "metadata.json"
            for path in [generated, grid, reference, pose_reference]:
                save_rgb(Image.new("RGB", (32, 32), "white"), path)
            metadata.write_text("{}", encoding="utf-8")

            copied = copy_pose_report_assets([generated], grid, metadata, report_dir)

            self.assertIn(report_dir / "pose_style_00_original.png", copied)
            self.assertIn(report_dir / "pose_style_grid.png", copied)
            self.assertFalse((report_dir / "identity_reference.png").exists())
            self.assertFalse((report_dir / "pose_reference_00.png").exists())

    def test_stylegan_inversion_module_cancelled(self) -> None:
        root = Path(__file__).resolve().parents[1]
        gallery_source = (root / "facegan_studio" / "modules" / "gallery.py").read_text(encoding="utf-8")

        self.assertFalse((root / "facegan_studio" / "modules" / "stylegan_inversion.py").exists())
        self.assertFalse((root / "scripts" / "run_stylegan_inversion_edit.py").exists())
        self.assertFalse((root / "scripts" / "stylegan_fast_project.py").exists())
        self.assertNotIn("stylegan_inversion_edit", gallery_source)

    def test_instantid_identity_styles_script_has_help(self) -> None:
        result = run(
            [sys.executable, "scripts/run_instantid_identity_styles.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--identity-strength", result.stdout)
        self.assertIn("--edit-strength", result.stdout)

    def test_instantid_generation_uses_img2img_for_face_preservation(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "facegan_studio" / "modules" / "pose_styler.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("StableDiffusionXLInstantIDImg2ImgPipeline", source)
        self.assertIn("control_image=kps_image", source)
        self.assertIn("strength=current_strength", source)

    def test_instantid_img2img_check_inputs_patch_drops_obsolete_placeholders(self) -> None:
        class Pipeline:
            calls: list[tuple[int, tuple[object, ...]]] = []

            def check_inputs(self, *args):
                self.calls.append((len(args), args))
                if len(args) == 18:
                    raise TypeError("check_inputs() takes from 7 to 17 positional arguments but 19 were given")
                return "ok"

        patch_instantid_img2img_check_inputs(Pipeline)
        result = Pipeline().check_inputs(*range(18))

        self.assertEqual(result, "ok")
        self.assertEqual(Pipeline.calls[-1][0], 16)
        self.assertEqual(Pipeline.calls[-1][1], tuple(list(range(12)) + list(range(14, 18))))

    def test_instantid_personal_script_does_not_sync_private_grid_to_handoff(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_instantid_identity_styles.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("shutil.copy2(result.grid_path", source)
        self.assertIn("Skip handoff sync", source)

    def test_glasses_overlay_preserves_canvas_size(self) -> None:
        image = Image.new("RGB", (240, 240), "white")
        keypoints = [(88.0, 96.0), (152.0, 96.0), (120.0, 124.0), (98.0, 154.0), (142.0, 154.0)]

        result = draw_glasses_overlay(image, keypoints, style="black_glasses")

        self.assertEqual(result.size, image.size)
        self.assertNotEqual(result.tobytes(), image.tobytes())

    def test_copy_files_preserves_duplicate_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "first"
            second_dir = root / "second"
            dest = root / "dest"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "input_face.png"
            second = second_dir / "input_face.png"
            save_rgb(Image.new("RGB", (32, 32), "red"), first)
            save_rgb(Image.new("RGB", (32, 32), "blue"), second)

            copied = copy_files([first, second], dest)

            self.assertEqual(len(copied), 2)
            self.assertEqual(len(list(dest.glob("*.png"))), 2)
            self.assertNotEqual(copied[0], copied[1])

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

    def test_cyclegan_script_reuses_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "cyclegan_repo"
            scripts_dir = repo / "scripts"
            checkpoint_dir = repo / "checkpoints" / "style_vangogh_pretrained"
            input_dir = root / "input"
            results_dir = root / "results"
            scripts_dir.mkdir(parents=True)
            checkpoint_dir.mkdir(parents=True)
            input_dir.mkdir()

            checkpoint = checkpoint_dir / "latest_net_G.pth"
            checkpoint.write_text("existing checkpoint", encoding="utf-8")
            save_rgb(Image.new("RGB", (64, 64), "white"), input_dir / "input.png")

            download_script = scripts_dir / "download_cyclegan_model.sh"
            download_script.write_text(
                "#!/usr/bin/env bash\n"
                "echo download should not run >&2\n"
                "exit 97\n",
                encoding="utf-8",
            )
            download_script.chmod(0o755)

            test_script = repo / "test.py"
            test_script.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "results_dir = Path(sys.argv[sys.argv.index('--results_dir') + 1])\n"
                "results_dir.mkdir(parents=True, exist_ok=True)\n"
                "(results_dir / 'test_was_run.txt').write_text('ok', encoding='utf-8')\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "CYCLEGAN_REPO": str(repo),
                    "CYCLEGAN_INPUT_DIR": str(input_dir),
                    "CYCLEGAN_RESULTS_DIR": str(results_dir),
                    "CYCLEGAN_MODEL_NAME": "style_vangogh",
                    "CYCLEGAN_NUM_TEST": "1",
                    "CYCLEGAN_MIN_CHECKPOINT_BYTES": "1",
                }
            )
            result = run(
                ["bash", "scripts/run_cyclegan_pretrained_style.sh"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), "existing checkpoint")
            self.assertTrue((results_dir / "test_was_run.txt").exists())


if __name__ == "__main__":
    unittest.main()
