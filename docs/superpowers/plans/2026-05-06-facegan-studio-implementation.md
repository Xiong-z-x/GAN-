# FaceGAN Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Gradio Web application that wraps the existing GAN course artifacts and provides upload-based face stylization, ID photo generation, and identity-preserving pose/style generation.

**Architecture:** Add a new `facegan_studio` Python package with small modules for paths, image utilities, gallery reading, ID-photo processing, AnimeGAN/CycleGAN orchestration, InstantID orchestration, and the Gradio app. Heavy model imports happen inside runtime functions so the app and unit tests can load on local machines without AutoDL model dependencies.

**Tech Stack:** Python 3, Pillow, optional OpenCV, optional Gradio, existing AnimeGANv2/CycleGAN/InstantID external repositories.

---

### Task 1: Tests for Core Utilities

**Files:**
- Create: `tests/test_facegan_studio_core.py`

- [x] Write tests for path creation, grid generation, ID photo output variants, and showcase asset discovery.
- [x] Use stdlib `unittest` so local verification does not require adding `pytest` to AutoDL dependencies.

### Task 2: Core Package

**Files:**
- Create: `facegan_studio/__init__.py`
- Create: `facegan_studio/config.py`
- Create: `facegan_studio/modules/__init__.py`
- Create: `facegan_studio/modules/paths.py`
- Create: `facegan_studio/modules/image_utils.py`
- Create: `facegan_studio/modules/id_photo.py`
- Create: `facegan_studio/modules/gallery.py`

- [x] Implement path management with non-destructive output directories.
- [x] Implement image loading, saving, and grid composition.
- [x] Implement conservative ID photo variants using face-aware crop and solid backgrounds.
- [x] Implement existing artifact discovery for the showcase tab.
- [x] Run core tests and verify pass.

### Task 3: Runtime Model Wrappers

**Files:**
- Create: `facegan_studio/modules/face_detector.py`
- Create: `facegan_studio/modules/anime_style.py`
- Create: `facegan_studio/modules/pose_styler.py`

- [x] Implement optional OpenCV face detection wrapper with safe fallback.
- [x] Implement AnimeGANv2 and optional CycleGAN subprocess orchestration.
- [x] Implement InstantID wrapper using local paths and lazy imports.
- [x] Ensure missing models produce clear errors.

### Task 4: Web App

**Files:**
- Create: `facegan_studio/app.py`

- [x] Build Gradio UI with upload, three generation modes, and project showcase tab.
- [x] Add CLI arguments for host, port, project root, and share.
- [x] Ensure importing `facegan_studio.app` does not require Gradio until launch.

### Task 5: Run Scripts and Docs

**Files:**
- Create: `scripts/run_facegan_studio.sh`
- Modify: `requirements_autodl.txt`
- Modify: `README.md`
- Modify: `代码附录.md`

- [x] Add AutoDL startup script.
- [x] Add `gradio` dependency.
- [x] Document run commands and output directories.
- [x] Add code appendix entry for the new program.

### Task 6: Verification

**Commands:**
- `python -m unittest tests.test_facegan_studio_core -v`
- `python -m compileall facegan_studio scripts`
- `python -m facegan_studio.app --help`

- [x] Run lightweight verification commands in local Windows workspace.
- [x] Report exact results and runtime limitations in migration handoff docs.

**Verification result on 2026-05-07:**

- `python -m compileall src scripts facegan_studio tests`: passed.
- `python -m unittest tests.test_facegan_studio_core -v`: 5 tests passed.
- `python -m facegan_studio.app --help`: passed.
- AutoDL GPU runtime verification for AnimeGANv2/CycleGAN/InstantID is still required because local Windows does not contain the full AutoDL model/runtime stack.
