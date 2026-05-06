from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    if project_root:
        return Path(project_root).expanduser().resolve()

    env_root = os.environ.get("FACEGAN_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "scripts").exists() and (candidate / "src").exists():
            return candidate
    return cwd


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    def __init__(self, project_root: str | Path | None = None) -> None:
        object.__setattr__(self, "project_root", resolve_project_root(project_root))

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs" / "facegan_studio"

    @property
    def report_assets_dir(self) -> Path:
        return self.project_root / "report" / "report_assets" / "facegan_studio"

    @property
    def showcase_results_dir(self) -> Path:
        return self.project_root / "GAN_new_showcase_results"

    @property
    def animegan_repo(self) -> Path:
        return self.project_root / "external" / "animegan2-pytorch"

    @property
    def cyclegan_repo(self) -> Path:
        return self.project_root / "external" / "pytorch-CycleGAN-and-pix2pix"

    @property
    def instantid_repo(self) -> Path:
        return self.project_root / "external" / "InstantID"

    def ensure_base_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.report_assets_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, mode: str, timestamp: str | None = None) -> Path:
        self.ensure_base_dirs()
        stamp = timestamp or current_timestamp()
        run_dir = self.outputs_dir / mode / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def create_report_run_dir(self, mode: str, timestamp: str | None = None) -> Path:
        self.ensure_base_dirs()
        stamp = timestamp or current_timestamp()
        run_dir = self.report_assets_dir / mode / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
