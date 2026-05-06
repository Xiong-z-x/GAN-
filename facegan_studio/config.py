from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False

    @classmethod
    def from_root(
        cls,
        project_root: str | Path | None = None,
        host: str = "0.0.0.0",
        port: int = 7860,
        share: bool = False,
    ) -> "AppConfig":
        root = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()
        return cls(project_root=root, host=host, port=port, share=share)
