"""Load and resolve paths from a paths.yaml config file."""

from __future__ import annotations

from pathlib import Path

import yaml


class Paths:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    @property
    def model_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output"

    @property
    def data_root(self) -> Path:
        return self.base_dir / "helmet" / "longtrain"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "helmet" / "longtrain_swift"

    def as_dict(self) -> dict[str, str]:
        return {
            "base_dir": str(self.base_dir),
            "model_dir": str(self.model_dir),
            "output_dir": str(self.output_dir),
            "data_dir": str(self.data_dir),
        }


def load_paths(paths_config: str) -> Paths:
    with open(paths_config) as f:
        raw = yaml.safe_load(f)
    return Paths(base_dir=raw["base_dir"])
