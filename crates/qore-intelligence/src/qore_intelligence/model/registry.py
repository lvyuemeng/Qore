from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.artifact import TrainedModelArtifact


@dataclass(slots=True)
class ModelRegistry:
    root: Path

    @classmethod
    def from_settings(cls, settings: IntelligenceSettings) -> ModelRegistry:
        return cls(root=Path(settings.model_store_root))

    def save(self, artifact: TrainedModelArtifact, version: str | None = None) -> Path:
        resolved_version = (
            version or artifact.manifest.training_metadata.trained_at.date().isoformat()
        )
        path = (
            self.root
            / artifact.manifest.model_name
            / resolved_version
            / "artifact.joblib"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)

        latest = self.root / artifact.manifest.model_name / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(path.parent, target_is_directory=True)
        except OSError:
            latest.write_text(str(path.parent), encoding="utf-8")
        return path

    def load(self, model_name: str, version: str = "latest") -> TrainedModelArtifact:
        root = self.root / model_name
        if version == "latest":
            latest = root / "latest"
            if latest.is_symlink():
                path = latest / "artifact.joblib"
            elif latest.exists():
                path = Path(latest.read_text(encoding="utf-8")) / "artifact.joblib"
            else:
                msg = f"No saved model artifact found for {model_name!r}"
                raise FileNotFoundError(msg)
        else:
            path = root / version / "artifact.joblib"
        return joblib.load(path)
