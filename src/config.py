"""Application-wide configuration — immutable after startup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATASETS_DIR: Final[Path] = _PROJECT_ROOT / "datasets"
MODELS_DIR: Final[Path] = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Runtime settings (overridable via env vars)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Settings:
    """Centralised, immutable runtime configuration."""

    # Paths
    datasets_dir: Path = DATASETS_DIR
    models_dir: Path = MODELS_DIR

    # Instruction parser
    tfidf_max_features: int = 5000
    keyword_match_threshold: float = 0.15
    classifier_confidence_threshold: float = 0.30

    # Workflow generation
    max_workflow_depth: int = 50
    max_workflow_nodes: int = 200
    max_retry_loops: int = 3
    max_retry_depth: int = 5

    # Hallucination mitigation
    strict_grounding: bool = True
    allow_custom_steps: bool = False
    duplicate_removal: bool = True

    # Layout
    node_horizontal_spacing: float = 220.0
    node_vertical_spacing: float = 120.0
    layout_padding: float = 40.0

    # Local model (optional)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_timeout: float = 30.0
    use_local_model: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    max_render_time_ms: int = 1000

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables with sensible defaults."""
        overrides: dict[str, object] = {}
        for fld in cls.__dataclass_fields__:
            env_key = f"WFG_{fld.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                ftype = cls.__dataclass_fields__[fld].type
                if ftype == "bool":
                    overrides[fld] = env_val.lower() in ("1", "true", "yes")
                elif ftype == "int":
                    overrides[fld] = int(env_val)
                elif ftype == "float":
                    overrides[fld] = float(env_val)
                elif ftype == "Path":
                    overrides[fld] = Path(env_val)
                else:
                    overrides[fld] = env_val
        return cls(**overrides)  # type: ignore[arg-type]


# Singleton
settings = Settings.from_env()
