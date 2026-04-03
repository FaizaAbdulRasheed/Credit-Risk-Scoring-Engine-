"""
Centralised configuration loader.
Reads config.yaml and exposes a typed Config dataclass.
"""
import logging
import yaml
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and return as a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def get_random_seed(config: Dict) -> int:
    return config.get("project", {}).get("random_seed", 42)
