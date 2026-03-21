import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """
    Loads YAML configuration files.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def load(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)