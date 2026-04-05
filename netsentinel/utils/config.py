"""
Configuration Module - Centralized configuration management.

Supports:
- Default configuration
- Environment variable overrides
- Configuration file loading
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class Config:
    """
    Application configuration.

    Settings are loaded in order of precedence:
    1. Environment variables (highest)
    2. Configuration file
    3. Default values (lowest)
    """

    # Database settings
    database_path: str = "data/netsentinel.db"

    # Sniffer settings
    sniffer_interface: Optional[str] = None
    sniffer_buffer_size: int = 10000
    sniffer_timeout: Optional[int] = None

    # Graph settings
    max_graph_nodes: int = 100
    max_graph_edges: int = 500

    # Anomaly detection settings
    enable_statistical_detection: bool = True
    enable_rule_detection: bool = True
    enable_ml_detection: bool = True
    ml_model_path: Optional[str] = "models/decision_tree.pkl"

    # Statistical detector settings
    zscore_threshold: float = 3.0
    statistical_window_size: int = 1000

    # Dashboard settings
    dashboard_port: int = 8501
    dashboard_refresh_rate: int = 2

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/netsentinel.log"

    # Cleanup settings
    data_retention_days: int = 7

    # Feature extraction settings
    feature_window_seconds: float = 10.0

    def __post_init__(self):
        """Apply environment variable overrides."""
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Override settings from environment variables."""
        env_mappings = {
            "NETSENTINEL_DB_PATH": ("database_path", str),
            "NETSENTINEL_INTERFACE": ("sniffer_interface", str),
            "NETSENTINEL_BUFFER_SIZE": ("sniffer_buffer_size", int),
            "NETSENTINEL_MAX_NODES": ("max_graph_nodes", int),
            "NETSENTINEL_ENABLE_ML": (
                "enable_ml_detection",
                lambda x: x.lower() == "true",
            ),
            "NETSENTINEL_ML_MODEL": ("ml_model_path", str),
            "NETSENTINEL_ZSCORE": ("zscore_threshold", float),
            "NETSENTINEL_LOG_LEVEL": ("log_level", str),
            "NETSENTINEL_LOG_FILE": ("log_file", str),
            "NETSENTINEL_RETENTION_DAYS": ("data_retention_days", int),
        }

        for env_var, (attr, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    setattr(self, attr, converter(value))
                except (ValueError, TypeError):
                    pass  # Keep default if conversion fails

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """
        Load configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            Config object
        """
        config_path = Path(path)

        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            data = json.load(f)

        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_file(self, path: str):
        """
        Save configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database_path": self.database_path,
            "sniffer_interface": self.sniffer_interface,
            "sniffer_buffer_size": self.sniffer_buffer_size,
            "sniffer_timeout": self.sniffer_timeout,
            "max_graph_nodes": self.max_graph_nodes,
            "max_graph_edges": self.max_graph_edges,
            "enable_statistical_detection": self.enable_statistical_detection,
            "enable_rule_detection": self.enable_rule_detection,
            "enable_ml_detection": self.enable_ml_detection,
            "ml_model_path": self.ml_model_path,
            "zscore_threshold": self.zscore_threshold,
            "statistical_window_size": self.statistical_window_size,
            "dashboard_port": self.dashboard_port,
            "dashboard_refresh_rate": self.dashboard_refresh_rate,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "data_retention_days": self.data_retention_days,
            "feature_window_seconds": self.feature_window_seconds,
        }

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if valid, raises ValueError otherwise
        """
        if self.zscore_threshold < 0:
            raise ValueError("zscore_threshold must be positive")

        if self.sniffer_buffer_size < 100:
            raise ValueError("sniffer_buffer_size must be at least 100")

        if self.max_graph_nodes < 1:
            raise ValueError("max_graph_nodes must be positive")

        if self.data_retention_days < 1:
            raise ValueError("data_retention_days must be positive")

        return True


# Default configuration instance
default_config = Config()


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration instance.

    Args:
        config_path: Optional path to config file

    Returns:
        Config object
    """
    if config_path:
        return Config.from_file(config_path)

    # Check for config file in standard locations
    standard_paths = [
        "config.json",
        "netsentinel.json",
        Path.home() / ".netsentinel" / "config.json",
    ]

    for path in standard_paths:
        if Path(path).exists():
            return Config.from_file(str(path))

    return Config()
