"""
Logger Module - Centralized logging configuration.

Provides:
- Console and file logging
- Log rotation
- Structured logging format
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading


# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class NetSentinelLogger:
    """
    Centralized logger for NetSentinel application.

    Features:
    - Console output with colors
    - File logging with rotation
    - Thread-safe logging
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for logger."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: str = "netsentinel",
        level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (None for console only)
        """
        if hasattr(self, "_initialized"):
            return

        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Remove existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, DATE_FORMAT))
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            self._setup_file_handler(log_file)

        self._initialized = True

    def _setup_file_handler(self, log_file: str):
        """Set up file logging."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        self.logger.addHandler(file_handler)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)

    def log_packet(self, src_ip: str, dst_ip: str, protocol: str, size: int):
        """Log a captured packet."""
        self.debug(f"Packet: {src_ip} -> {dst_ip} [{protocol}] {size} bytes")

    def log_anomaly(self, ip: str, anomaly_type: str, score: float, details: str):
        """Log a detected anomaly."""
        self.warning(f"ANOMALY [{anomaly_type}] IP={ip} Score={score:.2f}: {details}")

    def log_alert(self, alert_type: str, severity: str, message: str):
        """Log a security alert."""
        if severity in ("HIGH", "CRITICAL"):
            self.error(f"ALERT [{severity}] {alert_type}: {message}")
        else:
            self.warning(f"ALERT [{severity}] {alert_type}: {message}")


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        result = super().format(record)

        # Restore original levelname
        record.levelname = levelname

        return result


def setup_logger(
    name: str = "netsentinel", level: str = "INFO", log_file: Optional[str] = None
) -> NetSentinelLogger:
    """
    Set up and return the logger.

    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file path

    Returns:
        Configured NetSentinelLogger instance
    """
    return NetSentinelLogger(name, level, log_file)


def get_logger() -> NetSentinelLogger:
    """Get the singleton logger instance."""
    return NetSentinelLogger()


# Module-level logger
logger = setup_logger()
