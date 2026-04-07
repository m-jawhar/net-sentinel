"""
Database Models - Data structures for database tables.

Applies DBMS concepts (Sem 4):
- Schema Design
- Normalization (3NF)
- Entity-Relationship modeling
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertType(Enum):
    """Types of security alerts."""

    ANOMALY = "anomaly"
    HIGH_TRAFFIC = "high_traffic"
    PORT_SCAN = "port_scan"
    DDOS_SUSPECT = "ddos_suspect"
    UNUSUAL_PROTOCOL = "unusual_protocol"
    LARGE_PACKET = "large_packet"


@dataclass
class TrafficLog:
    """
    Represents a single traffic log entry.

    Database Schema:
    CREATE TABLE Traffic_Logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        src_ip VARCHAR(45) NOT NULL,
        dst_ip VARCHAR(45) NOT NULL,
        src_port INTEGER,
        dst_port INTEGER,
        protocol VARCHAR(10) NOT NULL,
        packet_size INTEGER NOT NULL,
        flags VARCHAR(50),
        ttl INTEGER,
        is_anomaly BOOLEAN DEFAULT FALSE,
        anomaly_score REAL
    );
    """

    timestamp: datetime
    src_ip: str
    dst_ip: str
    protocol: str
    packet_size: int
    src_port: int = 0
    dst_port: int = 0
    flags: Optional[str] = None
    ttl: Optional[int] = None
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    id: Optional[int] = None

    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.timestamp.isoformat(),
            self.src_ip,
            self.dst_ip,
            self.src_port,
            self.dst_port,
            self.protocol,
            self.packet_size,
            self.flags,
            self.ttl,
            self.is_anomaly,
            self.anomaly_score,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "TrafficLog":
        """Create TrafficLog from database row."""
        return cls(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            src_ip=row[2],
            dst_ip=row[3],
            src_port=row[4],
            dst_port=row[5],
            protocol=row[6],
            packet_size=row[7],
            flags=row[8],
            ttl=row[9],
            is_anomaly=bool(row[10]),
            anomaly_score=row[11] or 0.0,
        )


@dataclass
class Alert:
    """
    Represents a security alert.

    Database Schema:
    CREATE TABLE Alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        alert_type VARCHAR(50) NOT NULL,
        severity INTEGER NOT NULL,
        src_ip VARCHAR(45),
        dst_ip VARCHAR(45),
        description TEXT,
        is_resolved BOOLEAN DEFAULT FALSE,
        resolved_at DATETIME,
        packet_count INTEGER,
        byte_count INTEGER
    );
    """

    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    description: str
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    packet_count: int = 0
    byte_count: int = 0
    id: Optional[int] = None

    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.timestamp.isoformat(),
            self.alert_type.value,
            self.severity.value,
            self.src_ip,
            self.dst_ip,
            self.description,
            self.is_resolved,
            self.resolved_at.isoformat() if self.resolved_at else None,
            self.packet_count,
            self.byte_count,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "Alert":
        """Create Alert from database row."""
        return cls(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            alert_type=AlertType(row[2]),
            severity=AlertSeverity(row[3]),
            src_ip=row[4],
            dst_ip=row[5],
            description=row[6],
            is_resolved=bool(row[7]),
            resolved_at=datetime.fromisoformat(row[8]) if row[8] else None,
            packet_count=row[9] or 0,
            byte_count=row[10] or 0,
        )


@dataclass
class IPStatistics:
    """Statistics for a specific IP address."""

    ip_address: str
    total_packets_sent: int = 0
    total_packets_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    unique_destinations: int = 0
    unique_sources: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    protocols_used: List[str] = field(default_factory=list)

    @property
    def total_packets(self) -> int:
        return self.total_packets_sent + self.total_packets_received

    @property
    def total_bytes(self) -> int:
        return self.total_bytes_sent + self.total_bytes_received


@dataclass
class ConnectionPair:
    """Statistics for a connection between two IPs."""

    src_ip: str
    dst_ip: str
    packet_count: int = 0
    byte_count: int = 0
    protocols: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
