"""
Database Manager - Handles all database operations.

Applies DBMS concepts (Sem 4):
- DDL (CREATE, ALTER, DROP)
- DML (INSERT, UPDATE, DELETE, SELECT)
- Indexing for fast retrieval
- Aggregation queries
- Transaction management
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from contextlib import contextmanager

from .models import (
    TrafficLog,
    Alert,
    AlertType,
    AlertSeverity,
    IPStatistics,
    ConnectionPair,
)


class DatabaseManager:
    """
    SQLite database manager for storing network traffic logs and alerts.

    Features:
    - Thread-safe database operations
    - Connection pooling
    - Automatic schema creation
    - Efficient bulk inserts
    """

    def __init__(self, db_path: str = "data/netsentinel.db"):
        """
        Initialize the database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialize_schema()

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")

        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e

    def _initialize_schema(self):
        """
        Create database schema if it doesn't exist.

        Applies Normalization principles:
        - 1NF: Each column contains atomic values
        - 2NF: No partial dependencies
        - 3NF: No transitive dependencies
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Traffic Logs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Traffic_Logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    src_ip VARCHAR(45) NOT NULL,
                    dst_ip VARCHAR(45) NOT NULL,
                    src_port INTEGER DEFAULT 0,
                    dst_port INTEGER DEFAULT 0,
                    protocol VARCHAR(10) NOT NULL,
                    packet_size INTEGER NOT NULL,
                    flags VARCHAR(50),
                    ttl INTEGER,
                    is_anomaly BOOLEAN DEFAULT 0,
                    anomaly_score REAL DEFAULT 0.0
                )
            """
            )

            # Alerts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    alert_type VARCHAR(50) NOT NULL,
                    severity INTEGER NOT NULL,
                    src_ip VARCHAR(45),
                    dst_ip VARCHAR(45),
                    description TEXT,
                    is_resolved BOOLEAN DEFAULT 0,
                    resolved_at DATETIME,
                    packet_count INTEGER DEFAULT 0,
                    byte_count INTEGER DEFAULT 0
                )
            """
            )

            # Create indexes for faster queries
            # Index on timestamp for time-range queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traffic_timestamp
                ON Traffic_Logs(timestamp)
            """
            )

            # Index on source IP for filtering
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traffic_src_ip
                ON Traffic_Logs(src_ip)
            """
            )

            # Index on destination IP
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traffic_dst_ip
                ON Traffic_Logs(dst_ip)
            """
            )

            # Composite index for connection pairs
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traffic_connection
                ON Traffic_Logs(src_ip, dst_ip)
            """
            )

            # Index on anomalies
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traffic_anomaly
                ON Traffic_Logs(is_anomaly) WHERE is_anomaly = 1
            """
            )

            # Index on alerts timestamp
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
                ON Alerts(timestamp)
            """
            )

            # ==========================================
            # SQL Triggers (DBMS Sem 4)
            # ==========================================
            # Trigger: Auto-set timestamp on high-priority alerts
            # When a HIGH or CRITICAL alert is inserted, automatically
            # record the escalation timestamp in the description field.
            # This demonstrates DDL trigger concepts from DBMS.
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_alert_high_priority
                AFTER INSERT ON Alerts
                WHEN NEW.severity >= 3
                BEGIN
                    UPDATE Alerts
                    SET description = NEW.description ||
                        ' [AUTO-ESCALATED at ' || datetime('now') || ']'
                    WHERE id = NEW.id;
                END
            """
            )

            # Trigger: Auto-flag traffic as anomaly when packet size
            # exceeds threshold (demonstrates data integrity triggers)
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_large_packet_flag
                AFTER INSERT ON Traffic_Logs
                WHEN NEW.packet_size > 10000 AND NEW.is_anomaly = 0
                BEGIN
                    UPDATE Traffic_Logs
                    SET is_anomaly = 1,
                        anomaly_score = MIN(1.0, NEW.packet_size / 50000.0)
                    WHERE id = NEW.id;
                END
            """
            )

            conn.commit()

    # ===================
    # Traffic Log Operations
    # ===================

    def insert_traffic_log(self, log: TrafficLog) -> int:
        """Insert a single traffic log entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO Traffic_Logs
                (timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
                 packet_size, flags, ttl, is_anomaly, anomaly_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                log.to_tuple(),
            )
            conn.commit()
            return cursor.lastrowid

    def insert_traffic_logs_bulk(self, logs: List[TrafficLog]) -> int:
        """
        Bulk insert traffic logs for better performance.

        Uses transaction batching for efficiency.
        """
        if not logs:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO Traffic_Logs
                (timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
                 packet_size, flags, ttl, is_anomaly, anomaly_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [log.to_tuple() for log in logs],
            )
            conn.commit()
            return cursor.rowcount

    def get_recent_traffic(self, limit: int = 100) -> List[TrafficLog]:
        """Get the most recent traffic logs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM Traffic_Logs
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )
            return [TrafficLog.from_row(tuple(row)) for row in cursor.fetchall()]

    def get_traffic_in_timerange(
        self, start_time: datetime, end_time: datetime
    ) -> List[TrafficLog]:
        """Get traffic logs within a time range."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM Traffic_Logs
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            return [TrafficLog.from_row(tuple(row)) for row in cursor.fetchall()]

    def get_traffic_by_ip(
        self, ip_address: str, direction: str = "both"
    ) -> List[TrafficLog]:
        """
        Get traffic logs for a specific IP address.

        Args:
            ip_address: IP to filter by
            direction: "src", "dst", or "both"
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if direction == "src":
                cursor.execute(
                    "SELECT * FROM Traffic_Logs WHERE src_ip = ? ORDER BY timestamp DESC",
                    (ip_address,),
                )
            elif direction == "dst":
                cursor.execute(
                    "SELECT * FROM Traffic_Logs WHERE dst_ip = ? ORDER BY timestamp DESC",
                    (ip_address,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM Traffic_Logs WHERE src_ip = ? OR dst_ip = ? ORDER BY timestamp DESC",
                    (ip_address, ip_address),
                )

            return [TrafficLog.from_row(tuple(row)) for row in cursor.fetchall()]

    def get_anomalies(self, limit: int = 100) -> List[TrafficLog]:
        """Get traffic logs flagged as anomalies."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM Traffic_Logs
                WHERE is_anomaly = 1
                ORDER BY anomaly_score DESC, timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )
            return [TrafficLog.from_row(tuple(row)) for row in cursor.fetchall()]

    # ===================
    # Aggregation Queries
    # ===================

    def get_packet_count_by_ip(self, ip_address: str, minutes: int = 1) -> int:
        """
        Count packets from an IP in the last N minutes.

        Useful for DDoS detection: high packet count = suspicious.
        """
        start_time = datetime.now() - timedelta(minutes=minutes)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM Traffic_Logs
                WHERE src_ip = ? AND timestamp >= ?
            """,
                (ip_address, start_time.isoformat()),
            )
            return cursor.fetchone()[0]

    def get_traffic_statistics(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get traffic statistics for the last N minutes.

        Returns aggregated metrics for dashboard display.
        """
        start_time = datetime.now() - timedelta(minutes=minutes)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total packets and bytes
            cursor.execute(
                """
                SELECT
                    COUNT(*) as packet_count,
                    COALESCE(SUM(packet_size), 0) as total_bytes,
                    COALESCE(AVG(packet_size), 0) as avg_packet_size,
                    COUNT(DISTINCT src_ip) as unique_sources,
                    COUNT(DISTINCT dst_ip) as unique_destinations
                FROM Traffic_Logs
                WHERE timestamp >= ?
            """,
                (start_time.isoformat(),),
            )

            row = cursor.fetchone()

            # Protocol distribution
            cursor.execute(
                """
                SELECT protocol, COUNT(*) as count
                FROM Traffic_Logs
                WHERE timestamp >= ?
                GROUP BY protocol
                ORDER BY count DESC
            """,
                (start_time.isoformat(),),
            )

            protocols = {row[0]: row[1] for row in cursor.fetchall()}

            # Top talkers (source IPs)
            cursor.execute(
                """
                SELECT src_ip, COUNT(*) as packet_count, SUM(packet_size) as bytes
                FROM Traffic_Logs
                WHERE timestamp >= ?
                GROUP BY src_ip
                ORDER BY packet_count DESC
                LIMIT 10
            """,
                (start_time.isoformat(),),
            )

            top_sources = [
                {"ip": row[0], "packets": row[1], "bytes": row[2]}
                for row in cursor.fetchall()
            ]

            # Anomaly count
            cursor.execute(
                """
                SELECT COUNT(*) FROM Traffic_Logs
                WHERE timestamp >= ? AND is_anomaly = 1
            """,
                (start_time.isoformat(),),
            )
            anomaly_count = cursor.fetchone()[0]

            return {
                "packet_count": row[0],
                "total_bytes": row[1],
                "avg_packet_size": row[2],
                "unique_sources": row[3],
                "unique_destinations": row[4],
                "protocols": protocols,
                "top_sources": top_sources,
                "anomaly_count": anomaly_count,
                "time_range_minutes": minutes,
            }

    def get_connection_pairs(self, limit: int = 100) -> List[ConnectionPair]:
        """
        Get unique connection pairs with statistics.

        Used for graph visualization edges.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    src_ip,
                    dst_ip,
                    COUNT(*) as packet_count,
                    SUM(packet_size) as byte_count,
                    GROUP_CONCAT(DISTINCT protocol) as protocols,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM Traffic_Logs
                GROUP BY src_ip, dst_ip
                ORDER BY packet_count DESC
                LIMIT ?
            """,
                (limit,),
            )

            pairs = []
            for row in cursor.fetchall():
                pairs.append(
                    ConnectionPair(
                        src_ip=row[0],
                        dst_ip=row[1],
                        packet_count=row[2],
                        byte_count=row[3],
                        protocols=row[4].split(",") if row[4] else [],
                        first_seen=datetime.fromisoformat(row[5]) if row[5] else None,
                        last_seen=datetime.fromisoformat(row[6]) if row[6] else None,
                    )
                )

            return pairs

    def get_ip_statistics(self, ip_address: str) -> IPStatistics:
        """Get detailed statistics for a specific IP."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Outbound stats
            cursor.execute(
                """
                SELECT
                    COUNT(*) as packets,
                    COALESCE(SUM(packet_size), 0) as bytes,
                    COUNT(DISTINCT dst_ip) as destinations,
                    GROUP_CONCAT(DISTINCT protocol) as protocols,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM Traffic_Logs
                WHERE src_ip = ?
            """,
                (ip_address,),
            )
            out_row = cursor.fetchone()

            # Inbound stats
            cursor.execute(
                """
                SELECT
                    COUNT(*) as packets,
                    COALESCE(SUM(packet_size), 0) as bytes,
                    COUNT(DISTINCT src_ip) as sources
                FROM Traffic_Logs
                WHERE dst_ip = ?
            """,
                (ip_address,),
            )
            in_row = cursor.fetchone()

            return IPStatistics(
                ip_address=ip_address,
                total_packets_sent=out_row[0] or 0,
                total_bytes_sent=out_row[1] or 0,
                unique_destinations=out_row[2] or 0,
                total_packets_received=in_row[0] or 0,
                total_bytes_received=in_row[1] or 0,
                unique_sources=in_row[2] or 0,
                protocols_used=out_row[3].split(",") if out_row[3] else [],
                first_seen=datetime.fromisoformat(out_row[4]) if out_row[4] else None,
                last_seen=datetime.fromisoformat(out_row[5]) if out_row[5] else None,
            )

    # ===================
    # Alert Operations
    # ===================

    def insert_alert(self, alert: Alert) -> int:
        """Insert a security alert."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO Alerts
                (timestamp, alert_type, severity, src_ip, dst_ip, description,
                 is_resolved, resolved_at, packet_count, byte_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                alert.to_tuple(),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_alerts(
        self, limit: int = 50, unresolved_only: bool = False
    ) -> List[Alert]:
        """Get recent alerts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if unresolved_only:
                cursor.execute(
                    """
                    SELECT * FROM Alerts
                    WHERE is_resolved = 0
                    ORDER BY severity DESC, timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM Alerts
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            return [Alert.from_row(tuple(row)) for row in cursor.fetchall()]

    def resolve_alert(self, alert_id: int):
        """Mark an alert as resolved."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE Alerts
                SET is_resolved = 1, resolved_at = ?
                WHERE id = ?
            """,
                (datetime.now().isoformat(), alert_id),
            )
            conn.commit()

    # ===================
    # Maintenance
    # ===================

    def cleanup_old_logs(self, days: int = 7) -> int:
        """Delete traffic logs older than N days."""
        cutoff = datetime.now() - timedelta(days=days)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM Traffic_Logs
                WHERE timestamp < ?
            """,
                (cutoff.isoformat(),),
            )
            conn.commit()
            return cursor.rowcount

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM Traffic_Logs")
            log_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM Alerts")
            alert_count = cursor.fetchone()[0]

            # Database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "traffic_log_count": log_count,
                "alert_count": alert_count,
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
            }

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
