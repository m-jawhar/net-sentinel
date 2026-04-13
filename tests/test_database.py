"""
Tests for the database module.
"""

import os
import sys
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from netsentinel.database.db_manager import DatabaseManager
from netsentinel.database.models import TrafficLog, Alert, AlertType, AlertSeverity


class TestDatabaseManager:
    """Tests for DatabaseManager."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a test database."""
        db_path = str(tmp_path / "test.db")
        return DatabaseManager(db_path)

    @pytest.fixture
    def sample_log(self):
        """Create a sample traffic log."""
        return TrafficLog(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="8.8.8.8",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            packet_size=1024,
            flags="SYN-ACK",
            ttl=64,
        )

    def test_insert_and_retrieve(self, db, sample_log):
        """Test inserting and retrieving a traffic log."""
        log_id = db.insert_traffic_log(sample_log)
        assert log_id > 0

        logs = db.get_recent_traffic(limit=1)
        assert len(logs) == 1
        assert logs[0].src_ip == "192.168.1.1"
        assert logs[0].dst_ip == "8.8.8.8"
        assert logs[0].protocol == "TCP"

    def test_bulk_insert(self, db):
        """Test bulk insertion."""
        logs = [
            TrafficLog(
                timestamp=datetime.now(),
                src_ip=f"192.168.1.{i}",
                dst_ip="8.8.8.8",
                protocol="TCP",
                packet_size=100 + i,
            )
            for i in range(100)
        ]

        count = db.insert_traffic_logs_bulk(logs)
        assert count == 100

        stats = db.get_database_stats()
        assert stats["traffic_log_count"] == 100

    def test_traffic_by_ip(self, db, sample_log):
        """Test querying traffic by IP."""
        db.insert_traffic_log(sample_log)

        src_logs = db.get_traffic_by_ip("192.168.1.1", "src")
        assert len(src_logs) == 1

        dst_logs = db.get_traffic_by_ip("8.8.8.8", "dst")
        assert len(dst_logs) == 1

        both_logs = db.get_traffic_by_ip("192.168.1.1", "both")
        assert len(both_logs) == 1

    def test_statistics(self, db):
        """Test traffic statistics."""
        for i in range(50):
            log = TrafficLog(
                timestamp=datetime.now(),
                src_ip="192.168.1.1",
                dst_ip=f"10.0.0.{i % 10}",
                protocol="TCP" if i % 2 == 0 else "UDP",
                packet_size=100 * (i + 1),
            )
            db.insert_traffic_log(log)

        stats = db.get_traffic_statistics(minutes=5)
        assert stats["packet_count"] == 50
        assert stats["unique_sources"] == 1

    def test_alerts(self, db):
        """Test alert operations."""
        alert = Alert(
            timestamp=datetime.now(),
            alert_type=AlertType.PORT_SCAN,
            severity=AlertSeverity.HIGH,
            description="Port scan detected",
            src_ip="10.0.0.1",
        )

        alert_id = db.insert_alert(alert)
        assert alert_id > 0

        alerts = db.get_recent_alerts(limit=10)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.PORT_SCAN

        # Resolve alert
        db.resolve_alert(alert_id)

        unresolved = db.get_recent_alerts(limit=10, unresolved_only=True)
        assert len(unresolved) == 0

    def test_connection_pairs(self, db):
        """Test connection pair aggregation."""
        for i in range(10):
            log = TrafficLog(
                timestamp=datetime.now(),
                src_ip="192.168.1.1",
                dst_ip="8.8.8.8",
                protocol="TCP",
                packet_size=100,
            )
            db.insert_traffic_log(log)

        pairs = db.get_connection_pairs()
        assert len(pairs) == 1
        assert pairs[0].packet_count == 10

    def test_cleanup(self, db):
        """Test old data cleanup."""
        # Insert old log
        old_log = TrafficLog(
            timestamp=datetime.now() - timedelta(days=30),
            src_ip="192.168.1.1",
            dst_ip="8.8.8.8",
            protocol="TCP",
            packet_size=100,
        )
        db.insert_traffic_log(old_log)

        deleted = db.cleanup_old_logs(days=7)
        assert deleted == 1


class TestTrafficLog:
    """Tests for TrafficLog model."""

    def test_to_tuple(self):
        """Test conversion to tuple."""
        log = TrafficLog(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            src_ip="192.168.1.1",
            dst_ip="8.8.8.8",
            protocol="TCP",
            packet_size=1024,
        )

        t = log.to_tuple()
        assert len(t) == 11
        assert t[1] == "192.168.1.1"
        assert t[2] == "8.8.8.8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
