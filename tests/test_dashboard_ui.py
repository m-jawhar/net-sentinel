"""
Test the Streamlit UI layer of the dashboard.

Verifies that every _render_* method in DashboardApp actually calls the
correct underlying service methods and passes well-formed data to Streamlit
widgets.  We mock `streamlit` so we can run without a Streamlit server and
intercept every `st.*` call.
"""

import sys
import os
import time
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netsentinel.sniffer import SimulatedSniffer
from netsentinel.database import DatabaseManager, TrafficLog
from netsentinel.database.models import (
    Alert,
    AlertType,
    AlertSeverity,
    ConnectionPair,
    IPStatistics,
)
from netsentinel.visualizer import NetworkGraph, GraphAnalyzer
from netsentinel.ml import AnomalyDetector


# ---------------------------------------------------------------------------
# Helpers — build real test data once
# ---------------------------------------------------------------------------
TEST_DB = "data/test_ui_layer.db"


def _setup_test_data():
    """Capture simulated packets, store in DB, generate alerts."""
    db = DatabaseManager(TEST_DB)
    sniffer = SimulatedSniffer()
    captured = []
    sniffer.add_callback(lambda p: captured.append(p))
    sniffer.start(count=150)
    time.sleep(5)
    sniffer.stop()

    logs = [
        TrafficLog(
            timestamp=p.timestamp,
            src_ip=p.src_ip,
            dst_ip=p.dst_ip,
            src_port=p.src_port,
            dst_port=p.dst_port,
            protocol=p.protocol,
            packet_size=p.size,
            flags=p.flags,
            ttl=p.ttl,
        )
        for p in captured
    ]
    db.insert_traffic_logs_bulk(logs)

    detector = AnomalyDetector(
        enable_statistical=True, enable_rules=True, enable_ml=False
    )
    for p in captured:
        for anomaly in detector.check_packet(p):
            db.insert_alert(detector.create_alert(anomaly))

    return db, sniffer, detector, captured


# ---------------------------------------------------------------------------
# Build data once for the whole module
# ---------------------------------------------------------------------------
_db, _sniffer, _detector, _captured = _setup_test_data()

# Build a graph the same way the dashboard does
_graph = NetworkGraph()
_connections = _db.get_connection_pairs(limit=100)
for c in _connections:
    _graph.add_edge(
        c.src_ip, c.dst_ip, c.byte_count, c.protocols[0] if c.protocols else "TCP"
    )
_analyzer = GraphAnalyzer(_graph)


def _make_dashboard(mock_st):
    """
    Construct a DashboardApp with mocked streamlit but REAL services.

    The key insight: we mock `st` so the render methods don't need
    a browser, but the underlying data services are REAL — so any
    mismatch between what the UI expects and what the services
    return will raise an exception.
    """
    # session_state acts like a dict + attribute access
    session_state = {
        "db": _db,
        "graph": _graph,
        "analyzer": _analyzer,
        "anomaly_detector": _detector,
        "sniffer": _sniffer,
        "is_capturing": False,
        "max_nodes": 50,
        "refresh_rate": 2,
        "_packet_buffer": [],
    }

    class FakeSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    mock_st.session_state = FakeSessionState(session_state)

    # Make st.columns return the right number of context managers
    def make_columns(n):
        return [MagicMock() for _ in range(n)]

    mock_st.columns.side_effect = make_columns

    # st.tabs returns context managers too
    def make_tabs(labels):
        return [MagicMock() for _ in labels]

    mock_st.tabs.side_effect = make_tabs

    # st.expander returns a context manager
    mock_st.expander.return_value = MagicMock()

    # st.sidebar needs .button, .slider, .metric, etc.
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.button.return_value = False
    mock_st.sidebar.slider.return_value = 2

    # st.selectbox returns a valid option
    mock_st.selectbox.return_value = 60

    # st.text_input returns empty by default (no IP analysis triggered)
    mock_st.text_input.return_value = ""

    # st.button returns False (no resolve button clicked)
    mock_st.button.return_value = False

    # st.components.v1.html — kept for any legacy calls
    mock_st.components = MagicMock()
    # st.iframe — replaces st.components.v1.html for vis.js graph
    mock_st.iframe = MagicMock()

    # st.fragment — decorator that just calls the function immediately
    def fake_fragment(func=None, *, run_every=None):
        if func is not None:
            return func

        # Called as @st.fragment(run_every=...)
        def decorator(fn):
            return fn

        return decorator

    mock_st.fragment = fake_fragment

    # Now import and construct
    import netsentinel.dashboard.app as app_module

    app_module.st = mock_st
    dashboard = app_module.DashboardApp.__new__(app_module.DashboardApp)
    dashboard.config = app_module.Config()
    dashboard.db = _db
    dashboard.graph = _graph
    dashboard.analyzer = _analyzer
    dashboard.anomaly_detector = _detector
    dashboard.sniffer = _sniffer

    return dashboard, app_module


# ===========================================================================
# Test Cases
# ===========================================================================
class TestDashboardUILayer(unittest.TestCase):
    """Test that each render method calls services correctly and doesn't crash."""

    def setUp(self):
        self.mock_st = MagicMock()
        self.dashboard, self.app_module = _make_dashboard(self.mock_st)

    # -----------------------------------------------------------------------
    # _render_sidebar
    # -----------------------------------------------------------------------
    def test_render_sidebar_calls_db_stats(self):
        """Sidebar must call get_database_stats and display metrics."""
        self.dashboard._render_sidebar()

        # Verify db.get_database_stats was used — the sidebar displays 3 metrics from it
        # We check st.sidebar.metric was called (Traffic Logs, Alerts, DB Size)
        metric_calls = self.mock_st.sidebar.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Packets Captured", metric_labels)
        self.assertIn("Traffic Logs", metric_labels)
        self.assertIn("Alerts", metric_labels)
        self.assertIn("DB Size", metric_labels)

    def test_render_sidebar_cleanup_button(self):
        """If cleanup button clicked, cleanup_old_logs should be called."""
        # Simulate button click for "Clear Old Data"
        click_count = [0]

        def sidebar_button_effect(label, **kwargs):
            if "Clear" in label:
                return True
            return False

        self.mock_st.sidebar.button.side_effect = sidebar_button_effect
        # Should not crash — cleanup_old_logs is real
        self.dashboard._render_sidebar()
        self.mock_st.sidebar.success.assert_called()

    # -----------------------------------------------------------------------
    # _render_live_traffic — sniffer path
    # -----------------------------------------------------------------------
    def test_render_live_traffic_sniffer_path(self):
        """When sniffer has packets, should use sniffer stats."""
        # Our sniffer has captured data from setup
        self.dashboard._render_live_traffic()

        # Should have called st.metric for Total Packets, Total Bytes, TCP, UDP
        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Total Packets", metric_labels)
        self.assertIn("Total Bytes", metric_labels)
        self.assertIn("TCP Packets", metric_labels)
        self.assertIn("UDP Packets", metric_labels)

    def test_render_live_traffic_db_fallback(self):
        """When sniffer has 0 packets, should fall back to DB stats."""
        # Create a fresh sniffer with no captures
        fresh_sniffer = SimulatedSniffer()
        self.dashboard.sniffer = fresh_sniffer
        self.dashboard._render_live_traffic()

        # Should still render metrics without crashing
        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Total Packets", metric_labels)

    def test_render_live_traffic_shows_recent_packets(self):
        """Should call st.dataframe with recent packet data."""
        self.dashboard._render_live_traffic()
        # st.dataframe should have been called (we have data in DB)
        self.mock_st.dataframe.assert_called()

    def test_render_live_traffic_protocol_chart(self):
        """Should render a bar chart for protocol distribution."""
        self.dashboard._render_live_traffic()
        self.mock_st.bar_chart.assert_called()
        chart_data = self.mock_st.bar_chart.call_args[0][0]
        self.assertIn("TCP", chart_data)
        self.assertIn("UDP", chart_data)
        self.assertIn("ICMP", chart_data)

    def test_render_live_traffic_empty_db(self):
        """With empty DB, should show info messages instead of data."""
        empty_db = DatabaseManager("data/test_ui_empty.db")
        self.dashboard.db = empty_db
        # Also reset sniffer so sniffer path gives 0
        self.dashboard.sniffer = SimulatedSniffer()

        self.dashboard._render_live_traffic()
        # Should call st.info for "No packets captured yet" since DB is empty
        info_calls = [str(c) for c in self.mock_st.info.call_args_list]
        found_no_packets = any(
            "No packets" in str(c) or "Not enough" in str(c) for c in info_calls
        )
        self.assertTrue(
            found_no_packets or self.mock_st.info.called,
            "Expected st.info to be called for empty state",
        )
        empty_db.close()
        try:
            os.remove("data/test_ui_empty.db")
        except (PermissionError, FileNotFoundError):
            pass

    # -----------------------------------------------------------------------
    # _render_network_graph
    # -----------------------------------------------------------------------
    def test_render_network_graph_builds_graph(self):
        """Should rebuild graph from DB connections and show stats."""
        self.dashboard._render_network_graph()

        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Nodes (IPs)", metric_labels)
        self.assertIn("Edges (Connections)", metric_labels)
        self.assertIn("Avg Degree", metric_labels)
        self.assertIn("Components", metric_labels)

    def test_render_network_graph_top_talkers(self):
        """Should display top talkers dataframe."""
        self.dashboard._render_network_graph()
        # st.dataframe called for top talkers
        self.assertTrue(self.mock_st.dataframe.called)

    def test_render_network_graph_vis_js(self):
        """Should render vis.js HTML via st.iframe."""
        self.dashboard._render_network_graph()
        # st.iframe should be called with the vis.js graph HTML
        self.mock_st.iframe.assert_called()
        html_arg = self.mock_st.iframe.call_args[0][0]
        self.assertIn("vis-network", html_arg)
        self.assertIn("vis.DataSet", html_arg)

    def test_render_network_graph_analysis_report(self):
        """Graph analysis expander should use correct report keys."""
        self.dashboard._render_network_graph()
        # st.json should be called with degree distribution
        self.mock_st.json.assert_called()

    def test_render_network_graph_empty(self):
        """With empty DB, should show info about no connections."""
        empty_db = DatabaseManager("data/test_ui_empty2.db")
        self.dashboard.db = empty_db
        empty_graph = NetworkGraph()
        self.dashboard.graph = empty_graph
        self.dashboard.analyzer = GraphAnalyzer(empty_graph)

        self.dashboard._render_network_graph()
        # Should call st.info about no connections
        self.mock_st.info.assert_called()
        empty_db.close()
        try:
            os.remove("data/test_ui_empty2.db")
        except (PermissionError, FileNotFoundError):
            pass

    # -----------------------------------------------------------------------
    # _render_alerts
    # -----------------------------------------------------------------------
    def test_render_alerts_shows_metrics(self):
        """Should display anomaly count, rate, and packets analyzed."""
        self.dashboard._render_alerts()

        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Total Anomalies", metric_labels)
        self.assertIn("Anomaly Rate", metric_labels)
        self.assertIn("Packets Analyzed", metric_labels)

    def test_render_alerts_shows_alert_expanders(self):
        """Each alert should create an expander with correct format."""
        self.dashboard._render_alerts()
        # st.expander should be called for each alert
        expander_calls = self.mock_st.expander.call_args_list
        self.assertTrue(len(expander_calls) > 0, "No alert expanders rendered")

        # Each expander label should have severity icon and alert type
        for exp_call in expander_calls:
            label = exp_call[0][0] if exp_call[0] else str(exp_call)
            # Labels with alerts contain emoji icons
            if any(icon in label for icon in ["🟢", "🟡", "🟠", "🔴"]):
                # Verify it has a timestamp
                self.assertRegex(label, r"\d{4}-\d{2}-\d{2}")

    def test_render_alerts_db_fallback(self):
        """When detector has 0 packets checked, use DB counts."""
        fresh_det = AnomalyDetector(
            enable_statistical=True, enable_rules=True, enable_ml=False
        )
        self.dashboard.anomaly_detector = fresh_det
        self.dashboard._render_alerts()

        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Total Anomalies", metric_labels)

    def test_render_alerts_flagged_ips(self):
        """Flagged IPs section should render dataframe or success message."""
        self.dashboard._render_alerts()
        # Either st.dataframe (flagged IPs found) or st.success (none found)
        self.assertTrue(
            self.mock_st.dataframe.called or self.mock_st.success.called,
            "Neither flagged IPs dataframe nor success message rendered",
        )

    def test_render_alerts_detection_settings(self):
        """Detection settings expander should list enabled detectors."""
        self.dashboard._render_alerts()
        # st.checkbox should be called for each detector type
        checkbox_calls = self.mock_st.checkbox.call_args_list
        # We have statistical, rules, ml — at least 2 should show
        self.assertTrue(
            len(checkbox_calls) >= 2,
            f"Expected >=2 detector checkboxes, got {len(checkbox_calls)}",
        )

    def test_render_alerts_no_alerts(self):
        """With no alerts, should show success message."""
        empty_db = DatabaseManager("data/test_ui_empty3.db")
        self.dashboard.db = empty_db
        fresh_det = AnomalyDetector(
            enable_statistical=True, enable_rules=True, enable_ml=False
        )
        self.dashboard.anomaly_detector = fresh_det

        self.dashboard._render_alerts()
        # Should show "No alerts! Network looks healthy." or similar
        success_calls = [str(c) for c in self.mock_st.success.call_args_list]
        found_healthy = any("healthy" in s or "No alerts" in s for s in success_calls)
        self.assertTrue(
            found_healthy, f"Expected 'healthy' success message, got: {success_calls}"
        )
        empty_db.close()
        try:
            os.remove("data/test_ui_empty3.db")
        except (PermissionError, FileNotFoundError):
            pass

    # -----------------------------------------------------------------------
    # _render_analytics
    # -----------------------------------------------------------------------
    def test_render_analytics_metrics(self):
        """Should display packet count, bytes, sources, destinations."""
        self.dashboard._render_analytics()

        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Total Packets", metric_labels)
        self.assertIn("Total Bytes", metric_labels)
        self.assertIn("Unique Sources", metric_labels)
        self.assertIn("Unique Destinations", metric_labels)

    def test_render_analytics_protocol_chart(self):
        """Should render protocol breakdown bar chart."""
        self.dashboard._render_analytics()
        self.mock_st.bar_chart.assert_called()

    def test_render_analytics_top_sources(self):
        """Should render top sources dataframe."""
        self.dashboard._render_analytics()
        self.assertTrue(self.mock_st.dataframe.called)

    def test_render_analytics_anomaly_trend(self):
        """Should display normal vs anomalous traffic counts."""
        self.dashboard._render_analytics()
        write_calls = [str(c) for c in self.mock_st.write.call_args_list]
        found_normal = any("Normal Traffic" in s for s in write_calls)
        found_anomalous = any("Anomalous Traffic" in s for s in write_calls)
        self.assertTrue(found_normal, "Missing 'Normal Traffic' in analytics")
        self.assertTrue(found_anomalous, "Missing 'Anomalous Traffic' in analytics")

    def test_render_analytics_ip_analysis(self):
        """When IP is entered, should display IP statistics."""
        test_ip = _captured[0].src_ip
        self.mock_st.text_input.return_value = test_ip
        self.dashboard._render_analytics()

        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        self.assertIn("Packets Sent", metric_labels)
        self.assertIn("Packets Received", metric_labels)
        self.assertIn("Bytes Sent", metric_labels)
        self.assertIn("Bytes Received", metric_labels)

    def test_render_analytics_empty_ip(self):
        """When no IP entered, IP analysis section should not render metrics."""
        self.mock_st.text_input.return_value = ""
        self.mock_st.metric.reset_mock()
        self.dashboard._render_analytics()

        metric_calls = self.mock_st.metric.call_args_list
        metric_labels = [c[0][0] for c in metric_calls]
        # Should NOT have IP-specific metrics
        self.assertNotIn("Packets Sent", metric_labels)

    # -----------------------------------------------------------------------
    # _get_packet_volume_timeseries
    # -----------------------------------------------------------------------
    def test_timeseries_returns_correct_format(self):
        """Timeseries should return DataFrame with 'Packets' column or None."""
        import pandas as pd

        result = self.dashboard._get_packet_volume_timeseries()
        if result is not None:
            self.assertIsInstance(result, pd.DataFrame)
            self.assertIn("Packets", result.columns)
            # Index should contain time-label strings (e.g. "14:05")
            self.assertTrue(len(result.index) >= 2)
            self.assertTrue(all(isinstance(v, str) for v in result.index))

    def test_timeseries_empty_db(self):
        """With empty DB, should return None."""
        empty_db = DatabaseManager("data/test_ui_empty4.db")
        self.dashboard.db = empty_db
        result = self.dashboard._get_packet_volume_timeseries()
        self.assertIsNone(result)
        empty_db.close()
        try:
            os.remove("data/test_ui_empty4.db")
        except (PermissionError, FileNotFoundError):
            pass

    # -----------------------------------------------------------------------
    # _get_flagged_ips
    # -----------------------------------------------------------------------
    def test_flagged_ips_format(self):
        """Flagged IPs should return list of dicts with correct keys."""
        result = self.dashboard._get_flagged_ips()
        if result is not None:
            self.assertIsInstance(result, list)
            for row in result:
                self.assertIn("IP Address", row)
                self.assertIn("Alert Count", row)
                self.assertIn("Severity", row)
                self.assertIn("Reasons", row)
                # Reasons should be string, not set
                self.assertIsInstance(row["Reasons"], str)

    def test_flagged_ips_empty(self):
        """With empty DB, should return None."""
        empty_db = DatabaseManager("data/test_ui_empty5.db")
        self.dashboard.db = empty_db
        empty_graph = NetworkGraph()
        self.dashboard.graph = empty_graph
        self.dashboard.analyzer = GraphAnalyzer(empty_graph)
        result = self.dashboard._get_flagged_ips()
        self.assertIsNone(result)
        empty_db.close()
        try:
            os.remove("data/test_ui_empty5.db")
        except (PermissionError, FileNotFoundError):
            pass

    # -----------------------------------------------------------------------
    # _format_bytes (static)
    # -----------------------------------------------------------------------
    def test_format_bytes(self):
        """Static method should format bytes correctly."""
        from netsentinel.dashboard.app import DashboardApp

        self.assertEqual(DashboardApp._format_bytes(0), "0.0 B")
        self.assertEqual(DashboardApp._format_bytes(512), "512.0 B")
        self.assertEqual(DashboardApp._format_bytes(1024), "1.0 KB")
        self.assertEqual(DashboardApp._format_bytes(1048576), "1.0 MB")
        self.assertEqual(DashboardApp._format_bytes(1073741824), "1.0 GB")
        self.assertEqual(DashboardApp._format_bytes(1099511627776), "1.0 TB")

    # -----------------------------------------------------------------------
    # Full run() integration
    # -----------------------------------------------------------------------
    def test_full_run_no_crash(self):
        """The full run() method should not crash with real data."""
        self.dashboard.run()
        # If we get here, no exceptions were raised
        self.mock_st.set_page_config.assert_called_once()
        self.mock_st.title.assert_called()
        self.mock_st.tabs.assert_called()


# ===========================================================================
# Cleanup
# ===========================================================================
def teardown_module():
    _db.close()
    for f in [TEST_DB]:
        try:
            os.remove(f)
        except (PermissionError, FileNotFoundError):
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
