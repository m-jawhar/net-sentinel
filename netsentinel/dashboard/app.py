"""
Dashboard Application - Real-time network monitoring interface.

Uses Streamlit for web-based visualization.
Displays:
- Live traffic statistics
- Network topology graph
- Anomaly alerts
- Historical data
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path so absolute imports work when
# Streamlit runs this file directly as a script.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import project modules
from netsentinel.sniffer import SimulatedSniffer
from netsentinel.database import DatabaseManager, TrafficLog
from netsentinel.visualizer import NetworkGraph, GraphAnalyzer
from netsentinel.ml import AnomalyDetector
from netsentinel.utils import Config


class DashboardApp:
    """
    Streamlit-based dashboard for network monitoring.

    Features:
    - Real-time traffic visualization
    - Network topology graph
    - Anomaly detection alerts
    - Historical analysis
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize dashboard.

        Uses st.session_state to persist objects across Streamlit reruns.
        """
        self.config = config or Config()

        # Persist heavy objects across reruns via session_state
        if "db" not in st.session_state:
            st.session_state.db = DatabaseManager(self.config.database_path)
        if "graph" not in st.session_state:
            st.session_state.graph = NetworkGraph()
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = GraphAnalyzer(st.session_state.graph)
        if "anomaly_detector" not in st.session_state:
            # Auto-detect ML model if available
            ml_model_path = self.config.ml_model_path
            enable_ml = self.config.enable_ml_detection
            if ml_model_path and not Path(ml_model_path).exists():
                enable_ml = False
                ml_model_path = None

            st.session_state.anomaly_detector = AnomalyDetector(
                enable_statistical=True,
                enable_rules=True,
                enable_ml=enable_ml,
                ml_model_path=ml_model_path,
            )
        if "sniffer" not in st.session_state:
            st.session_state.sniffer = SimulatedSniffer()
        if "is_capturing" not in st.session_state:
            st.session_state.is_capturing = False
        if "refresh_rate" not in st.session_state:
            st.session_state.refresh_rate = 2
        if "_packet_buffer" not in st.session_state:
            st.session_state._packet_buffer = []
        # Convenience aliases
        self.db = st.session_state.db
        self.graph = st.session_state.graph
        self.analyzer = st.session_state.analyzer
        self.anomaly_detector = st.session_state.anomaly_detector
        self.sniffer = st.session_state.sniffer

    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="NetSentinel - Network Analyzer",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Sidebar
        self._render_sidebar()

        # Main content
        st.title("🔍 NetSentinel: Network Traffic Analyzer")
        st.markdown("Real-time network monitoring with ML-powered anomaly detection")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Live Traffic", "🕸️ Network Graph", "⚠️ Alerts", "📈 Analytics"]
        )

        with tab1:
            # Auto-refresh live traffic while capturing
            run_every = (
                st.session_state.refresh_rate if st.session_state.is_capturing else None
            )

            @st.fragment(run_every=run_every)
            def _live_traffic_fragment():
                # Flush any buffered packets to DB before rendering
                self._flush_packet_buffer()
                self._render_live_traffic()

            _live_traffic_fragment()

        with tab2:
            self._render_network_graph()

        with tab3:
            self._render_alerts()

        with tab4:
            self._render_analytics()

    def _render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.title("Controls")

        # Capture controls
        st.sidebar.subheader("Packet Capture")

        def _toggle_capture():
            st.session_state.is_capturing = not st.session_state.is_capturing

        clicked = st.sidebar.button(
            (
                "▶️ Start Capture"
                if not st.session_state.is_capturing
                else "⏹️ Stop Capture"
            ),
            key="capture_toggle",
            on_click=_toggle_capture,
        )

        if clicked:
            # on_click already toggled is_capturing before this rerun,
            # so the button label above is already correct.
            if st.session_state.is_capturing:
                self._register_capture_callback()
                self.sniffer.start()
                st.sidebar.success("Capture started!")
            else:
                self.sniffer.stop()
                self._flush_packet_buffer()
                st.sidebar.info("Capture stopped.")

        st.sidebar.metric("Packets Captured", self.sniffer.stats["total_packets"])

        # Settings
        st.sidebar.subheader("Settings")

        st.session_state.refresh_rate = st.sidebar.slider(
            "Refresh Rate (seconds)",
            min_value=1,
            max_value=10,
            value=st.session_state.refresh_rate,
        )

        if "max_nodes" not in st.session_state:
            st.session_state.max_nodes = 50
        st.session_state.max_nodes = st.sidebar.slider(
            "Max Graph Nodes",
            min_value=10,
            max_value=100,
            value=st.session_state.max_nodes,
        )

        # Database stats
        st.sidebar.subheader("Database")
        db_stats = self.db.get_database_stats()
        st.sidebar.metric("Traffic Logs", db_stats["traffic_log_count"])
        st.sidebar.metric("Alerts", db_stats["alert_count"])
        st.sidebar.metric("DB Size", f"{db_stats['database_size_mb']} MB")

        if st.sidebar.button("🗑️ Clear Old Data", key="clear_old_data"):
            deleted = self.db.cleanup_old_logs(days=7)
            st.sidebar.success(f"Deleted {deleted} old records")

    def _register_capture_callback(self):
        """Register a packet callback that persists to DB and runs anomaly detection.

        The callback runs on the sniffer's background thread, so it must not
        touch st.session_state.  DB operations are thread-safe because
        DatabaseManager uses thread-local connections.
        """
        # Remove any leftover callbacks from a previous capture session
        # to avoid duplicate processing on Start→Stop→Start cycles.
        self.sniffer._callbacks.clear()

        db = self.db
        detector = self.anomaly_detector
        # Plain list; list.append is thread-safe in CPython (GIL).
        buf = st.session_state._packet_buffer

        def on_packet(packet_info):
            buf.append(packet_info)

            # Check for anomalies
            anomalies = detector.check_packet(packet_info)
            for anomaly in anomalies:
                alert = detector.create_alert(anomaly)
                db.insert_alert(alert)

            # Bulk insert to DB periodically.
            # Flush inline using captured `buf`/`db` — do NOT call
            # self._flush_packet_buffer() because this callback runs on
            # the sniffer's daemon thread and must not touch st.session_state.
            if len(buf) >= 50:
                packets, buf[:] = list(buf), []
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
                    for p in packets
                ]
                db.insert_traffic_logs_bulk(logs)

        self.sniffer.add_callback(on_packet)

    def _flush_packet_buffer(self):
        """Flush any remaining buffered packets to the database."""
        buf = st.session_state._packet_buffer
        if not buf:
            return
        # Snapshot and clear atomically (list slice + clear is safe under GIL)
        packets, buf[:] = list(buf), []
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
            for p in packets
        ]
        self.db.insert_traffic_logs_bulk(logs)

    def _render_live_traffic(self):
        """Render live traffic view."""
        st.subheader("Live Traffic Monitor")

        col1, col2, col3, col4 = st.columns(4)

        sniffer_stats = self.sniffer.get_stats()

        # Use sniffer stats if capturing, otherwise show database totals
        if sniffer_stats["total_packets"] > 0:
            display_stats = sniffer_stats
        else:
            db_traffic = self.db.get_traffic_statistics(minutes=60)
            display_stats = {
                "total_packets": db_traffic["packet_count"],
                "total_bytes": db_traffic["total_bytes"],
                "tcp_packets": db_traffic["protocols"].get("TCP", 0),
                "udp_packets": db_traffic["protocols"].get("UDP", 0),
                "icmp_packets": db_traffic["protocols"].get("ICMP", 0),
                "other_packets": db_traffic["protocols"].get("Other", 0),
            }

        with col1:
            st.metric("Total Packets", display_stats["total_packets"])
        with col2:
            st.metric("Total Bytes", self._format_bytes(display_stats["total_bytes"]))
        with col3:
            st.metric("TCP Packets", display_stats["tcp_packets"])
        with col4:
            st.metric("UDP Packets", display_stats["udp_packets"])

        # Recent packets table
        st.subheader("Recent Packets")

        recent_logs = self.db.get_recent_traffic(limit=20)

        if recent_logs:
            data = []
            for log in recent_logs:
                data.append(
                    {
                        "Time": log.timestamp.strftime("%H:%M:%S"),
                        "Source": f"{log.src_ip}:{log.src_port}",
                        "Destination": f"{log.dst_ip}:{log.dst_port}",
                        "Protocol": log.protocol,
                        "Size": log.packet_size,
                        "Anomaly": "⚠️" if log.is_anomaly else "✓",
                    }
                )

            st.dataframe(data, use_container_width=True)
        else:
            st.info("No packets captured yet. Start capture to see traffic.")

        # Protocol distribution chart
        st.subheader("Protocol Distribution")

        protocol_data = {
            "TCP": display_stats["tcp_packets"],
            "UDP": display_stats["udp_packets"],
            "ICMP": display_stats["icmp_packets"],
            "Other": display_stats.get("other_packets", 0),
        }

        if sum(protocol_data.values()) > 0:
            st.bar_chart(pd.Series(protocol_data, name="Packets"))

        # Live packet volume over time (time-series chart)
        st.subheader("Packet Volume Over Time")

        volume_data = self._get_packet_volume_timeseries()
        if volume_data is not None:
            st.line_chart(volume_data)
        else:
            st.info("Not enough data for time-series chart. Capture more traffic.")

    def _render_network_graph(self):
        """Render network topology graph."""
        st.subheader("Network Topology")

        # Build graph from recent connections
        max_nodes = st.session_state.get("max_nodes", 50)
        connections = self.db.get_connection_pairs(limit=max_nodes)

        self.graph.clear()
        for conn in connections:
            self.graph.add_edge(
                conn.src_ip,
                conn.dst_ip,
                conn.byte_count,
                conn.protocols[0] if conn.protocols else "TCP",
            )

        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)

        graph_stats = self.graph.get_statistics()

        with col1:
            st.metric("Nodes (IPs)", graph_stats["vertex_count"])
        with col2:
            st.metric("Edges (Connections)", graph_stats["edge_count"])
        with col3:
            st.metric("Avg Degree", graph_stats["avg_degree"])
        with col4:
            st.metric("Components", graph_stats["components"])

        # Top talkers
        st.subheader("Top Talkers (by Degree)")

        top_nodes = self.graph.get_top_vertices_by_degree(10)

        if top_nodes:
            data = []
            for vertex in top_nodes:
                data.append(
                    {
                        "IP Address": vertex.id,
                        "Type": "Internal" if vertex.is_internal else "External",
                        "In-Degree": vertex.in_degree,
                        "Out-Degree": vertex.out_degree,
                        "Total Degree": vertex.degree,
                        "Bytes": self._format_bytes(vertex.total_bytes),
                    }
                )

            st.dataframe(data, use_container_width=True)

        # Graph visualization (using Streamlit's built-in)
        st.subheader("Connection Graph")

        if connections:
            # Render interactive graph using pyvis via vis.js
            try:
                vis_data = self.graph.to_vis_js_format()
                nodes_js = json.dumps(vis_data["nodes"])
                edges_js = json.dumps(vis_data["edges"])

                graph_html = f"""
                <html>
                <head>
                    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                    <style>
                        #network {{ width: 100%; height: 500px; border: 1px solid #ccc; background: #1e1e2e; }}
                    </style>
                </head>
                <body>
                    <div id="network"></div>
                    <script>
                        var nodes = new vis.DataSet({nodes_js});
                        var edges = new vis.DataSet({edges_js});
                        var container = document.getElementById('network');
                        var data = {{ nodes: nodes, edges: edges }};
                        var options = {{
                            nodes: {{ shape: 'dot', font: {{ color: '#ffffff' }} }},
                            edges: {{ arrows: 'to', color: {{ color: '#888888' }}, smooth: {{ type: 'continuous' }} }},
                            physics: {{ stabilization: {{ iterations: 100 }}, barnesHut: {{ gravitationalConstant: -3000 }} }},
                            interaction: {{ hover: true, tooltipDelay: 200 }}
                        }};
                        var network = new vis.Network(container, data, options);
                    </script>
                </body>
                </html>
                """
                st.components.v1.html(graph_html, height=520)
            except Exception as e:
                st.warning(f"Could not render interactive graph: {e}")
                st.json(
                    {"nodes": len(self.graph.vertices), "edges": len(self.graph.edges)}
                )
        else:
            st.info("No connections to display. Start capture to see network topology.")

        # Analysis results
        with st.expander("Graph Analysis"):
            analysis = self.analyzer.get_analysis_report()

            st.write("**Degree Distribution:**")
            st.json(analysis["degree_distribution"])

            st.write("**Potential Port Scanners:**")
            if analysis["potential_scanners"]:
                st.warning(
                    f"Found {len(analysis['potential_scanners'])} potential scanners"
                )
                for ip, count in analysis["potential_scanners"][:5]:
                    st.write(f"  - {ip}: {count} destinations")
            else:
                st.success("No port scanners detected")

            st.write("**Hub Nodes:**")
            for ip, details in analysis["hub_nodes"][:5]:
                st.write(f"  - {ip}: degree={details['degree']}")

    def _render_alerts(self):
        """Render alerts view."""
        st.subheader("Security Alerts")

        # Alert statistics
        col1, col2, col3 = st.columns(3)

        detector_stats = self.anomaly_detector.get_statistics()

        # Use live detector stats if packets have been analyzed, otherwise use DB counts
        if detector_stats["total_packets_checked"] > 0:
            anomaly_count = detector_stats["anomalies_detected"]
            anomaly_rate = detector_stats["anomaly_rate"] * 100
            packets_checked = detector_stats["total_packets_checked"]
        else:
            db_stats = self.db.get_database_stats()
            anomaly_count = db_stats["alert_count"]
            packets_checked = db_stats["traffic_log_count"]
            anomaly_rate = (anomaly_count / max(packets_checked, 1)) * 100

        with col1:
            st.metric("Total Anomalies", anomaly_count)
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        with col3:
            st.metric("Packets Analyzed", packets_checked)

        # Recent alerts
        st.subheader("Recent Alerts")

        alerts = self.db.get_recent_alerts(limit=20)

        if alerts:
            for alert in alerts:
                severity_color = {
                    1: "🟢",  # LOW
                    2: "🟡",  # MEDIUM
                    3: "🟠",  # HIGH
                    4: "🔴",  # CRITICAL
                }

                icon = severity_color.get(alert.severity.value, "⚪")

                with st.expander(
                    f"{icon} {alert.alert_type.value.upper()} - {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                ):
                    st.write(f"**Source IP:** {alert.src_ip}")
                    st.write(f"**Destination IP:** {alert.dst_ip}")
                    st.write(f"**Description:** {alert.description}")
                    st.write(
                        f"**Status:** {'Resolved ✓' if alert.is_resolved else 'Active'}"
                    )

                    if not alert.is_resolved and st.button(
                        f"Mark Resolved", key=f"resolve_{alert.id}"
                    ):
                        self.db.resolve_alert(alert.id)
                        st.rerun()
        else:
            st.success("No alerts! Network looks healthy.")

        # Flagged IPs - consolidated list
        st.subheader("Flagged IP Addresses")
        flagged_ips = self._get_flagged_ips()
        if flagged_ips:
            st.dataframe(flagged_ips, use_container_width=True)
        else:
            st.success("No flagged IPs detected.")

        # Anomaly detection settings
        with st.expander("Detection Settings"):
            st.write("**Enabled Detectors:**")
            for detector, enabled in detector_stats["detectors_enabled"].items():
                st.checkbox(detector.title(), value=enabled, disabled=True)

    def _render_analytics(self):
        """Render analytics view."""
        st.subheader("Traffic Analytics")

        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            options=[5, 15, 30, 60],
            format_func=lambda x: f"Last {x} minutes",
        )

        # Get statistics
        stats = self.db.get_traffic_statistics(minutes=time_range)

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Packets", stats["packet_count"])
        with col2:
            st.metric("Total Bytes", self._format_bytes(stats["total_bytes"]))
        with col3:
            st.metric("Unique Sources", stats["unique_sources"])
        with col4:
            st.metric("Unique Destinations", stats["unique_destinations"])

        # Protocol breakdown
        st.subheader("Protocol Breakdown")

        if stats["protocols"]:
            st.bar_chart(pd.Series(stats["protocols"], name="Packets"))

        # Top sources
        st.subheader("Top Traffic Sources")

        if stats["top_sources"]:
            data = []
            for src in stats["top_sources"]:
                data.append(
                    {
                        "IP Address": src["ip"],
                        "Packets": src["packets"],
                        "Bytes": self._format_bytes(src["bytes"]),
                    }
                )

            st.dataframe(data, use_container_width=True)

        # Anomaly trend
        st.subheader("Anomaly Trend")

        anomaly_count = stats["anomaly_count"]
        normal_count = stats["packet_count"] - anomaly_count

        st.write(
            f"**Normal Traffic:** {normal_count} packets ({100 * normal_count / max(stats['packet_count'], 1):.1f}%)"
        )
        st.write(
            f"**Anomalous Traffic:** {anomaly_count} packets ({100 * anomaly_count / max(stats['packet_count'], 1):.1f}%)"
        )

        # IP analysis
        st.subheader("IP Analysis")

        ip_address = st.text_input("Enter IP address to analyze:")

        if ip_address:
            ip_stats = self.db.get_ip_statistics(ip_address)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Outbound Traffic:**")
                st.metric("Packets Sent", ip_stats.total_packets_sent)
                st.metric("Bytes Sent", self._format_bytes(ip_stats.total_bytes_sent))
                st.metric("Unique Destinations", ip_stats.unique_destinations)

            with col2:
                st.write("**Inbound Traffic:**")
                st.metric("Packets Received", ip_stats.total_packets_received)
                st.metric(
                    "Bytes Received", self._format_bytes(ip_stats.total_bytes_received)
                )
                st.metric("Unique Sources", ip_stats.unique_sources)

            if ip_stats.protocols_used:
                st.write(f"**Protocols Used:** {', '.join(ip_stats.protocols_used)}")

    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes into human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f} TB"

    def _get_packet_volume_timeseries(self) -> Optional["pd.DataFrame"]:
        """
        Get packet counts bucketed by time interval for a live line chart.

        Returns DataFrame with time-string index and 'Packets' column,
        or None if insufficient data.
        """
        try:
            recent_logs = self.db.get_recent_traffic(limit=500)
            if len(recent_logs) < 2:
                return None

            # Bucket by minute
            buckets: Dict[str, int] = {}
            for log in recent_logs:
                minute_key = log.timestamp.strftime("%H:%M")
                buckets[minute_key] = buckets.get(minute_key, 0) + 1

            if len(buckets) < 2:
                return None

            # Sort by time and return as DataFrame with time index
            sorted_keys = sorted(buckets.keys())
            return pd.DataFrame(
                {"Packets": [buckets[k] for k in sorted_keys]},
                index=sorted_keys,
            )
        except Exception as e:
            logging.getLogger("netsentinel").warning("Time-series chart failed: %s", e)
            return None

    def _get_flagged_ips(self) -> Optional[list]:
        """
        Build a consolidated list of all flagged/anomalous IP addresses.

        Aggregates from:
        - Alerts (source and destination IPs)
        - Anomalous traffic logs
        - Graph analysis (port scanners, DDoS targets)
        """
        flagged: Dict[str, Dict[str, Any]] = {}

        # From alerts
        alerts = self.db.get_recent_alerts(limit=100, unresolved_only=False)
        for alert in alerts:
            for ip in [alert.src_ip, alert.dst_ip]:
                if ip:
                    if ip not in flagged:
                        flagged[ip] = {
                            "IP Address": ip,
                            "Alert Count": 0,
                            "Highest Severity": 0,
                            "Reasons": set(),
                        }
                    flagged[ip]["Alert Count"] += 1
                    flagged[ip]["Highest Severity"] = max(
                        flagged[ip]["Highest Severity"], alert.severity.value
                    )
                    flagged[ip]["Reasons"].add(alert.alert_type.value)

        # From graph analysis (scanners, DDoS targets)
        try:
            analysis = self.analyzer.get_analysis_report()
            for ip, count in analysis.get("potential_scanners", []):
                if ip not in flagged:
                    flagged[ip] = {
                        "IP Address": ip,
                        "Alert Count": 0,
                        "Highest Severity": 0,
                        "Reasons": set(),
                    }
                flagged[ip]["Highest Severity"] = max(
                    flagged[ip]["Highest Severity"], 2
                )
                flagged[ip]["Reasons"].add("port_scan")

            for ip, count in analysis.get("potential_ddos_targets", []):
                if ip not in flagged:
                    flagged[ip] = {
                        "IP Address": ip,
                        "Alert Count": 0,
                        "Highest Severity": 0,
                        "Reasons": set(),
                    }
                flagged[ip]["Highest Severity"] = max(
                    flagged[ip]["Highest Severity"], 3
                )
                flagged[ip]["Reasons"].add("ddos_suspect")
        except Exception as e:
            logging.getLogger("netsentinel").warning(
                "Graph analysis failed in flagged IPs: %s", e
            )

        if not flagged:
            return None

        # Convert to list for display
        severity_labels = {0: "-", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
        result = []
        for ip_data in sorted(
            flagged.values(), key=lambda x: x["Highest Severity"], reverse=True
        ):
            result.append(
                {
                    "IP Address": ip_data["IP Address"],
                    "Alert Count": ip_data["Alert Count"],
                    "Severity": severity_labels.get(
                        ip_data["Highest Severity"], "UNKNOWN"
                    ),
                    "Reasons": ", ".join(sorted(ip_data["Reasons"])),
                }
            )

        return result


def run_dashboard():
    """Entry point for running the dashboard."""
    app = DashboardApp()
    app.run()


# Module-level function for Streamlit
def main():
    """Main function for Streamlit."""
    run_dashboard()


if __name__ == "__main__":
    main()
