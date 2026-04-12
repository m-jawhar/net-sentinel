#!/usr/bin/env python3
"""
NetSentinel: AI-Driven Network Traffic Analyzer & Anomaly Detector

Main entry point for the application.

Usage:
    python main.py --mode dashboard    # Start the web dashboard
    python main.py --mode capture      # Start packet capture (CLI)
    python main.py --mode train        # Train ML model
    python main.py --mode analyze      # Analyze captured data
"""

import argparse
import sys
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from netsentinel.sniffer import PacketSniffer, SimulatedSniffer
from netsentinel.database import DatabaseManager, TrafficLog
from netsentinel.visualizer import NetworkGraph, GraphAnalyzer
from netsentinel.ml import AnomalyDetector, FeatureExtractor, ModelTrainer
from netsentinel.utils import Config, setup_logger


def run_dashboard(config: Config):
    """Run the Streamlit dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent / "netsentinel" / "dashboard" / "app.py"

    print(f"Starting NetSentinel Dashboard on port {config.dashboard_port}...")
    print(f"Open http://localhost:{config.dashboard_port} in your browser")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(config.dashboard_port),
            "--server.headless",
            "true",
        ]
    )


def run_capture(config: Config):
    """Run packet capture in CLI mode."""
    logger = setup_logger(level=config.log_level, log_file=config.log_file)
    db = DatabaseManager(config.database_path)
    graph = NetworkGraph()

    # Auto-detect ML model path if models directory has a trained model
    ml_model_path = config.ml_model_path
    enable_ml = config.enable_ml_detection
    if not ml_model_path:
        default_model = Path("models/decision_tree.pkl")
        if default_model.exists():
            ml_model_path = str(default_model)
            enable_ml = True

    anomaly_detector = AnomalyDetector(
        enable_statistical=config.enable_statistical_detection,
        enable_rules=config.enable_rule_detection,
        enable_ml=enable_ml,
        ml_model_path=ml_model_path,
    )

    # Use simulated sniffer for demo (real sniffer needs admin privileges)
    use_simulation = True  # Set to False for real capture (requires admin)

    if use_simulation:
        print("Using SIMULATED traffic (no admin privileges required)")
        print("For real capture, run as administrator and set use_simulation=False")
        sniffer = SimulatedSniffer(interface=config.sniffer_interface)
    else:
        sniffer = PacketSniffer(interface=config.sniffer_interface)

    # Packet processing callback
    packet_buffer = []
    buffer_size = 100

    def process_packet(packet_info):
        """Process each captured packet."""
        nonlocal packet_buffer

        # Add to buffer
        packet_buffer.append(packet_info)

        # Update graph
        graph.add_edge(
            packet_info.src_ip,
            packet_info.dst_ip,
            packet_info.size,
            packet_info.protocol,
        )

        # Check for anomalies
        anomalies = anomaly_detector.check_packet(packet_info)

        for anomaly in anomalies:
            logger.log_anomaly(
                anomaly.related_ips[0] if anomaly.related_ips else "unknown",
                anomaly.detection_method,
                anomaly.score,
                str(anomaly.details),
            )

            # Create and store alert
            alert = anomaly_detector.create_alert(anomaly)
            db.insert_alert(alert)

        # Bulk insert to database periodically
        if len(packet_buffer) >= buffer_size:
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
                for p in packet_buffer
            ]
            db.insert_traffic_logs_bulk(logs)
            packet_buffer.clear()

    # Register callback
    sniffer.add_callback(process_packet)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping capture...")
        sniffer.stop()

        # Save remaining packets
        if packet_buffer:
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
                for p in packet_buffer
            ]
            db.insert_traffic_logs_bulk(logs)

        # Print summary
        stats = sniffer.get_stats()
        graph_stats = graph.get_statistics()
        detector_stats = anomaly_detector.get_statistics()

        print("\n" + "=" * 50)
        print("CAPTURE SUMMARY")
        print("=" * 50)
        print(f"Total Packets: {stats['total_packets']}")
        print(f"Total Bytes: {stats['total_bytes']}")
        print(
            f"TCP: {stats['tcp_packets']} | UDP: {stats['udp_packets']} | ICMP: {stats['icmp_packets']}"
        )
        print(f"Unique IPs: {graph_stats['vertex_count']}")
        print(f"Connections: {graph_stats['edge_count']}")
        print(f"Anomalies Detected: {detector_stats['anomalies_detected']}")
        print("=" * 50)

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start capture
    print("=" * 50)
    print("NetSentinel - Network Traffic Capture")
    print("=" * 50)
    print(f"Interface: {config.sniffer_interface or 'default'}")
    print(f"Database: {config.database_path}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    sniffer.start()

    # Display live statistics
    while sniffer.is_running():
        time.sleep(2)
        stats = sniffer.get_stats()
        print(
            f"\r[{datetime.now().strftime('%H:%M:%S')}] "
            f"Packets: {stats['total_packets']} | "
            f"Bytes: {stats['total_bytes']} | "
            f"Anomalies: {anomaly_detector.get_statistics()['anomalies_detected']}",
            end="",
            flush=True,
        )


def run_train(config: Config, nsl_kdd_path: Optional[str] = None):
    """Train ML model on captured data or an NSL-KDD dataset file."""
    print("=" * 50)
    print("NetSentinel - ML Model Training")
    print("=" * 50)

    db = DatabaseManager(config.database_path)
    feature_extractor = FeatureExtractor()
    trainer = ModelTrainer()

    # ---- NSL-KDD benchmark dataset path ----
    if nsl_kdd_path:
        from netsentinel.ml.dataset_loader import NSLKDDLoader

        loader = NSLKDDLoader()
        info = loader.get_dataset_info()
        print(f"\nUsing standard benchmark dataset: {info['name']}")
        print(f"File: {nsl_kdd_path}")
        X, y = loader.load(nsl_kdd_path, binary=True)
        print(
            f"Loaded {len(X)} records  |  Normal: {y.count('normal')}  "
            + f"Anomaly: {y.count('anomaly')}"
        )
        # Jump directly to model training with these vectors
        return _train_models(trainer, X, y, feature_list=None)

    # Get training data
    print("Loading traffic data...")
    logs = db.get_recent_traffic(limit=10000)

    if len(logs) < 100:
        print("Insufficient data for training. Need at least 100 traffic logs.")
        print(f"Current data: {len(logs)} logs")
        print("Run capture mode first to collect data.")
        return

    print(f"Loaded {len(logs)} traffic logs")

    # Convert to PacketInfo and extract features
    from netsentinel.sniffer.packet_parser import PacketInfo

    packets = [
        PacketInfo(
            timestamp=log.timestamp,
            src_ip=log.src_ip,
            dst_ip=log.dst_ip,
            src_port=log.src_port,
            dst_port=log.dst_port,
            protocol=log.protocol,
            size=log.packet_size,
            flags=log.flags,
            ttl=log.ttl,
        )
        for log in logs
    ]

    # Extract windowed features
    print("Extracting features...")
    feature_list = feature_extractor.extract_features_windowed(
        packets, window_seconds=5
    )

    # Create labels (simple heuristic - large packets or high frequency = anomaly)
    # In real use, you'd have labeled data
    print("Generating training labels...")
    X = [f.to_vector() for f in feature_list]
    y = []

    for f in feature_list:
        # Simple anomaly heuristics for demo
        is_anomaly = (
            f.max_packet_size > 10000
            or f.packets_per_second > 50
            or f.unique_destinations > 20
            or f.syn_ratio > 0.8
        )
        y.append("anomaly" if is_anomaly else "normal")

    print(f"Training samples: {len(X)}")
    print(f"Normal: {y.count('normal')} | Anomaly: {y.count('anomaly')}")

    if len(set(y)) < 2:
        print("Warning: Only one class in data. Adding synthetic anomaly samples...")
        # Add some synthetic anomalies
        from copy import deepcopy

        for i in range(10):
            anomaly_features = X[i % len(X)].copy()
            anomaly_features[0] *= 10  # Large avg packet size
            anomaly_features[6] *= 20  # High packets per second
            X.append(anomaly_features)
            y.append("anomaly")

    # Train Decision Tree
    print("\nTraining Decision Tree...")
    dt_model, dt_result = trainer.train_decision_tree(
        X,
        y,
        feature_names=feature_list[0].feature_names() if feature_list else None,
        max_depth=10,
    )

    print(f"  Accuracy: {dt_result.accuracy:.4f}")
    print(f"  Precision: {dt_result.precision:.4f}")
    print(f"  Recall: {dt_result.recall:.4f}")
    print(f"  F1 Score: {dt_result.f1_score:.4f}")

    if dt_result.feature_importance:
        print("  Top Features:")
        sorted_features = sorted(
            dt_result.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for name, importance in sorted_features[:5]:
            print(f"    - {name}: {importance:.4f}")

    # Train K-Means
    print("\nTraining K-Means Clustering...")
    kmeans_model, kmeans_stats = trainer.train_kmeans(X, n_clusters=2)
    print(f"  Inertia: {kmeans_stats['inertia']:.2f}")
    print(f"  Cluster sizes: {kmeans_stats['cluster_sizes']}")

    # Train Naive Bayes
    print("\nTraining Naive Bayes Classifier...")
    nb_model, nb_result = trainer.train_naive_bayes(X, y)
    print(f"  Accuracy: {nb_result.accuracy:.4f}")
    print(f"  Precision: {nb_result.precision:.4f}")
    print(f"  Recall: {nb_result.recall:.4f}")
    print(f"  F1 Score: {nb_result.f1_score:.4f}")

    # Train Linear Regression (predicts anomaly score)
    print("\nTraining Linear Regression...")
    # Generate continuous anomaly scores as regression targets
    y_scores = []
    for f in feature_list:
        score = 0.0
        if f.max_packet_size > 10000:
            score += 0.3
        if f.packets_per_second > 50:
            score += 0.3
        if f.unique_destinations > 20:
            score += 0.2
        if f.syn_ratio > 0.8:
            score += 0.2
        y_scores.append(min(score, 1.0))

    # Extend scores for any synthetic samples added earlier
    while len(y_scores) < len(X):
        y_scores.append(0.9)  # Synthetic anomalies get high score

    lr_model, lr_stats = trainer.train_linear_regression(
        X,
        y_scores,
        feature_names=feature_list[0].feature_names() if feature_list else None,
    )
    print(f"  MSE: {lr_stats['mse']:.6f}")
    print(f"  RMSE: {lr_stats['rmse']:.6f}")
    print(f"  R²: {lr_stats['r_squared']:.4f}")
    print("  Top Coefficients:")
    sorted_coeffs = sorted(
        lr_stats["coefficients"].items(), key=lambda x: abs(x[1]), reverse=True
    )
    for name, coeff in sorted_coeffs[:5]:
        print(f"    - {name}: {coeff:.4f}")

    # Fit scaler on training data so it can be saved with each model
    from netsentinel.ml.feature_extractor import FeatureScaler

    scaler = FeatureScaler()
    scaler.fit(X)

    # Save models (with scaler)
    print("\nSaving models...")
    trainer.save_model(dt_model, "decision_tree", scaler=scaler)
    trainer.save_model(kmeans_model, "kmeans", scaler=scaler)
    trainer.save_model(lr_model, "linear_regression", scaler=scaler)
    trainer.save_model(nb_model, "naive_bayes", scaler=scaler)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Models saved to: {trainer.models_dir}")
    print("=" * 50)


def _train_models(
    trainer: ModelTrainer,
    X: List[List[float]],
    y: List[str],
    feature_list=None,
):
    """Shared helper that trains all four models and saves them."""
    print(f"\nTraining samples: {len(X)}")
    print(f"Normal: {y.count('normal')} | Anomaly: {y.count('anomaly')}")

    if len(set(y)) < 2:
        print("Warning: Only one class. Adding synthetic anomaly samples...")
        for i in range(10):
            anomaly_features = X[i % len(X)].copy()
            anomaly_features[0] *= 10
            anomaly_features[6] *= 20
            X.append(anomaly_features)
            y.append("anomaly")

    feature_names = feature_list[0].feature_names() if feature_list else None

    # Decision Tree
    print("\nTraining Decision Tree...")
    dt_model, dt_result = trainer.train_decision_tree(
        X,
        y,
        feature_names=feature_names,
        max_depth=10,
    )
    print(f"  Accuracy: {dt_result.accuracy:.4f}  F1: {dt_result.f1_score:.4f}")

    # K-Means
    print("Training K-Means Clustering...")
    kmeans_model, kmeans_stats = trainer.train_kmeans(X, n_clusters=2)
    print(f"  Inertia: {kmeans_stats['inertia']:.2f}")

    # Naive Bayes
    print("Training Naive Bayes Classifier...")
    nb_model, nb_result = trainer.train_naive_bayes(X, y)
    print(f"  Accuracy: {nb_result.accuracy:.4f}  F1: {nb_result.f1_score:.4f}")

    # Linear Regression
    print("Training Linear Regression...")
    y_scores = [0.9 if label == "anomaly" else 0.1 for label in y]
    lr_model, lr_stats = trainer.train_linear_regression(
        X,
        y_scores,
        feature_names=feature_names,
    )
    print(f"  R²: {lr_stats['r_squared']:.4f}")

    # Fit scaler on training data so it can be saved with each model
    from netsentinel.ml.feature_extractor import FeatureScaler

    scaler = FeatureScaler()
    scaler.fit(X)

    # Save (with scaler)
    print("\nSaving models...")
    trainer.save_model(dt_model, "decision_tree", scaler=scaler)
    trainer.save_model(kmeans_model, "kmeans", scaler=scaler)
    trainer.save_model(nb_model, "naive_bayes", scaler=scaler)
    trainer.save_model(lr_model, "linear_regression", scaler=scaler)
    print("Models saved.")


def run_analyze(config: Config):
    """Analyze captured data."""
    print("=" * 50)
    print("NetSentinel - Traffic Analysis")
    print("=" * 50)

    db = DatabaseManager(config.database_path)

    # Get database stats
    db_stats = db.get_database_stats()
    print(f"\nDatabase: {config.database_path}")
    print(f"Traffic Logs: {db_stats['traffic_log_count']}")
    print(f"Alerts: {db_stats['alert_count']}")
    print(f"Size: {db_stats['database_size_mb']} MB")

    # Get traffic statistics
    print("\n--- Traffic Statistics (last 60 minutes) ---")
    stats = db.get_traffic_statistics(minutes=60)

    print(f"Total Packets: {stats['packet_count']}")
    print(f"Total Bytes: {stats['total_bytes']}")
    print(f"Avg Packet Size: {stats['avg_packet_size']:.1f} bytes")
    print(f"Unique Sources: {stats['unique_sources']}")
    print(f"Unique Destinations: {stats['unique_destinations']}")
    print(f"Anomalies: {stats['anomaly_count']}")

    print("\nProtocol Distribution:")
    for protocol, count in stats["protocols"].items():
        print(f"  {protocol}: {count}")

    print("\nTop Sources:")
    for src in stats["top_sources"][:5]:
        print(f"  {src['ip']}: {src['packets']} packets, {src['bytes']} bytes")

    # Build and analyze graph
    print("\n--- Network Graph Analysis ---")
    graph = NetworkGraph()

    connections = db.get_connection_pairs(limit=500)
    for conn in connections:
        protocol = conn.protocols[0] if conn.protocols else "TCP"
        graph.add_edge(conn.src_ip, conn.dst_ip, conn.byte_count, protocol)

    analyzer = GraphAnalyzer(graph)
    graph_stats = graph.get_statistics()

    print(f"Nodes (IPs): {graph_stats['vertex_count']}")
    print(f"Edges (Connections): {graph_stats['edge_count']}")
    print(f"Graph Density: {graph_stats['density']:.4f}")
    print(f"Average Degree: {graph_stats['avg_degree']:.2f}")
    print(f"Connected Components: {graph_stats['components']}")

    # Detect suspicious activity
    print("\n--- Security Analysis ---")

    scanners = analyzer.detect_port_scanners(threshold=10)
    if scanners:
        print(f"\nPotential Port Scanners ({len(scanners)} found):")
        for ip, count in scanners[:5]:
            print(f"  {ip}: {count} unique destinations")

    ddos_targets = analyzer.detect_ddos_targets(threshold=10)
    if ddos_targets:
        print(f"\nPotential DDoS Targets ({len(ddos_targets)} found):")
        for ip, count in ddos_targets[:5]:
            print(f"  {ip}: {count} unique sources")

    hubs = analyzer.detect_hub_nodes(threshold_factor=2.0)
    if hubs:
        print(f"\nHub Nodes ({len(hubs)} found):")
        for ip, details in hubs[:5]:
            print(
                f"  {ip}: degree={details['degree']}, "
                f"in={details['in_degree']}, out={details['out_degree']}"
            )

    # Recent alerts
    print("\n--- Recent Alerts ---")
    alerts = db.get_recent_alerts(limit=10)

    if alerts:
        for alert in alerts:
            status = "✓" if alert.is_resolved else "!"
            print(
                f"  [{status}] {alert.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"{alert.severity.name} | {alert.alert_type.value}"
            )
    else:
        print("  No alerts found")

    print("\n" + "=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NetSentinel: AI-Driven Network Traffic Analyzer & Anomaly Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode dashboard    Start the web dashboard
  python main.py --mode capture      Start packet capture
  python main.py --mode train        Train ML model
  python main.py --mode analyze      Analyze captured data
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["dashboard", "capture", "train", "analyze"],
        default="dashboard",
        help="Operation mode (default: dashboard)",
    )

    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to configuration file"
    )

    parser.add_argument(
        "--interface",
        "-i",
        type=str,
        default=None,
        help="Network interface for capture",
    )

    parser.add_argument("--db", "-d", type=str, default=None, help="Database path")

    parser.add_argument(
        "--nsl-kdd",
        type=str,
        default=None,
        help="Path to NSL-KDD dataset file (KDDTrain+.txt) for benchmark training",
    )

    parser.add_argument("--port", "-p", type=int, default=None, help="Dashboard port")

    args = parser.parse_args()

    # Load configuration
    config = Config.from_file(args.config) if args.config else Config()

    # Apply command-line overrides
    if args.interface:
        config.sniffer_interface = args.interface
    if args.db:
        config.database_path = args.db
    if args.port:
        config.dashboard_port = args.port

    # Run selected mode
    if args.mode == "dashboard":
        run_dashboard(config)
    elif args.mode == "capture":
        run_capture(config)
    elif args.mode == "train":
        run_train(config, nsl_kdd_path=args.nsl_kdd)
    elif args.mode == "analyze":
        run_analyze(config)


if __name__ == "__main__":
    main()
