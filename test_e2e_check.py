"""Quick end-to-end pipeline verification."""

import sys, os, time

sys.path.insert(0, ".")

# 1. Test SimulatedSniffer generates packets
from netsentinel.sniffer import SimulatedSniffer

sniffer = SimulatedSniffer()
captured = []
sniffer.add_callback(lambda p: captured.append(p))
sniffer.start(count=50)
time.sleep(4)
sniffer.stop()
print(f"[1] Sniffer captured: {len(captured)} packets")
assert len(captured) > 0, "FAIL: No packets captured"
p = captured[0]
print(
    f"    Sample: {p.src_ip} -> {p.dst_ip} proto={p.protocol} size={p.size} flags={p.flags}"
)

# 2. Test DB stores and retrieves
from netsentinel.database import DatabaseManager, TrafficLog

db = DatabaseManager("data/test_e2e.db")
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
    for p in captured[:20]
]
inserted = db.insert_traffic_logs_bulk(logs)
print(f"[2] DB inserted: {inserted} rows")
assert inserted == 20, f"FAIL: Expected 20 inserts, got {inserted}"
retrieved = db.get_recent_traffic(limit=5)
print(f"    Retrieved: {len(retrieved)} rows")
assert len(retrieved) == 5

# 3. Test Graph builds
from netsentinel.visualizer import NetworkGraph

graph = NetworkGraph()
for p in captured[:20]:
    graph.add_edge(p.src_ip, p.dst_ip, p.size, p.protocol)
stats = graph.get_statistics()
print(f"[3] Graph: {stats['vertex_count']} nodes, {stats['edge_count']} edges")
assert stats["vertex_count"] > 0

# 4. Test Feature Extraction
from netsentinel.ml import FeatureExtractor

fe = FeatureExtractor()
features = fe.extract_features(captured[:20])
vec = features.to_vector()
print(
    f"[4] Features: {len(vec)} dims, avg_pkt_size={vec[0]:.1f}, tcp_ratio={vec[7]:.2f}"
)
assert len(vec) == 22, f"FAIL: Expected 22 features, got {len(vec)}"
assert vec[0] > 0, "FAIL: avg_packet_size should be > 0"

# 5. Test Anomaly Detector (statistical + rule)
from netsentinel.ml import AnomalyDetector

detector = AnomalyDetector(enable_statistical=True, enable_rules=True, enable_ml=False)
total_anomalies = 0
for p in captured:
    anomalies = detector.check_packet(p)
    total_anomalies += len(anomalies)
det_stats = detector.get_statistics()
print(
    f"[5] Anomaly Detector: checked={det_stats['total_packets_checked']}, anomalies={det_stats['anomalies_detected']}"
)
assert det_stats["total_packets_checked"] == len(captured)

# 6. Test ML model loading + prediction
from netsentinel.ml.anomaly_detector import MLDetector

ml = MLDetector(model_path="models/decision_tree.pkl")
print(f"[6] ML model loaded: {ml.model is not None}, type={ml.model_type}")
assert ml.model is not None, "FAIL: ML model did not load"
for p in captured[:15]:
    ml.add_packet(p)
result = ml.check_anomaly()
print(
    f"    ML prediction result: {'anomaly detected' if result else 'normal (no anomaly)'}"
)

# 7. Test full AnomalyDetector with ML enabled
detector_ml = AnomalyDetector(
    enable_statistical=True,
    enable_rules=True,
    enable_ml=True,
    ml_model_path="models/decision_tree.pkl",
)
assert detector_ml.ml_detector is not None, "FAIL: ML detector not enabled"
assert (
    detector_ml.ml_detector.model is not None
), "FAIL: ML model not loaded in full detector"
ml_anomalies = 0
for i, p in enumerate(captured):
    results = detector_ml.check_packet(p)
    ml_results = [r for r in results if r.detection_method.startswith("ml:")]
    ml_anomalies += len(ml_results)
print(
    f"[7] Full detector with ML: {ml_anomalies} ML-based anomalies out of {len(captured)} packets"
)

# 8. Verify training pipeline produces usable models
from netsentinel.ml.feature_extractor import FeatureScaler

windowed = fe.extract_features_windowed(captured, window_seconds=2)
print(f"[8] Windowed features: {len(windowed)} windows from {len(captured)} packets")
X = [f.to_vector() for f in windowed]
assert len(X) > 0, "FAIL: No feature windows extracted"
for vec in X:
    assert len(vec) == 22
    assert all(not (v != v) for v in vec), "FAIL: NaN in feature vector"  # NaN check

# 9. Verify scaler round-trips
scaler = FeatureScaler()
scaler.fit(X)
scaled = scaler.transform_minmax(X[0])
assert len(scaled) == 22
assert all(
    0.0 <= v <= 1.0 + 1e-9 for v in scaled
), f"FAIL: Scaled values out of [0,1]: {scaled}"
print(f"[9] Scaler: min-max scaled values all in [0,1] range")

# 10. Verify analyze mode data
conn_pairs = db.get_connection_pairs(limit=10)
print(f"[10] Connection pairs from DB: {len(conn_pairs)}")
traffic_stats = db.get_traffic_statistics(minutes=60)
print(
    f"     Traffic stats: {traffic_stats['packet_count']} packets, {traffic_stats['unique_sources']} sources"
)

# Cleanup
db.close()
os.remove("data/test_e2e.db")
print("\n=== ALL 10 CHECKS PASSED ===")
