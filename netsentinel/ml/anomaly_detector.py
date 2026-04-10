"""
Anomaly Detector Module - Real-time network traffic anomaly detection.

Combines multiple ML approaches:
1. Statistical anomaly detection (Z-score)
2. Rule-based detection
3. Machine Learning models (Decision Tree, K-Means)

Applies Machine Learning concepts (Sem 4):
- Supervised learning (classification)
- Unsupervised learning (clustering)
- Feature-based detection
"""

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..database.models import Alert, AlertSeverity, AlertType
from ..sniffer.packet_parser import PacketInfo
from .feature_extractor import FeatureExtractor, FeatureScaler


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    is_anomaly: bool
    score: float  # 0.0 to 1.0
    detection_method: str
    details: Dict[str, Any]
    timestamp: datetime
    related_ips: List[str]


class StatisticalDetector:
    """
    Statistical anomaly detection using Z-scores and moving averages.

    Detects:
    - Unusual packet sizes (too large or too small)
    - Unusual packet rates (traffic spikes)
    - Unusual port distributions

    Uses concepts from Probability & Statistics.
    """

    def __init__(self, window_size: int = 1000, threshold: float = 3.0):
        """
        Initialize detector.

        Args:
            window_size: Number of samples for calculating statistics
            threshold: Z-score threshold for anomalies (default 3σ)
        """
        self.window_size = window_size
        self.threshold = threshold

        # Running statistics
        self._packet_sizes: List[int] = []
        self._inter_arrival_times: List[float] = []
        self._last_packet_time: Optional[datetime] = None

        # Computed statistics
        self._size_mean = 0.0
        self._size_std = 0.0
        self._iat_mean = 0.0
        self._iat_std = 0.0

    def _update_statistics(self, packet: PacketInfo):
        """Update running statistics with new packet."""
        # Update packet sizes
        self._packet_sizes.append(packet.size)
        if len(self._packet_sizes) > self.window_size:
            self._packet_sizes.pop(0)

        # Update inter-arrival time
        if self._last_packet_time:
            iat = (packet.timestamp - self._last_packet_time).total_seconds()
            self._inter_arrival_times.append(iat)
            if len(self._inter_arrival_times) > self.window_size:
                self._inter_arrival_times.pop(0)

        self._last_packet_time = packet.timestamp

        # Recalculate statistics
        if len(self._packet_sizes) >= 10:
            self._size_mean = sum(self._packet_sizes) / len(self._packet_sizes)
            variance = sum(
                (x - self._size_mean) ** 2 for x in self._packet_sizes
            ) / len(self._packet_sizes)
            self._size_std = math.sqrt(variance) if variance > 0 else 1.0

        if len(self._inter_arrival_times) >= 10:
            self._iat_mean = sum(self._inter_arrival_times) / len(
                self._inter_arrival_times
            )
            variance = sum(
                (x - self._iat_mean) ** 2 for x in self._inter_arrival_times
            ) / len(self._inter_arrival_times)
            self._iat_std = math.sqrt(variance) if variance > 0 else 1.0

    def _z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate Z-score."""
        if std < 1e-6:
            return 0.0
        return abs(value - mean) / std

    def check_packet(self, packet: PacketInfo) -> Optional[AnomalyResult]:
        """
        Check a packet for anomalies.

        Returns AnomalyResult if anomaly detected, None otherwise.
        """
        self._update_statistics(packet)

        # Need enough samples for statistics
        if len(self._packet_sizes) < 20:
            return None

        anomalies = []
        max_score = 0.0

        # Check packet size Z-score
        size_z = self._z_score(packet.size, self._size_mean, self._size_std)
        if size_z > self.threshold:
            anomalies.append(
                f"Unusual packet size: {packet.size} bytes (Z={size_z:.2f})"
            )
            max_score = max(max_score, min(1.0, size_z / (self.threshold * 2)))

        # Check inter-arrival time
        if self._inter_arrival_times and self._iat_std > 0:
            current_iat = self._inter_arrival_times[-1]
            iat_z = self._z_score(current_iat, self._iat_mean, self._iat_std)

            if iat_z > self.threshold:
                anomalies.append(
                    f"Unusual timing: IAT={current_iat:.4f}s (Z={iat_z:.2f})"
                )
                max_score = max(max_score, min(1.0, iat_z / (self.threshold * 2)))

        if anomalies:
            return AnomalyResult(
                is_anomaly=True,
                score=max_score,
                detection_method="statistical",
                details={
                    "anomalies": anomalies,
                    "packet_size": packet.size,
                    "size_mean": self._size_mean,
                    "size_std": self._size_std,
                    "z_score": size_z,
                },
                timestamp=datetime.now(),
                related_ips=[packet.src_ip, packet.dst_ip],
            )

        return None


class RuleBasedDetector:
    """
    Rule-based anomaly detection.

    Implements security heuristics:
    - Port scan detection
    - DDoS pattern detection
    - Unusual protocol detection
    - Large transfer detection
    """

    # Suspicious ports commonly used by malware
    SUSPICIOUS_PORTS = {
        4444,  # Metasploit default
        5555,  # Android Debug Bridge
        6666,  # IRC
        6667,  # IRC
        31337,  # "Elite" backdoor
        12345,  # NetBus
        27374,  # Sub7
    }

    # Maximum expected packet size (threshold for "large packet")
    MAX_NORMAL_PACKET = 1500  # MTU

    def __init__(self):
        # Track connection patterns per IP
        self._ip_connections: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._ip_packet_count: Dict[str, int] = defaultdict(int)
        self._port_connections: Dict[str, set] = defaultdict(set)
        self._syn_count: Dict[str, int] = defaultdict(int)
        self._window_start = datetime.now()
        self._window_duration = 60  # seconds

    def _reset_window_if_needed(self):
        """Reset tracking window periodically."""
        now = datetime.now()
        if (now - self._window_start).total_seconds() > self._window_duration:
            self._ip_connections.clear()
            self._ip_packet_count.clear()
            self._port_connections.clear()
            self._syn_count.clear()
            self._window_start = now

    def check_packet(self, packet: PacketInfo) -> List[AnomalyResult]:
        """
        Check a packet against security rules.

        Returns list of detected anomalies.
        """
        self._reset_window_if_needed()

        anomalies = []

        # Track connection
        self._ip_connections[packet.src_ip][packet.dst_ip] += 1
        self._ip_packet_count[packet.src_ip] += 1
        self._port_connections[packet.src_ip].add(packet.dst_port)

        # Rule 1: Suspicious port
        if (
            packet.dst_port in self.SUSPICIOUS_PORTS
            or packet.src_port in self.SUSPICIOUS_PORTS
        ):
            anomalies.append(
                AnomalyResult(
                    is_anomaly=True,
                    score=0.7,
                    detection_method="rule:suspicious_port",
                    details={
                        "rule": "Suspicious port detected",
                        "src_port": packet.src_port,
                        "dst_port": packet.dst_port,
                    },
                    timestamp=datetime.now(),
                    related_ips=[packet.src_ip, packet.dst_ip],
                )
            )

        # Rule 2: Large packet
        if packet.size > self.MAX_NORMAL_PACKET:
            anomalies.append(
                AnomalyResult(
                    is_anomaly=True,
                    score=0.5,
                    detection_method="rule:large_packet",
                    details={
                        "rule": "Unusually large packet",
                        "size": packet.size,
                        "threshold": self.MAX_NORMAL_PACKET,
                    },
                    timestamp=datetime.now(),
                    related_ips=[packet.src_ip, packet.dst_ip],
                )
            )

        # Rule 3: Port scan detection (many ports from single source)
        unique_ports = len(self._port_connections[packet.src_ip])
        if unique_ports > 20:
            anomalies.append(
                AnomalyResult(
                    is_anomaly=True,
                    score=0.8,
                    detection_method="rule:port_scan",
                    details={
                        "rule": "Potential port scan",
                        "source_ip": packet.src_ip,
                        "unique_ports": unique_ports,
                    },
                    timestamp=datetime.now(),
                    related_ips=[packet.src_ip],
                )
            )

        # Rule 4: High packet rate (potential DDoS) — absolute threshold
        packet_count = self._ip_packet_count[packet.src_ip]
        if packet_count > 100:  # More than 100 packets in window
            anomalies.append(
                AnomalyResult(
                    is_anomaly=True,
                    score=0.6,
                    detection_method="rule:high_rate",
                    details={
                        "rule": "High packet rate",
                        "source_ip": packet.src_ip,
                        "packet_count": packet_count,
                        "window_seconds": self._window_duration,
                    },
                    timestamp=datetime.now(),
                    related_ips=[packet.src_ip],
                )
            )

        # Rule 6: DDoS detection — relative threshold (10× average)
        # If a source IP sends ≥ 10× the average request count across all
        # tracked IPs, flag it as a probable DDoS source.
        if len(self._ip_packet_count) >= 2:
            total_all = sum(self._ip_packet_count.values())
            avg_count = total_all / len(self._ip_packet_count)
            if avg_count > 0 and packet_count >= 10 * avg_count:
                anomalies.append(
                    AnomalyResult(
                        is_anomaly=True,
                        score=0.9,
                        detection_method="rule:ddos_10x_avg",
                        details={
                            "rule": "DDoS suspect — 10× average request rate",
                            "source_ip": packet.src_ip,
                            "packet_count": packet_count,
                            "average_count": round(avg_count, 2),
                            "ratio": round(packet_count / avg_count, 2),
                        },
                        timestamp=datetime.now(),
                        related_ips=[packet.src_ip],
                    )
                )

        # Rule 5: SYN flood detection
        if packet.protocol == "TCP" and packet.flags and "SYN" in packet.flags:
            self._syn_count[packet.src_ip] += 1
            syn_count = self._syn_count[packet.src_ip]
            if syn_count > 50:
                anomalies.append(
                    AnomalyResult(
                        is_anomaly=True,
                        score=0.85,
                        detection_method="rule:syn_flood",
                        details={
                            "rule": "Potential SYN flood",
                            "source_ip": packet.src_ip,
                            "syn_count": syn_count,
                        },
                        timestamp=datetime.now(),
                        related_ips=[packet.src_ip, packet.dst_ip],
                    )
                )

        return anomalies


class MLDetector:
    """
    Machine Learning-based anomaly detection.

    Supports multiple models:
    - Decision Tree (supervised)
    - K-Means Clustering (unsupervised)
    - Naive Bayes (supervised)
    """

    def __init__(
        self, model_path: Optional[str] = None, model_type: str = "decision_tree"
    ):
        """
        Initialize ML detector.

        Args:
            model_path: Path to saved model (None to use default/untrained)
            model_type: "decision_tree", "kmeans", or "naive_bayes"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = FeatureScaler()
        self.feature_extractor = FeatureExtractor()
        self._packet_buffer: List[PacketInfo] = []
        self._buffer_window = 30  # seconds

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        """Load a trained model."""
        if not Path(path).exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data.get("model")
            self.scaler = data.get("scaler", FeatureScaler())
            self.model_type = data.get("model_type", self.model_type)

    def save_model(self, path: str):
        """Save the current model."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "model_type": self.model_type,
                },
                f,
            )

    def add_packet(self, packet: PacketInfo):
        """Add packet to buffer for feature extraction."""
        self._packet_buffer.append(packet)

        # Clean old packets
        cutoff = datetime.now() - timedelta(seconds=self._buffer_window)
        self._packet_buffer = [p for p in self._packet_buffer if p.timestamp > cutoff]

    def check_anomaly(self) -> Optional[AnomalyResult]:
        """
        Check for anomalies using ML model.

        Returns AnomalyResult if anomaly detected.
        """
        if not self.model or len(self._packet_buffer) < 10:
            return None

        # Extract features
        features = self.feature_extractor.extract_features(self._packet_buffer)
        feature_vector = features.to_vector()

        # Scale features
        try:
            scaled = self.scaler.transform_minmax(feature_vector)
        except (RuntimeError, TypeError, ValueError, IndexError):
            scaled = feature_vector

        # Predict
        if self.model_type == "decision_tree":
            prediction = self.model.predict([scaled])[0]
            is_anomaly = prediction == "anomaly"
            score = 0.8 if is_anomaly else 0.2

        elif self.model_type == "kmeans":
            score = self.model.distance_to_centroid(scaled)
            is_anomaly = score > 0.7  # Threshold for anomaly

        elif self.model_type == "naive_bayes":
            prediction = self.model.predict([scaled])[0]
            probs = self.model.predict_proba([scaled])[0]
            is_anomaly = prediction == "anomaly"
            score = probs.get("anomaly", 0.0)

        else:
            return None

        if is_anomaly:
            return AnomalyResult(
                is_anomaly=True,
                score=score,
                detection_method=f"ml:{self.model_type}",
                details={
                    "model_type": self.model_type,
                    "features": features.to_dict(),
                    "prediction_score": score,
                },
                timestamp=datetime.now(),
                related_ips=list(
                    set(
                        [p.src_ip for p in self._packet_buffer[-10:]]
                        + [p.dst_ip for p in self._packet_buffer[-10:]]
                    )
                ),
            )

        return None


class AnomalyDetector:
    """
    Master anomaly detection system combining multiple detection methods.

    Ensemble approach:
    1. Statistical detection (fast, low false positive)
    2. Rule-based detection (domain knowledge)
    3. ML-based detection (learned patterns)

    Final score is weighted combination of all detectors.
    """

    def __init__(
        self,
        enable_statistical: bool = True,
        enable_rules: bool = True,
        enable_ml: bool = False,
        ml_model_path: Optional[str] = None,
    ):
        """
        Initialize anomaly detector.

        Args:
            enable_statistical: Enable statistical detection
            enable_rules: Enable rule-based detection
            enable_ml: Enable ML-based detection
            ml_model_path: Path to trained ML model
        """
        self.statistical_detector = (
            StatisticalDetector() if enable_statistical else None
        )
        self.rule_detector = RuleBasedDetector() if enable_rules else None
        self.ml_detector = MLDetector(ml_model_path) if enable_ml else None

        # Tracking
        self._total_checked = 0
        self._anomalies_detected = 0
        self._anomaly_history: List[AnomalyResult] = []

    def check_packet(self, packet: PacketInfo) -> List[AnomalyResult]:
        """
        Check a packet for anomalies using all enabled detectors.

        Returns list of all detected anomalies.
        """
        self._total_checked += 1
        anomalies = []

        # Statistical check
        if self.statistical_detector:
            result = self.statistical_detector.check_packet(packet)
            if result:
                anomalies.append(result)

        # Rule-based check
        if self.rule_detector:
            rule_results = self.rule_detector.check_packet(packet)
            anomalies.extend(rule_results)

        # ML check
        if self.ml_detector:
            self.ml_detector.add_packet(packet)
            # Only check ML periodically (every 10 packets)
            if self._total_checked % 10 == 0:
                result = self.ml_detector.check_anomaly()
                if result:
                    anomalies.append(result)

        # Track anomalies
        if anomalies:
            self._anomalies_detected += len(anomalies)
            self._anomaly_history.extend(anomalies)

            # Keep history bounded
            if len(self._anomaly_history) > 1000:
                self._anomaly_history = self._anomaly_history[-500:]

        return anomalies

    def get_aggregate_score(self, packet: PacketInfo) -> float:
        """
        Get aggregate anomaly score for a packet.

        Combines scores from all detectors.
        Returns value between 0.0 (normal) and 1.0 (definitely anomaly).
        """
        anomalies = self.check_packet(packet)

        if not anomalies:
            return 0.0

        # Weight by detection method
        weights = {"statistical": 0.3, "rule": 0.4, "ml": 0.5}

        total_weight = 0.0
        weighted_score = 0.0

        for anomaly in anomalies:
            method_type = anomaly.detection_method.split(":")[0]
            weight = weights.get(method_type, 0.3)
            weighted_score += anomaly.score * weight
            total_weight += weight

        return min(1.0, weighted_score / max(total_weight, 1))

    def create_alert(self, anomaly: AnomalyResult) -> Alert:
        """Create an Alert object from an AnomalyResult."""
        # Determine alert type
        method = anomaly.detection_method
        if "port_scan" in method:
            alert_type = AlertType.PORT_SCAN
        elif "ddos" in method or "high_rate" in method or "syn_flood" in method:
            alert_type = AlertType.DDOS_SUSPECT
        elif "large_packet" in method:
            alert_type = AlertType.LARGE_PACKET
        else:
            alert_type = AlertType.ANOMALY

        # Determine severity
        if anomaly.score >= 0.8:
            severity = AlertSeverity.CRITICAL
        elif anomaly.score >= 0.6:
            severity = AlertSeverity.HIGH
        elif anomaly.score >= 0.4:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        return Alert(
            timestamp=anomaly.timestamp,
            alert_type=alert_type,
            severity=severity,
            description=str(anomaly.details),
            src_ip=anomaly.related_ips[0] if anomaly.related_ips else None,
            dst_ip=anomaly.related_ips[1] if len(anomaly.related_ips) > 1 else None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_packets_checked": self._total_checked,
            "anomalies_detected": self._anomalies_detected,
            "anomaly_rate": self._anomalies_detected / max(self._total_checked, 1),
            "recent_anomalies": len(self._anomaly_history),
            "detectors_enabled": {
                "statistical": self.statistical_detector is not None,
                "rules": self.rule_detector is not None,
                "ml": self.ml_detector is not None,
            },
        }

    def get_recent_anomalies(self, limit: int = 20) -> List[AnomalyResult]:
        """Get recent anomalies."""
        return self._anomaly_history[-limit:]
