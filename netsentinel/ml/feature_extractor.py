"""
Feature Extractor Module - Extracts features from network traffic for ML.

Applies Machine Learning concepts (Sem 4):
- Feature engineering
- Data preprocessing
- Feature scaling/normalization
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import math

from ..sniffer.packet_parser import PacketInfo


@dataclass
class TrafficFeatures:
    """
    Feature vector for a traffic sample.

    Features extracted:
    - Packet size statistics
    - Inter-arrival time statistics
    - Protocol distribution
    - Port statistics
    - Connection patterns
    """

    # Packet size features
    avg_packet_size: float = 0.0
    std_packet_size: float = 0.0
    min_packet_size: int = 0
    max_packet_size: int = 0

    # Time-based features
    avg_inter_arrival_time: float = 0.0
    std_inter_arrival_time: float = 0.0
    packets_per_second: float = 0.0

    # Protocol features
    tcp_ratio: float = 0.0
    udp_ratio: float = 0.0
    icmp_ratio: float = 0.0

    # Port features
    unique_src_ports: int = 0
    unique_dst_ports: int = 0
    well_known_port_ratio: float = 0.0  # Ports < 1024

    # Connection features
    unique_destinations: int = 0
    unique_sources: int = 0
    avg_bytes_per_connection: float = 0.0

    # Flag features (TCP)
    syn_ratio: float = 0.0
    ack_ratio: float = 0.0
    fin_ratio: float = 0.0
    rst_ratio: float = 0.0

    # Time-of-day features
    hour_of_day: float = 0.0  # 0-23, average hour packets were seen
    is_business_hours: float = 0.0  # Ratio of packets during 9-17

    # Label (for supervised learning)
    label: str = "normal"  # "normal" or "anomaly"
    anomaly_score: float = 0.0

    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML models."""
        return [
            self.avg_packet_size,
            self.std_packet_size,
            self.min_packet_size,
            self.max_packet_size,
            self.avg_inter_arrival_time,
            self.std_inter_arrival_time,
            self.packets_per_second,
            self.tcp_ratio,
            self.udp_ratio,
            self.icmp_ratio,
            self.unique_src_ports,
            self.unique_dst_ports,
            self.well_known_port_ratio,
            self.unique_destinations,
            self.unique_sources,
            self.avg_bytes_per_connection,
            self.syn_ratio,
            self.ack_ratio,
            self.fin_ratio,
            self.rst_ratio,
            self.hour_of_day,
            self.is_business_hours,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model interpretation."""
        return [
            "avg_packet_size",
            "std_packet_size",
            "min_packet_size",
            "max_packet_size",
            "avg_inter_arrival_time",
            "std_inter_arrival_time",
            "packets_per_second",
            "tcp_ratio",
            "udp_ratio",
            "icmp_ratio",
            "unique_src_ports",
            "unique_dst_ports",
            "well_known_port_ratio",
            "unique_destinations",
            "unique_sources",
            "avg_bytes_per_connection",
            "syn_ratio",
            "ack_ratio",
            "fin_ratio",
            "rst_ratio",
            "hour_of_day",
            "is_business_hours",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            name: value for name, value in zip(self.feature_names(), self.to_vector())
        }


class FeatureExtractor:
    """
    Extracts ML features from network traffic data.

    Feature Engineering Process:
    1. Aggregate packets over time window
    2. Calculate statistical features
    3. Normalize/scale features
    4. Create feature vectors
    """

    # Well-known ports (0-1023)
    WELL_KNOWN_PORTS = set(range(1024))

    def __init__(self, window_size_seconds: float = 10.0):
        """
        Initialize feature extractor.

        Args:
            window_size_seconds: Time window for aggregating packets
        """
        self.window_size = window_size_seconds
        self._packet_buffer: List[PacketInfo] = []
        self._last_extraction: Optional[datetime] = None

    def add_packet(self, packet: PacketInfo):
        """Add a packet to the buffer."""
        self._packet_buffer.append(packet)

        # Remove old packets outside window
        cutoff = datetime.now() - timedelta(seconds=self.window_size * 2)
        self._packet_buffer = [p for p in self._packet_buffer if p.timestamp > cutoff]

    def extract_features(
        self, packets: Optional[List[PacketInfo]] = None
    ) -> TrafficFeatures:
        """
        Extract features from a list of packets.

        Args:
            packets: List of PacketInfo objects (uses buffer if None)

        Returns:
            TrafficFeatures object
        """
        if packets is None:
            packets = self._packet_buffer

        if not packets:
            return TrafficFeatures()

        features = TrafficFeatures()

        # ==================
        # Packet Size Features
        # ==================
        sizes = [p.size for p in packets]
        features.avg_packet_size = self._mean(sizes)
        features.std_packet_size = self._std_dev(sizes)
        features.min_packet_size = min(sizes)
        features.max_packet_size = max(sizes)

        # ==================
        # Time-based Features
        # ==================
        if len(packets) >= 2:
            # Sort by timestamp
            sorted_packets = sorted(packets, key=lambda p: p.timestamp)

            # Calculate inter-arrival times
            inter_arrivals = []
            for i in range(1, len(sorted_packets)):
                delta = (
                    sorted_packets[i].timestamp - sorted_packets[i - 1].timestamp
                ).total_seconds()
                inter_arrivals.append(delta)

            if inter_arrivals:
                features.avg_inter_arrival_time = self._mean(inter_arrivals)
                features.std_inter_arrival_time = self._std_dev(inter_arrivals)

            # Packets per second
            total_time = (
                sorted_packets[-1].timestamp - sorted_packets[0].timestamp
            ).total_seconds()
            if total_time > 0:
                features.packets_per_second = len(packets) / total_time

        # ==================
        # Protocol Features
        # ==================
        total = len(packets)
        protocol_counts = defaultdict(int)
        for p in packets:
            protocol_counts[p.protocol] += 1

        features.tcp_ratio = protocol_counts.get("TCP", 0) / total
        features.udp_ratio = protocol_counts.get("UDP", 0) / total
        features.icmp_ratio = protocol_counts.get("ICMP", 0) / total

        # ==================
        # Port Features
        # ==================
        src_ports = set()
        dst_ports = set()
        well_known_count = 0

        for p in packets:
            src_ports.add(p.src_port)
            dst_ports.add(p.dst_port)

            if p.src_port in self.WELL_KNOWN_PORTS:
                well_known_count += 1
            if p.dst_port in self.WELL_KNOWN_PORTS:
                well_known_count += 1

        features.unique_src_ports = len(src_ports)
        features.unique_dst_ports = len(dst_ports)
        features.well_known_port_ratio = well_known_count / (
            total * 2
        )  # Each packet has 2 ports

        # ==================
        # Connection Features
        # ==================
        destinations = set(p.dst_ip for p in packets)
        sources = set(p.src_ip for p in packets)
        connections = set((p.src_ip, p.dst_ip) for p in packets)

        features.unique_destinations = len(destinations)
        features.unique_sources = len(sources)

        if connections:
            total_bytes = sum(p.size for p in packets)
            features.avg_bytes_per_connection = total_bytes / len(connections)

        # ==================
        # TCP Flag Features
        # ==================
        tcp_packets = [p for p in packets if p.protocol == "TCP" and p.flags]
        if tcp_packets:
            tcp_total = len(tcp_packets)
            syn_count = sum(1 for p in tcp_packets if p.flags and "SYN" in p.flags)
            ack_count = sum(1 for p in tcp_packets if p.flags and "ACK" in p.flags)
            fin_count = sum(1 for p in tcp_packets if p.flags and "FIN" in p.flags)
            rst_count = sum(1 for p in tcp_packets if p.flags and "RST" in p.flags)

            features.syn_ratio = syn_count / tcp_total
            features.ack_ratio = ack_count / tcp_total
            features.fin_ratio = fin_count / tcp_total
            features.rst_ratio = rst_count / tcp_total

        # ==================
        # Time-of-Day Features
        # ==================
        hours = [p.timestamp.hour for p in packets]
        features.hour_of_day = self._mean(hours) if hours else 0.0
        business_count = sum(1 for h in hours if 9 <= h < 17)
        features.is_business_hours = business_count / total if total > 0 else 0.0

        self._last_extraction = datetime.now()
        return features

    def extract_features_for_ip(
        self, packets: List[PacketInfo], ip_address: str, direction: str = "src"
    ) -> TrafficFeatures:
        """
        Extract features for a specific IP's traffic.

        Args:
            packets: List of packets
            ip_address: IP to filter
            direction: "src" (outgoing), "dst" (incoming), or "both"
        """
        if direction == "src":
            filtered = [p for p in packets if p.src_ip == ip_address]
        elif direction == "dst":
            filtered = [p for p in packets if p.dst_ip == ip_address]
        else:
            filtered = [
                p for p in packets if p.src_ip == ip_address or p.dst_ip == ip_address
            ]

        return self.extract_features(filtered)

    def extract_features_windowed(
        self, packets: List[PacketInfo], window_seconds: float = 10.0
    ) -> List[TrafficFeatures]:
        """
        Extract features over multiple time windows.

        Useful for time-series analysis and detecting patterns over time.
        """
        if not packets:
            return []

        # Sort by timestamp
        sorted_packets = sorted(packets, key=lambda p: p.timestamp)

        features_list = []
        window_start = sorted_packets[0].timestamp
        window_end = window_start + timedelta(seconds=window_seconds)

        current_window = []

        for packet in sorted_packets:
            if packet.timestamp <= window_end:
                current_window.append(packet)
            else:
                # Extract features for current window
                if current_window:
                    features_list.append(self.extract_features(current_window))

                # Start new window
                window_start = packet.timestamp
                window_end = window_start + timedelta(seconds=window_seconds)
                current_window = [packet]

        # Don't forget the last window
        if current_window:
            features_list.append(self.extract_features(current_window))

        return features_list

    @staticmethod
    def _mean(values: List[float]) -> float:
        """Calculate mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std_dev(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def clear_buffer(self):
        """Clear the packet buffer."""
        self._packet_buffer.clear()


class FeatureScaler:
    """
    Scales features for ML models.

    Implements:
    - Min-Max scaling: x' = (x - min) / (max - min)
    - Z-score normalization: x' = (x - mean) / std
    """

    def __init__(self):
        self._mins: Dict[str, float] = {}
        self._maxs: Dict[str, float] = {}
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        self._fitted = False

    def fit(self, feature_vectors: List[List[float]]):
        """
        Fit scaler to training data.

        Calculates min, max, mean, std for each feature.
        """
        if not feature_vectors:
            return

        n_features = len(feature_vectors[0])
        feature_names = TrafficFeatures.feature_names()

        for i in range(n_features):
            values = [fv[i] for fv in feature_vectors]
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"

            self._mins[name] = min(values)
            self._maxs[name] = max(values)
            self._means[name] = sum(values) / len(values)

            variance = sum((v - self._means[name]) ** 2 for v in values) / len(values)
            self._stds[name] = math.sqrt(variance) if variance > 0 else 1.0

        self._fitted = True

    def transform_minmax(self, feature_vector: List[float]) -> List[float]:
        """Apply min-max scaling."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        feature_names = TrafficFeatures.feature_names()
        scaled = []

        for i, value in enumerate(feature_vector):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            min_val = self._mins.get(name, 0)
            max_val = self._maxs.get(name, 1)

            range_val = max_val - min_val
            if range_val > 0:
                scaled.append((value - min_val) / range_val)
            else:
                scaled.append(0.0)

        return scaled

    def transform_zscore(self, feature_vector: List[float]) -> List[float]:
        """Apply z-score normalization."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        feature_names = TrafficFeatures.feature_names()
        scaled = []

        for i, value in enumerate(feature_vector):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            mean = self._means.get(name, 0)
            std = self._stds.get(name, 1)

            scaled.append((value - mean) / std)

        return scaled

    def fit_transform_minmax(
        self, feature_vectors: List[List[float]]
    ) -> List[List[float]]:
        """Fit and transform in one step."""
        self.fit(feature_vectors)
        return [self.transform_minmax(fv) for fv in feature_vectors]

    def fit_transform_zscore(
        self, feature_vectors: List[List[float]]
    ) -> List[List[float]]:
        """Fit and transform using z-score."""
        self.fit(feature_vectors)
        return [self.transform_zscore(fv) for fv in feature_vectors]
