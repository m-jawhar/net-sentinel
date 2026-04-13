"""
Tests for the ML anomaly detection module.
"""

import sys
import pytest
import math
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from netsentinel.sniffer.packet_parser import PacketInfo
from netsentinel.ml.feature_extractor import (
    FeatureExtractor,
    TrafficFeatures,
    FeatureScaler,
)
from netsentinel.ml.model_trainer import (
    DecisionTreeClassifier,
    KMeansCluster,
    NaiveBayesClassifier,
    LinearRegressionModel,
    ModelTrainer,
)
from netsentinel.ml.anomaly_detector import (
    AnomalyDetector,
    StatisticalDetector,
    RuleBasedDetector,
)


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    @pytest.fixture
    def packets(self):
        """Create sample packets."""
        base_time = datetime.now()
        return [
            PacketInfo(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                src_ip="192.168.1.1",
                dst_ip="8.8.8.8",
                src_port=12345,
                dst_port=80,
                protocol="TCP",
                size=100 + i * 10,
                flags="ACK",
                ttl=64,
            )
            for i in range(50)
        ]

    def test_extract_features(self, packets):
        """Test feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(packets)

        assert features.avg_packet_size > 0
        assert features.tcp_ratio == 1.0
        assert features.unique_destinations == 1
        assert features.unique_sources == 1

    def test_feature_vector(self, packets):
        """Test feature vector conversion."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(packets)

        vector = features.to_vector()
        assert len(vector) == 22  # 22 features (including time-of-day)
        assert all(isinstance(v, (int, float)) for v in vector)

    def test_windowed_extraction(self, packets):
        """Test windowed feature extraction."""
        extractor = FeatureExtractor()
        windows = extractor.extract_features_windowed(packets, window_seconds=1)

        assert len(windows) >= 1

    def test_empty_packets(self):
        """Test with no packets."""
        extractor = FeatureExtractor()
        features = extractor.extract_features([])

        assert features.avg_packet_size == 0.0

    def test_time_of_day_feature(self, packets):
        """Test time-of-day feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(packets)

        # hour_of_day should be a valid hour (0-23)
        assert 0.0 <= features.hour_of_day <= 23.0
        # is_business_hours should be a ratio (0-1)
        assert 0.0 <= features.is_business_hours <= 1.0


class TestFeatureScaler:
    """Tests for FeatureScaler."""

    def test_minmax_scaling(self):
        """Test min-max scaling."""
        scaler = FeatureScaler()

        vectors = [[0.0] * 22, [100.0] * 22, [50.0] * 22]

        scaler.fit(vectors)
        scaled = scaler.transform_minmax(vectors[2])

        # All should be 0.5 (midpoint)
        for v in scaled:
            assert abs(v - 0.5) < 0.01

    def test_zscore_scaling(self):
        """Test z-score scaling."""
        scaler = FeatureScaler()

        vectors = [
            [10.0] * 22,
            [20.0] * 22,
            [30.0] * 22,
        ]

        scaler.fit(vectors)
        scaled = scaler.transform_zscore(vectors[1])

        # Mean value should have z-score = 0
        for v in scaled:
            assert abs(v) < 0.01


class TestDecisionTree:
    """Tests for DecisionTreeClassifier."""

    def test_simple_classification(self):
        """Test simple classification problem."""
        # Create linearly separable data
        X = [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],  # Class A
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0],  # Class B
        ]
        y = ["a", "a", "a", "b", "b", "b"]

        dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
        dt.fit(X, y, feature_names=["f1", "f2"])

        predictions = dt.predict(X)
        assert predictions[0] == "a"
        assert predictions[-1] == "b"

    def test_entropy_calculation(self):
        """Test entropy calculation."""
        dt = DecisionTreeClassifier()

        # Pure class: entropy = 0
        assert dt._entropy(["a", "a", "a"]) == 0.0

        # Equal split: entropy = 1
        entropy = dt._entropy(["a", "b", "a", "b"])
        assert abs(entropy - 1.0) < 0.01

    def test_feature_importance(self):
        """Test feature importance is calculated."""
        X = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [10.0, 5.0], [11.0, 5.0], [12.0, 5.0]]
        y = ["a", "a", "a", "b", "b", "b"]

        dt = DecisionTreeClassifier()
        dt.fit(X, y, feature_names=["important", "useless"])

        assert len(dt.feature_importance_) > 0


class TestKMeans:
    """Tests for KMeansCluster."""

    def test_clustering(self):
        """Test basic clustering."""
        # Two clear clusters
        X = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],  # Cluster 1
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],  # Cluster 2
        ]

        km = KMeansCluster(n_clusters=2, random_state=42)
        km.fit(X)

        labels = km.predict(X)

        # First 3 should be in same cluster, last 3 in another
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_anomaly_scores(self):
        """Test anomaly score calculation."""
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]  # Outlier

        km = KMeansCluster(n_clusters=1, random_state=42)
        km.fit(X)

        scores = km.get_anomaly_scores(X)

        # Outlier should have highest score
        assert scores[3] == max(scores)

    def test_euclidean_distance(self):
        """Test distance calculation."""
        km = KMeansCluster()

        dist = km._euclidean_distance([0, 0], [3, 4])
        assert abs(dist - 5.0) < 0.01


class TestLinearRegression:
    """Tests for LinearRegressionModel."""

    def test_simple_regression(self):
        """Test simple linear relationship: y = 2x + 1."""
        X = [[float(i)] for i in range(20)]
        y = [2.0 * i + 1.0 for i in range(20)]

        lr = LinearRegressionModel()
        lr.fit(X, y)

        predictions = lr.predict([[5.0], [10.0]])
        assert abs(predictions[0] - 11.0) < 1.0
        assert abs(predictions[1] - 21.0) < 1.0

    def test_r_squared(self):
        """Test R² for perfect linear data."""
        X = [[float(i)] for i in range(30)]
        y = [3.0 * i + 5.0 for i in range(30)]

        lr = LinearRegressionModel()
        lr.fit(X, y)

        # R² should be very close to 1.0 for perfect data
        assert lr.r_squared > 0.95

    def test_coefficients(self):
        """Test coefficient retrieval."""
        X = [[float(i), float(i * 2)] for i in range(20)]
        y = [float(i) for i in range(20)]

        lr = LinearRegressionModel()
        lr.fit(X, y, feature_names=["feat_a", "feat_b"])

        coeffs = lr.get_coefficients()
        assert "feat_a" in coeffs
        assert "feat_b" in coeffs

    def test_trainer_integration(self):
        """Test LinearRegression through ModelTrainer."""
        X = [[float(i), float(i**2)] for i in range(50)]
        y = [2.0 * i + 0.5 * i**2 for i in range(50)]

        trainer = ModelTrainer(models_dir="test_models_lr")
        model, stats = trainer.train_linear_regression(
            X, y, feature_names=["x", "x_squared"]
        )

        assert stats["model_type"] == "LinearRegression"
        assert "mse" in stats
        assert "r_squared" in stats
        assert "coefficients" in stats

        # Cleanup
        import shutil

        shutil.rmtree("test_models_lr", ignore_errors=True)


class TestNaiveBayes:
    """Tests for NaiveBayesClassifier."""

    def test_simple_classification(self):
        """Test simple classification."""
        X = [[1.0], [1.5], [2.0], [10.0], [10.5], [11.0]]  # Class A  # Class B
        y = ["a", "a", "a", "b", "b", "b"]

        nb = NaiveBayesClassifier()
        nb.fit(X, y)

        predictions = nb.predict([[1.0], [10.0]])
        assert predictions[0] == "a"
        assert predictions[1] == "b"

    def test_predict_proba(self):
        """Test probability prediction."""
        X = [[1.0], [1.5], [2.0], [10.0], [10.5], [11.0]]
        y = ["a", "a", "a", "b", "b", "b"]

        nb = NaiveBayesClassifier()
        nb.fit(X, y)

        probs = nb.predict_proba([[1.0]])
        assert len(probs) == 1
        assert "a" in probs[0]
        assert "b" in probs[0]
        assert abs(sum(probs[0].values()) - 1.0) < 0.01


class TestStatisticalDetector:
    """Tests for StatisticalDetector."""

    def test_normal_traffic(self):
        """Test that normal traffic doesn't trigger alerts."""
        detector = StatisticalDetector(window_size=100, threshold=3.0)

        # Send many normal packets with consistent inter-arrival times
        base_time = datetime(2025, 1, 1, 12, 0, 0)
        for i in range(100):
            packet = PacketInfo(
                timestamp=base_time + timedelta(milliseconds=i * 10),
                src_ip="192.168.1.1",
                dst_ip="8.8.8.8",
                src_port=12345,
                dst_port=80,
                protocol="TCP",
                size=100,
            )
            result = detector.check_packet(packet)

        # Normal sized packet with consistent timing should not trigger
        normal_packet = PacketInfo(
            timestamp=base_time + timedelta(milliseconds=100 * 10),
            src_ip="192.168.1.1",
            dst_ip="8.8.8.8",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            size=100,
        )
        result = detector.check_packet(normal_packet)
        assert result is None


class TestRuleBasedDetector:
    """Tests for RuleBasedDetector."""

    def test_suspicious_port(self):
        """Test suspicious port detection."""
        detector = RuleBasedDetector()

        packet = PacketInfo(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=4444,  # Metasploit default
            protocol="TCP",
            size=100,
        )

        results = detector.check_packet(packet)
        assert len(results) > 0
        assert any("suspicious_port" in r.detection_method for r in results)

    def test_large_packet(self):
        """Test large packet detection."""
        detector = RuleBasedDetector()

        packet = PacketInfo(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            size=50000,  # Very large packet
        )

        results = detector.check_packet(packet)
        assert any("large_packet" in r.detection_method for r in results)


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def test_creation(self):
        """Test detector creation."""
        detector = AnomalyDetector(
            enable_statistical=True, enable_rules=True, enable_ml=False
        )

        stats = detector.get_statistics()
        assert stats["total_packets_checked"] == 0
        assert stats["detectors_enabled"]["statistical"] == True
        assert stats["detectors_enabled"]["rules"] == True
        assert stats["detectors_enabled"]["ml"] == False

    def test_check_normal_packet(self):
        """Test checking a normal packet."""
        detector = AnomalyDetector(enable_statistical=False, enable_ml=False)

        packet = PacketInfo(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="8.8.8.8",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            size=100,
        )

        anomalies = detector.check_packet(packet)
        # Normal packet may or may not trigger rules
        assert isinstance(anomalies, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
