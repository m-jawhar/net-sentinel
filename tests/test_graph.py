"""
Tests for the graph analyzer and network graph modules.
"""

import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from netsentinel.visualizer.network_graph import NetworkGraph, Vertex, Edge
from netsentinel.visualizer.graph_analyzer import GraphAnalyzer


class TestNetworkGraph:
    """Tests for NetworkGraph."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        g = NetworkGraph()

        # Create a small network topology
        g.add_edge("192.168.1.1", "8.8.8.8", 100, "TCP")
        g.add_edge("192.168.1.1", "1.1.1.1", 200, "UDP")
        g.add_edge("192.168.1.2", "8.8.8.8", 150, "TCP")
        g.add_edge("8.8.8.8", "192.168.1.1", 300, "TCP")  # Bidirectional
        g.add_edge("10.0.0.1", "192.168.1.1", 50, "ICMP")

        return g

    def test_vertex_count(self, graph):
        """Test that vertices are created correctly."""
        assert len(graph.vertices) == 5  # 5 unique IPs

    def test_edge_count(self, graph):
        """Test that edges are created correctly."""
        assert len(graph.edges) == 5

    def test_degree(self, graph):
        """Test degree calculation."""
        # 192.168.1.1: out to 8.8.8.8 and 1.1.1.1, in from 8.8.8.8 and 10.0.0.1
        v = graph.get_vertex("192.168.1.1")
        assert v.out_degree == 2
        assert v.in_degree == 2
        assert v.degree == 4

    def test_neighbors(self, graph):
        """Test neighbor retrieval."""
        out_neighbors = graph.get_neighbors("192.168.1.1", "out")
        assert "8.8.8.8" in out_neighbors
        assert "1.1.1.1" in out_neighbors

        in_neighbors = graph.get_neighbors("8.8.8.8", "in")
        assert "192.168.1.1" in in_neighbors
        assert "192.168.1.2" in in_neighbors

    def test_internal_ip(self, graph):
        """Test internal IP detection."""
        assert graph.get_vertex("192.168.1.1").is_internal
        assert graph.get_vertex("10.0.0.1").is_internal
        assert not graph.get_vertex("8.8.8.8").is_internal

    def test_bidirectional_edge(self, graph):
        """Test bidirectional edge detection."""
        edge1 = graph.get_edge("192.168.1.1", "8.8.8.8")
        edge2 = graph.get_edge("8.8.8.8", "192.168.1.1")

        assert edge1.is_bidirectional
        assert edge2.is_bidirectional

    def test_adjacency_matrix(self, graph):
        """Test adjacency matrix generation."""
        ips, matrix = graph.get_adjacency_matrix()

        assert len(ips) == 5
        assert len(matrix) == 5
        """Test path finding."""
        path = graph.find_path("10.0.0.1", "1.1.1.1")
        assert path is not None
        assert path[0] == "10.0.0.1"
        assert path[-1] == "1.1.1.1"

    def test_connected_components(self, graph):
        """Test connected components."""
        components = graph.get_connected_components()
        assert len(components) == 1  # All connected

    def test_statistics(self, graph):
        """Test graph statistics."""
        stats = graph.get_statistics()
        assert stats["vertex_count"] == 5
        assert stats["edge_count"] == 5
        assert stats["avg_degree"] > 0

    def test_top_vertices(self, graph):
        """Test getting top vertices by degree."""
        top = graph.get_top_vertices_by_degree(2)
        assert len(top) == 2
        # 192.168.1.1 should be first (highest degree = 4)
        assert top[0].id == "192.168.1.1"

    def test_clear(self, graph):
        """Test graph clearing."""
        graph.clear()
        assert len(graph.vertices) == 0
        assert len(graph.edges) == 0

    def test_vis_js_format(self, graph):
        """Test export to vis.js format."""
        data = graph.to_vis_js_format()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) == 5


class TestGraphAnalyzer:
    """Tests for GraphAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer."""
        g = NetworkGraph()

        # Build a more complex graph
        for i in range(5):
            for j in range(5):
                if i != j:
                    g.add_edge(f"10.0.0.{i}", f"10.0.0.{j}", 100, "TCP")

        # Add a scanner-like node
        for i in range(25):
            g.add_edge("10.0.0.99", f"172.16.0.{i}", 50, "TCP")

        return GraphAnalyzer(g)

    def test_degree_distribution(self, analyzer):
        """Test degree distribution."""
        dist = analyzer.get_degree_distribution()
        assert len(dist) > 0

    def test_degree_statistics(self, analyzer):
        """Test degree statistics."""
        stats = analyzer.get_degree_statistics()
        assert stats["mean"] > 0
        assert stats["max"] >= stats["min"]

    def test_degree_centrality(self, analyzer):
        """Test degree centrality calculation."""
        centrality = analyzer.calculate_degree_centrality()
        assert len(centrality) > 0

        # All values should be between 0 and 1
        for ip, value in centrality.items():
            assert 0.0 <= value <= 1.0

    def test_detect_port_scanners(self, analyzer):
        """Test port scanner detection."""
        scanners = analyzer.detect_port_scanners(threshold=10)
        assert len(scanners) > 0
        assert scanners[0][0] == "10.0.0.99"

    def test_detect_hub_nodes(self, analyzer):
        """Test hub node detection."""
        hubs = analyzer.detect_hub_nodes(threshold_factor=1.5)
        assert len(hubs) > 0

    def test_clustering_coefficient(self, analyzer):
        """Test clustering coefficient."""
        avg_cc = analyzer.calculate_average_clustering_coefficient()
        assert 0.0 <= avg_cc <= 1.0

    def test_analysis_report(self, analyzer):
        """Test comprehensive analysis report."""
        report = analyzer.get_analysis_report()

        assert "graph_stats" in report
        assert "degree_stats" in report
        assert "potential_scanners" in report
        assert "hub_nodes" in report


class TestVertex:
    """Tests for Vertex dataclass."""

    def test_degree_property(self):
        """Test degree property."""
        v = Vertex(id="test", in_degree=3, out_degree=5)
        assert v.degree == 8

    def test_to_dict(self):
        """Test dictionary conversion."""
        v = Vertex(id="10.0.0.1", in_degree=2, out_degree=3)
        d = v.to_dict()

        assert d["id"] == "10.0.0.1"
        assert d["degree"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
