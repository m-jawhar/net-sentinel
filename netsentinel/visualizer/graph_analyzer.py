"""
Graph Analyzer Module - Analyzes network graph for patterns and anomalies.

Applies Graph Theory concepts (Sem 4):
- Centrality measures (degree, betweenness)
- Clustering coefficients
- Community detection
- Anomaly detection based on graph properties
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import math

from .network_graph import NetworkGraph, Vertex, Edge


class GraphAnalyzer:
    """
    Analyzes network graphs to identify patterns and potential threats.

    Key Analysis Methods:
    1. Degree Analysis - Identify highly connected nodes
    2. Centrality Analysis - Find influential nodes
    3. Clustering - Group related nodes
    4. Anomaly Detection - Flag unusual patterns
    """

    def __init__(self, graph: NetworkGraph):
        """
        Initialize analyzer with a network graph.

        Args:
            graph: NetworkGraph instance to analyze
        """
        self.graph = graph

    # ===================
    # Degree Analysis
    # ===================

    def get_degree_distribution(self) -> Dict[int, int]:
        """
        Calculate degree distribution.

        Returns dict mapping degree -> count of vertices with that degree.

        Used for:
        - Identifying network structure (scale-free, random, etc.)
        - Detecting anomalies (unusual degree patterns)
        """
        distribution = defaultdict(int)
        for vertex in self.graph.vertices.values():
            distribution[vertex.degree] += 1
        return dict(sorted(distribution.items()))

    def get_degree_statistics(self) -> Dict[str, float]:
        """
        Calculate degree statistics.

        Returns mean, variance, and standard deviation of degrees.
        """
        if not self.graph.vertices:
            return {"mean": 0, "variance": 0, "std_dev": 0}

        degrees = [v.degree for v in self.graph.vertices.values()]
        n = len(degrees)

        mean = sum(degrees) / n
        variance = sum((d - mean) ** 2 for d in degrees) / n
        std_dev = math.sqrt(variance)

        return {
            "mean": round(mean, 2),
            "variance": round(variance, 2),
            "std_dev": round(std_dev, 2),
            "min": min(degrees),
            "max": max(degrees),
        }

    def find_high_degree_nodes(self, threshold_factor: float = 2.0) -> List[Vertex]:
        """
        Find nodes with degree significantly above average.

        Args:
            threshold_factor: Multiplier for mean (default 2x mean)

        High degree nodes may indicate:
        - Central servers (legitimate)
        - Scanners or attackers (suspicious)
        """
        stats = self.get_degree_statistics()
        threshold = stats["mean"] + (threshold_factor * stats["std_dev"])

        return [v for v in self.graph.vertices.values() if v.degree > threshold]

    # ===================
    # Centrality Measures
    # ===================

    def calculate_degree_centrality(self) -> Dict[str, float]:
        """
        Calculate normalized degree centrality for all vertices.

        Degree Centrality = degree(v) / (n - 1)

        Measures how connected a node is relative to the entire network.
        """
        n = len(self.graph.vertices)
        if n <= 1:
            return {ip: 0.0 for ip in self.graph.vertices}

        return {
            ip: round(v.degree / (n - 1), 4) for ip, v in self.graph.vertices.items()
        }

    def calculate_in_degree_centrality(self) -> Dict[str, float]:
        """
        Calculate in-degree centrality (for directed graphs).

        High in-degree = many connections TO this node (popular destination)
        """
        n = len(self.graph.vertices)
        if n <= 1:
            return {ip: 0.0 for ip in self.graph.vertices}

        return {
            ip: round(v.in_degree / (n - 1), 4) for ip, v in self.graph.vertices.items()
        }

    def calculate_out_degree_centrality(self) -> Dict[str, float]:
        """
        Calculate out-degree centrality (for directed graphs).

        High out-degree = many connections FROM this node (active sender)
        """
        n = len(self.graph.vertices)
        if n <= 1:
            return {ip: 0.0 for ip in self.graph.vertices}

        return {
            ip: round(v.out_degree / (n - 1), 4)
            for ip, v in self.graph.vertices.items()
        }

    def calculate_closeness_centrality(self) -> Dict[str, float]:
        """
        Calculate closeness centrality using BFS for shortest paths.

        Closeness(v) = (n - 1) / sum of shortest path distances from v

        Measures how "close" a node is to all other nodes.
        High closeness = quickly reachable from many nodes.
        """
        centrality = {}
        n = len(self.graph.vertices)

        if n <= 1:
            return {ip: 0.0 for ip in self.graph.vertices}

        for source in self.graph.vertices:
            distances = self._bfs_distances(source)

            # Sum of distances to reachable nodes
            total_distance = sum(d for d in distances.values() if d > 0)
            reachable = sum(1 for d in distances.values() if d > 0)

            if total_distance > 0 and reachable > 0:
                # Normalize by reachable nodes
                centrality[source] = round(
                    (reachable / (n - 1)) * (reachable / total_distance), 4
                )
            else:
                centrality[source] = 0.0

        return centrality

    def _bfs_distances(self, source: str) -> Dict[str, int]:
        """Calculate shortest path distances from source using BFS."""
        distances = {source: 0}
        queue = [source]

        while queue:
            current = queue.pop(0)
            current_dist = distances[current]

            for neighbor in self.graph.get_neighbors(current, "both"):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def get_most_central_nodes(
        self, n: int = 5, centrality_type: str = "degree"
    ) -> List[Tuple[str, float]]:
        """
        Get the N most central nodes.

        Args:
            n: Number of nodes to return
            centrality_type: "degree", "in_degree", "out_degree", or "closeness"
        """
        if centrality_type == "degree":
            centrality = self.calculate_degree_centrality()
        elif centrality_type == "in_degree":
            centrality = self.calculate_in_degree_centrality()
        elif centrality_type == "out_degree":
            centrality = self.calculate_out_degree_centrality()
        elif centrality_type == "closeness":
            centrality = self.calculate_closeness_centrality()
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:n]

    # ===================
    # Clustering Analysis
    # ===================

    def calculate_local_clustering_coefficient(self, ip: str) -> float:
        """
        Calculate local clustering coefficient for a vertex.

        C(v) = 2 * |edges among neighbors| / (degree(v) * (degree(v) - 1))

        Measures how interconnected a node's neighbors are.
        High coefficient = neighbors form a tight cluster.
        """
        neighbors = self.graph.get_neighbors(ip, "both")
        k = len(neighbors)

        if k < 2:
            return 0.0

        # Count edges among neighbors
        neighbor_edges = 0
        neighbor_list = list(neighbors)

        for i, n1 in enumerate(neighbor_list):
            for n2 in neighbor_list[i + 1 :]:
                if self.graph.is_connected(n1, n2):
                    neighbor_edges += 1

        # Maximum possible edges among neighbors
        max_edges = k * (k - 1) / 2

        return round(neighbor_edges / max_edges, 4)

    def calculate_average_clustering_coefficient(self) -> float:
        """
        Calculate average clustering coefficient for the entire graph.

        Measures overall "cliquishness" of the network.
        """
        if not self.graph.vertices:
            return 0.0

        coefficients = [
            self.calculate_local_clustering_coefficient(ip)
            for ip in self.graph.vertices
        ]

        return round(sum(coefficients) / len(coefficients), 4)

    # ===================
    # Anomaly Detection
    # ===================

    def detect_port_scanners(self, threshold: int = 20) -> List[Tuple[str, int]]:
        """
        Detect potential port scanners.

        Heuristic: High out-degree to many unique destinations.

        Args:
            threshold: Minimum unique destinations to flag

        Returns:
            List of (IP, destination_count) tuples
        """
        suspects = []

        for ip, vertex in self.graph.vertices.items():
            destinations = self.graph.get_neighbors(ip, "out")
            if len(destinations) >= threshold:
                suspects.append((ip, len(destinations)))

        return sorted(suspects, key=lambda x: x[1], reverse=True)

    def detect_ddos_targets(self, threshold: int = 20) -> List[Tuple[str, int]]:
        """
        Detect potential DDoS targets.

        Heuristic: High in-degree (many sources sending to one destination).

        Args:
            threshold: Minimum unique sources to flag

        Returns:
            List of (IP, source_count) tuples
        """
        suspects = []

        for ip, vertex in self.graph.vertices.items():
            sources = self.graph.get_neighbors(ip, "in")
            if len(sources) >= threshold:
                suspects.append((ip, len(sources)))

        return sorted(suspects, key=lambda x: x[1], reverse=True)

    def detect_isolated_nodes(self) -> List[str]:
        """
        Detect isolated nodes (no connections).

        In network traffic, isolated nodes are unusual and may
        indicate reconnaissance or stealth activity.
        """
        return [ip for ip, v in self.graph.vertices.items() if v.degree == 0]

    def detect_hub_nodes(
        self, threshold_factor: float = 3.0
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Detect hub nodes (extremely high connectivity).

        Hubs with unusual patterns may indicate:
        - Compromised servers
        - Command & Control (C2) servers
        - Data exfiltration points

        Args:
            threshold_factor: Standard deviations above mean
        """
        stats = self.get_degree_statistics()
        threshold = stats["mean"] + (threshold_factor * stats["std_dev"])

        hubs = []
        for ip, vertex in self.graph.vertices.items():
            if vertex.degree > threshold:
                hubs.append(
                    (
                        ip,
                        {
                            "degree": vertex.degree,
                            "in_degree": vertex.in_degree,
                            "out_degree": vertex.out_degree,
                            "total_bytes": vertex.total_bytes,
                            "is_internal": vertex.is_internal,
                            "ratio": vertex.out_degree / max(vertex.in_degree, 1),
                        },
                    )
                )

        return sorted(hubs, key=lambda x: x[1]["degree"], reverse=True)

    def analyze_traffic_asymmetry(self) -> List[Tuple[str, float]]:
        """
        Analyze traffic asymmetry (in vs out traffic).

        High asymmetry may indicate:
        - Data exfiltration (high outbound)
        - DDoS target (high inbound)

        Returns:
            List of (IP, asymmetry_ratio) where ratio > 1 = more outbound
        """
        asymmetry = []

        for ip, vertex in self.graph.vertices.items():
            if vertex.in_degree > 0 and vertex.out_degree > 0:
                ratio = vertex.out_degree / vertex.in_degree
                asymmetry.append((ip, round(ratio, 2)))

        return sorted(asymmetry, key=lambda x: abs(x[1] - 1), reverse=True)

    def get_analysis_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Returns dictionary with all analysis results.
        """
        return {
            "graph_stats": self.graph.get_statistics(),
            "degree_stats": self.get_degree_statistics(),
            "degree_distribution": self.get_degree_distribution(),
            "avg_clustering": self.calculate_average_clustering_coefficient(),
            "top_degree_nodes": self.get_most_central_nodes(5, "degree"),
            "top_in_degree": self.get_most_central_nodes(5, "in_degree"),
            "top_out_degree": self.get_most_central_nodes(5, "out_degree"),
            "potential_scanners": self.detect_port_scanners(10),
            "potential_ddos_targets": self.detect_ddos_targets(10),
            "hub_nodes": self.detect_hub_nodes(2.0),
            "isolated_nodes": self.detect_isolated_nodes(),
        }
