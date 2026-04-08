"""
Network Graph Module - Represents network topology as a graph.

Applies Graph Theory concepts (Sem 4):
- Vertices (IP addresses) and Edges (connections)
- Adjacency matrix representation
- Degree calculation
- Connectivity analysis
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Vertex:
    """
    Represents a node (IP address) in the network graph.

    Properties:
    - id: IP address
    - in_degree: Number of incoming connections
    - out_degree: Number of outgoing connections
    - weight: Total bytes transferred
    """

    id: str
    label: str = ""
    in_degree: int = 0
    out_degree: int = 0
    total_bytes: int = 0
    packet_count: int = 0
    is_internal: bool = False
    is_anomalous: bool = False

    @property
    def degree(self) -> int:
        """Total degree (in + out)."""
        return self.in_degree + self.out_degree

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label or self.id,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "degree": self.degree,
            "total_bytes": self.total_bytes,
            "packet_count": self.packet_count,
            "is_internal": self.is_internal,
            "is_anomalous": self.is_anomalous,
        }


@dataclass
class Edge:
    """
    Represents a connection between two IP addresses.

    Properties:
    - source: Source IP
    - target: Destination IP
    - weight: Number of packets or bytes
    - protocols: List of protocols used
    """

    source: str
    target: str
    weight: int = 1
    byte_count: int = 0
    protocols: Set[str] = field(default_factory=set)
    is_bidirectional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "byte_count": self.byte_count,
            "protocols": list(self.protocols),
            "is_bidirectional": self.is_bidirectional,
        }


class TopKTracker:
    """
    Min-heap based priority queue that maintains the top-k items by value.

    Uses Python's ``heapq`` module (binary min-heap) so that each
    ``update()`` call runs in O(log k) time instead of re-sorting the
    full vertex list.

    Data Structures concept (Sem 3): **Priority Queue / Binary Heap**.
    """

    def __init__(self, k: int = 5):
        self.k = k
        # Min-heap of (value, key) tuples – smallest value on top.
        self._heap: List[Tuple[int, str]] = []
        # Fast lookup: key → current value
        self._map: Dict[str, int] = {}

    def update(self, key: str, value: int) -> None:
        """
        Insert or update *key* with *value*.

        The heap always keeps at most *k* entries.  When a key that is
        already inside the heap receives a higher value the entry is
        replaced (lazy-deletion style on next ``top_k`` call).
        """
        self._map[key] = value
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (value, key))
        else:
            # If new value beats the current minimum, replace it
            if value > self._heap[0][0]:
                heapq.heapreplace(self._heap, (value, key))

    def top_k(self) -> List[Tuple[str, int]]:
        """
        Return the current top-k items as ``[(key, value), ...]``
        sorted from highest to lowest value.

        Lazily rebuilds the heap to purge stale entries whose values
        have since been superseded by ``update()``.
        """
        # Rebuild from authoritative map to clear stale entries
        candidates = sorted(self._map.items(), key=lambda x: x[1], reverse=True)
        top = candidates[: self.k]
        # Reconstruct heap from the fresh top-k
        self._heap = [(v, k) for k, v in top]
        heapq.heapify(self._heap)
        return top

    def clear(self) -> None:
        self._heap.clear()
        self._map.clear()

    def __len__(self) -> int:
        return len(self._map)


class NetworkGraph:
    """
    Graph representation of network topology.

    Mathematical Foundations:
    - G = (V, E) where V is set of vertices (IPs) and E is set of edges (connections)
    - Adjacency Matrix: A[i][j] = weight if edge exists, 0 otherwise
    - Incidence Matrix: I[v][e] = 1 if vertex v is incident to edge e

    Used for:
    - Visualizing network topology
    - Identifying central nodes (high degree)
    - Detecting anomalies (unusual connectivity patterns)
    """

    # Internal IP ranges (RFC 1918)
    INTERNAL_RANGES = [
        ("10.0.0.0", "10.255.255.255"),
        ("172.16.0.0", "172.31.255.255"),
        ("192.168.0.0", "192.168.255.255"),
        ("127.0.0.0", "127.255.255.255"),
    ]

    def __init__(self):
        self.vertices: Dict[str, Vertex] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self._adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

        # --- Data Structures showcase (Sem 3) ---
        # Hash Map: Key = (src_ip, dst_ip), Value = cumulative byte count
        self.connection_byte_counts: Dict[Tuple[str, str], int] = {}

        # Priority Queue: real-time top-5 bandwidth hogs (min-heap)
        self.top_bandwidth_hogs: TopKTracker = TopKTracker(k=5)

    def add_vertex(self, ip: str) -> Vertex:
        """Add a vertex (IP) to the graph if it doesn't exist."""
        if ip not in self.vertices:
            self.vertices[ip] = Vertex(id=ip, is_internal=self._is_internal_ip(ip))
        return self.vertices[ip]

    def add_edge(
        self, src_ip: str, dst_ip: str, packet_size: int = 0, protocol: str = "TCP"
    ):
        """
        Add or update an edge (connection) in the graph.

        Updates:
        - Edge weight (packet count)
        - Vertex degrees
        - Byte counts
        """
        # Ensure vertices exist
        src_vertex = self.add_vertex(src_ip)
        dst_vertex = self.add_vertex(dst_ip)

        # Create or update edge
        edge_key = (src_ip, dst_ip)

        if edge_key not in self.edges:
            self.edges[edge_key] = Edge(source=src_ip, target=dst_ip)

            # Update adjacency lists
            self._adjacency_list[src_ip].add(dst_ip)
            self._reverse_adjacency[dst_ip].add(src_ip)

            # Update degrees
            src_vertex.out_degree += 1
            dst_vertex.in_degree += 1

        # Update edge properties
        edge = self.edges[edge_key]
        edge.weight += 1
        edge.byte_count += packet_size
        edge.protocols.add(protocol)

        # Update vertex bytes and packet counts
        src_vertex.packet_count += 1
        src_vertex.total_bytes += packet_size

        # Update connection byte-count hash map (IP Pair → Byte Count)
        self.connection_byte_counts[edge_key] = (
            self.connection_byte_counts.get(edge_key, 0) + packet_size
        )

        # Update priority queue – top-5 bandwidth hogs by total bytes
        self.top_bandwidth_hogs.update(src_ip, src_vertex.total_bytes)

        # Check for bidirectional edge
        reverse_key = (dst_ip, src_ip)
        if reverse_key in self.edges:
            self.edges[edge_key].is_bidirectional = True
            self.edges[reverse_key].is_bidirectional = True

    def get_vertex(self, ip: str) -> Optional[Vertex]:
        """Get a vertex by IP address."""
        return self.vertices.get(ip)

    def get_edge(self, src_ip: str, dst_ip: str) -> Optional[Edge]:
        """Get an edge by source and destination."""
        return self.edges.get((src_ip, dst_ip))

    def get_neighbors(self, ip: str, direction: str = "out") -> Set[str]:
        """
        Get neighboring vertices.

        Args:
            ip: IP address
            direction: "out" (destinations), "in" (sources), or "both"
        """
        if direction == "out":
            return self._adjacency_list.get(ip, set())
        elif direction == "in":
            return self._reverse_adjacency.get(ip, set())
        else:
            return self._adjacency_list.get(ip, set()) | self._reverse_adjacency.get(
                ip, set()
            )

    def get_degree(self, ip: str) -> int:
        """Get the degree of a vertex."""
        vertex = self.vertices.get(ip)
        return vertex.degree if vertex else 0

    def get_adjacency_matrix(self) -> Tuple[List[str], List[List[int]]]:
        """
        Generate adjacency matrix representation.

        Returns:
            Tuple of (list of IPs, 2D matrix)

        Mathematical representation:
        A[i][j] = edge weight from vertex i to vertex j
        """
        ips = sorted(self.vertices.keys())
        ip_to_idx = {ip: idx for idx, ip in enumerate(ips)}
        n = len(ips)

        matrix = [[0] * n for _ in range(n)]

        for (src, dst), edge in self.edges.items():
            i = ip_to_idx.get(src)
            j = ip_to_idx.get(dst)
            if i is not None and j is not None:
                matrix[i][j] = edge.weight

        return ips, matrix

    def get_top_vertices_by_degree(self, n: int = 10) -> List[Vertex]:
        """
        Get vertices with highest degree.

        High degree nodes may indicate:
        - Central servers (legitimate)
        - Port scanners or attackers (suspicious)
        """
        sorted_vertices = sorted(
            self.vertices.values(), key=lambda v: v.degree, reverse=True
        )
        return sorted_vertices[:n]

    def get_top_vertices_by_traffic(self, n: int = 10) -> List[Vertex]:
        """Get vertices with highest traffic volume."""
        sorted_vertices = sorted(
            self.vertices.values(), key=lambda v: v.total_bytes, reverse=True
        )
        return sorted_vertices[:n]

    def get_bandwidth_hogs(self, k: int = 5) -> List[Tuple[str, int]]:
        """
        Return the top-*k* bandwidth hogs using the internal priority queue.

        Each entry is ``(ip, total_bytes)`` sorted from highest to lowest.
        Runs in O(k log k) instead of O(n log n).
        """
        if k != self.top_bandwidth_hogs.k:
            # Rebuild tracker for a different k
            tracker = TopKTracker(k=k)
            for ip, v in self.vertices.items():
                tracker.update(ip, v.total_bytes)
            return tracker.top_k()
        return self.top_bandwidth_hogs.top_k()

    def get_isolated_vertices(self) -> List[Vertex]:
        """Get vertices with no connections (degree 0)."""
        return [v for v in self.vertices.values() if v.degree == 0]

    def is_connected(self, ip1: str, ip2: str) -> bool:
        """Check if two IPs have a direct connection."""
        return (ip1, ip2) in self.edges or (ip2, ip1) in self.edges

    def find_path(self, src_ip: str, dst_ip: str) -> Optional[List[str]]:
        """
        Find a path between two IPs using BFS.

        Returns None if no path exists.
        """
        if src_ip not in self.vertices or dst_ip not in self.vertices:
            return None

        if src_ip == dst_ip:
            return [src_ip]

        # BFS
        visited = {src_ip}
        queue = [(src_ip, [src_ip])]

        while queue:
            current, path = queue.pop(0)

            for neighbor in self.get_neighbors(current, "both"):
                if neighbor == dst_ip:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_connected_components(self) -> List[Set[str]]:
        """
        Find all connected components in the graph.

        Uses Union-Find algorithm for efficiency.
        """
        visited = set()
        components = []

        for ip in self.vertices:
            if ip not in visited:
                component = set()
                self._dfs(ip, visited, component)
                components.append(component)

        return components

    def _dfs(self, ip: str, visited: Set[str], component: Set[str]):
        """Depth-first search helper for connected components."""
        visited.add(ip)
        component.add(ip)

        for neighbor in self.get_neighbors(ip, "both"):
            if neighbor not in visited:
                self._dfs(neighbor, visited, component)

    def _is_internal_ip(self, ip: str) -> bool:
        """Check if an IP is in a private range."""
        try:
            parts = [int(p) for p in ip.split(".")]
            if len(parts) != 4:
                return False

            ip_int = (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]

            for start, end in self.INTERNAL_RANGES:
                start_parts = [int(p) for p in start.split(".")]
                end_parts = [int(p) for p in end.split(".")]

                start_int = (
                    (start_parts[0] << 24)
                    + (start_parts[1] << 16)
                    + (start_parts[2] << 8)
                    + start_parts[3]
                )
                end_int = (
                    (end_parts[0] << 24)
                    + (end_parts[1] << 16)
                    + (end_parts[2] << 8)
                    + end_parts[3]
                )

                if start_int <= ip_int <= end_int:
                    return True

            return False
        except (ValueError, IndexError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format (for JSON serialization)."""
        return {
            "vertices": [v.to_dict() for v in self.vertices.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "vertex_count": len(self.vertices),
            "edge_count": len(self.edges),
        }

    def to_vis_js_format(self) -> Dict[str, List]:
        """
        Export graph in vis.js compatible format for web visualization.

        Returns nodes and edges arrays for vis.js Network.
        """
        nodes = []
        for ip, vertex in self.vertices.items():
            # Size nodes by degree
            size = max(10, min(50, vertex.degree * 5))

            # Color by type
            if vertex.is_anomalous:
                color = "#ff4444"  # Red for anomalous
            elif vertex.is_internal:
                color = "#4488ff"  # Blue for internal
            else:
                color = "#44ff88"  # Green for external

            nodes.append(
                {
                    "id": ip,
                    "label": ip,
                    "size": size,
                    "color": color,
                    "title": f"Degree: {vertex.degree}<br>Packets: {vertex.packet_count}<br>Bytes: {vertex.total_bytes}",
                }
            )

        edges = []
        for (src, dst), edge in self.edges.items():
            # Width by packet count
            width = max(1, min(10, edge.weight // 10))

            edges.append(
                {
                    "from": src,
                    "to": dst,
                    "width": width,
                    "arrows": "to" if not edge.is_bidirectional else "to,from",
                    "title": f"Packets: {edge.weight}<br>Bytes: {edge.byte_count}<br>Protocols: {', '.join(edge.protocols)}",
                }
            )

        return {"nodes": nodes, "edges": edges}

    def clear(self):
        """Clear all vertices and edges."""
        self.vertices.clear()
        self.edges.clear()
        self._adjacency_list.clear()
        self._reverse_adjacency.clear()
        self.connection_byte_counts.clear()
        self.top_bandwidth_hogs.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.vertices:
            return {
                "vertex_count": 0,
                "edge_count": 0,
                "density": 0,
                "avg_degree": 0,
                "max_degree": 0,
                "components": 0,
            }

        degrees = [v.degree for v in self.vertices.values()]
        n = len(self.vertices)
        m = len(self.edges)

        # Graph density: E / (V * (V-1)) for directed graphs
        max_edges = n * (n - 1) if n > 1 else 1
        density = m / max_edges

        return {
            "vertex_count": n,
            "edge_count": m,
            "density": round(density, 4),
            "avg_degree": round(sum(degrees) / n, 2),
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "components": len(self.get_connected_components()),
            "internal_nodes": sum(1 for v in self.vertices.values() if v.is_internal),
            "external_nodes": sum(
                1 for v in self.vertices.values() if not v.is_internal
            ),
        }
