"""
Packet Sniffer Module - Captures network packets in real-time.

Uses scapy for packet capture and raw socket support.
Applies concepts from Computer Networks (Sem 3): TCP/IP, UDP, ICMP protocols.
"""

import threading
import queue
import time
from typing import Callable, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    from scapy.layers.inet import Ether

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: scapy not installed. Install with: pip install scapy")

from .packet_parser import PacketParser, PacketInfo


class PacketSniffer:
    """
    Real-time packet sniffer using scapy or raw sockets.

    Applies Operating Systems concepts (Sem 4):
    - Multithreading for non-blocking capture
    - Queue for thread-safe data transfer
    - Process synchronization
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        packet_queue: Optional[queue.Queue] = None,
    ):
        """
        Initialize the packet sniffer.

        Args:
            interface: Network interface to sniff on (None for default)
            packet_queue: Queue to put captured packets (for async processing)
        """
        self.interface = interface
        self.packet_queue = packet_queue or queue.Queue(maxsize=10000)
        self.parser = PacketParser()
        self._running = False
        self._sniffer_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[PacketInfo], None]] = []
        self._packet_count = 0
        self._start_time: Optional[float] = None

        # Statistics
        self.stats = {
            "total_packets": 0,
            "tcp_packets": 0,
            "udp_packets": 0,
            "icmp_packets": 0,
            "other_packets": 0,
            "total_bytes": 0,
        }

    def add_callback(self, callback: Callable[[PacketInfo], None]):
        """Add a callback function to be called for each captured packet."""
        self._callbacks.append(callback)

    def _process_packet(self, packet):
        """
        Process a captured packet.

        Applies Computer Networks concepts:
        - Extracting IP headers (Source IP, Dest IP)
        - Identifying Transport Layer protocol (TCP/UDP)
        - Parsing packet flags (SYN, ACK, FIN)
        """
        if not SCAPY_AVAILABLE:
            return

        try:
            packet_info = self.parser.parse(packet)
            if packet_info:
                # Update statistics
                self.stats["total_packets"] += 1
                self.stats["total_bytes"] += packet_info.size

                if packet_info.protocol == "TCP":
                    self.stats["tcp_packets"] += 1
                elif packet_info.protocol == "UDP":
                    self.stats["udp_packets"] += 1
                elif packet_info.protocol == "ICMP":
                    self.stats["icmp_packets"] += 1
                else:
                    self.stats["other_packets"] += 1

                # Put packet in queue for async processing
                try:
                    self.packet_queue.put_nowait(packet_info)
                except queue.Full:
                    # Queue is full, drop oldest packet
                    try:
                        self.packet_queue.get_nowait()
                        self.packet_queue.put_nowait(packet_info)
                    except:
                        pass

                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        callback(packet_info)
                    except Exception as e:
                        print(f"Callback error: {e}")

        except Exception as e:
            print(f"Error processing packet: {e}")

    def start(
        self, count: int = 0, timeout: Optional[int] = None, filter_str: str = ""
    ):
        """
        Start capturing packets in a separate thread.

        Args:
            count: Number of packets to capture (0 for infinite)
            timeout: Capture timeout in seconds
            filter_str: BPF filter string (e.g., "tcp port 80")
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError(
                "scapy is not installed. Please install it: pip install scapy"
            )

        if self._running:
            print("Sniffer is already running")
            return

        self._running = True
        self._start_time = time.time()

        def sniffer_worker():
            """Worker thread for packet capture."""
            try:
                sniff(
                    iface=self.interface,
                    prn=self._process_packet,
                    count=count if count > 0 else 0,
                    timeout=timeout,
                    filter=filter_str,
                    store=False,  # Don't store packets in memory
                    stop_filter=lambda _: not self._running,
                )
            except Exception as e:
                print(f"Sniffer error: {e}")
            finally:
                self._running = False

        self._sniffer_thread = threading.Thread(target=sniffer_worker, daemon=True)
        self._sniffer_thread.start()
        print(f"Packet sniffer started on interface: {self.interface or 'default'}")

    def stop(self):
        """Stop the packet capture."""
        self._running = False
        if self._sniffer_thread:
            self._sniffer_thread.join(timeout=2.0)
        print("Packet sniffer stopped")

    def is_running(self) -> bool:
        """Check if the sniffer is currently running."""
        return self._running

    def get_packet(self, timeout: float = 1.0) -> Optional[PacketInfo]:
        """
        Get a packet from the queue.

        Args:
            timeout: How long to wait for a packet

        Returns:
            PacketInfo or None if queue is empty
        """
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> dict:
        """Get capture statistics."""
        stats = self.stats.copy()
        if self._start_time:
            stats["runtime_seconds"] = time.time() - self._start_time
            stats["packets_per_second"] = stats["total_packets"] / max(
                stats["runtime_seconds"], 1
            )
        return stats

    def reset_stats(self):
        """Reset capture statistics."""
        self.stats = {
            "total_packets": 0,
            "tcp_packets": 0,
            "udp_packets": 0,
            "icmp_packets": 0,
            "other_packets": 0,
            "total_bytes": 0,
        }
        self._start_time = time.time()


class SimulatedSniffer(PacketSniffer):
    """
    Simulated packet sniffer for testing without requiring admin privileges.
    Generates fake network traffic data for demonstration purposes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simulation_thread: Optional[threading.Thread] = None

    def start(
        self, count: int = 0, timeout: Optional[int] = None, filter_str: str = ""
    ):
        """Start generating simulated packets."""
        import random

        if self._running:
            print("Simulated sniffer is already running")
            return

        self._running = True
        self._start_time = time.time()

        # Sample IP addresses for simulation
        internal_ips = [
            "192.168.1.1",
            "192.168.1.10",
            "192.168.1.20",
            "192.168.1.100",
            "10.0.0.1",
        ]
        external_ips = [
            "8.8.8.8",
            "1.1.1.1",
            "172.217.14.206",
            "151.101.1.69",
            "31.13.70.36",
        ]
        protocols = ["TCP", "UDP", "ICMP"]
        tcp_flags = ["SYN", "ACK", "SYN-ACK", "FIN", "RST", "PSH-ACK"]

        def simulation_worker():
            """Generate simulated packet data."""
            packet_num = 0
            while self._running:
                if count > 0 and packet_num >= count:
                    break

                # Generate random packet
                src_ip = random.choice(internal_ips + external_ips)
                dst_ip = random.choice(internal_ips + external_ips)
                while dst_ip == src_ip:
                    dst_ip = random.choice(internal_ips + external_ips)

                protocol = random.choices(protocols, weights=[0.6, 0.3, 0.1])[0]

                if protocol == "TCP":
                    src_port = random.choice(
                        [80, 443, 8080, random.randint(1024, 65535)]
                    )
                    dst_port = random.choice(
                        [80, 443, 22, 3306, random.randint(1024, 65535)]
                    )
                    flags = random.choice(tcp_flags)
                elif protocol == "UDP":
                    src_port = random.choice([53, 123, random.randint(1024, 65535)])
                    dst_port = random.choice(
                        [53, 123, 5353, random.randint(1024, 65535)]
                    )
                    flags = None
                else:  # ICMP
                    src_port = 0
                    dst_port = 0
                    flags = None

                # Random packet size (with occasional large packets)
                if random.random() < 0.05:
                    size = random.randint(5000, 65535)  # Large packet
                else:
                    size = random.randint(40, 1500)  # Normal packet

                packet_info = PacketInfo(
                    timestamp=datetime.now(),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=protocol,
                    size=size,
                    flags=flags,
                    ttl=random.choice([64, 128, 255]),
                    payload_preview=None,
                )

                # Update stats
                self.stats["total_packets"] += 1
                self.stats["total_bytes"] += size
                if protocol == "TCP":
                    self.stats["tcp_packets"] += 1
                elif protocol == "UDP":
                    self.stats["udp_packets"] += 1
                else:
                    self.stats["icmp_packets"] += 1

                # Add to queue
                try:
                    self.packet_queue.put_nowait(packet_info)
                except queue.Full:
                    try:
                        self.packet_queue.get_nowait()
                        self.packet_queue.put_nowait(packet_info)
                    except:
                        pass

                # Call callbacks
                for callback in self._callbacks:
                    try:
                        callback(packet_info)
                    except Exception as e:
                        print(f"Callback error: {e}")

                packet_num += 1

                # Random delay between packets
                time.sleep(random.uniform(0.01, 0.1))

            self._running = False

        self._simulation_thread = threading.Thread(
            target=simulation_worker, daemon=True
        )
        self._simulation_thread.start()
        print("Simulated packet sniffer started")

    def stop(self):
        """Stop the simulation."""
        self._running = False
        if self._simulation_thread:
            self._simulation_thread.join(timeout=2.0)
        print("Simulated packet sniffer stopped")
