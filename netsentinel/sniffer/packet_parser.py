"""
Packet Parser Module - Parses raw network packets into structured data.

Applies Computer Networks concepts (Sem 3):
- TCP/IP packet structure
- Protocol headers (IP, TCP, UDP, ICMP)
- Network byte order and parsing
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from scapy.all import IP, TCP, UDP, ICMP, Raw
    from scapy.layers.inet import Ether

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


@dataclass
class PacketInfo:
    """
    Structured representation of a network packet.

    Maps to database schema:
    Traffic_Logs (ID, Src_IP, Dst_IP, Protocol, Packet_Size, Timestamp)
    """

    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    size: int
    flags: Optional[str] = None
    ttl: Optional[int] = None
    payload_preview: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "size": self.size,
            "flags": self.flags,
            "ttl": self.ttl,
        }

    def get_connection_key(self) -> tuple:
        """
        Get a unique key for this connection (bidirectional).
        Used for graph edge identification.
        """
        ips = tuple(sorted([self.src_ip, self.dst_ip]))
        return ips

    def get_directed_key(self) -> tuple:
        """Get a directed connection key (source -> dest)."""
        return (self.src_ip, self.dst_ip)


class PacketParser:
    """
    Parser for extracting information from raw network packets.

    Demonstrates understanding of:
    - IP Header structure (RFC 791)
    - TCP Header structure (RFC 793)
    - UDP Header structure (RFC 768)
    - ICMP Header structure (RFC 792)
    """

    # Protocol number mappings (from IP header)
    PROTOCOL_MAP = {
        1: "ICMP",
        6: "TCP",
        17: "UDP",
        41: "IPv6",
        47: "GRE",
        50: "ESP",
        51: "AH",
        58: "ICMPv6",
        89: "OSPF",
    }

    # TCP flag mappings
    TCP_FLAGS = {
        "F": "FIN",
        "S": "SYN",
        "R": "RST",
        "P": "PSH",
        "A": "ACK",
        "U": "URG",
        "E": "ECE",
        "C": "CWR",
    }

    def __init__(self, max_payload_preview: int = 100):
        """
        Initialize the parser.

        Args:
            max_payload_preview: Maximum bytes to store as payload preview
        """
        self.max_payload_preview = max_payload_preview

    def parse(self, packet) -> Optional[PacketInfo]:
        """
        Parse a scapy packet into PacketInfo.

        Args:
            packet: Scapy packet object

        Returns:
            PacketInfo or None if packet cannot be parsed
        """
        if not SCAPY_AVAILABLE:
            return None

        # Must have IP layer
        if not packet.haslayer(IP):
            return None

        ip_layer = packet[IP]

        # Extract basic IP information
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        size = len(packet)
        ttl = ip_layer.ttl

        # Default values
        src_port = 0
        dst_port = 0
        protocol = self.PROTOCOL_MAP.get(ip_layer.proto, f"UNKNOWN({ip_layer.proto})")
        flags = None
        payload_preview = None

        # Parse TCP layer
        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            protocol = "TCP"
            flags = self._parse_tcp_flags(tcp_layer)

            # Extract payload preview
            if packet.haslayer(Raw):
                raw_data = bytes(packet[Raw].load)
                payload_preview = raw_data[: self.max_payload_preview]

        # Parse UDP layer
        elif packet.haslayer(UDP):
            udp_layer = packet[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
            protocol = "UDP"

            if packet.haslayer(Raw):
                raw_data = bytes(packet[Raw].load)
                payload_preview = raw_data[: self.max_payload_preview]

        # Parse ICMP layer
        elif packet.haslayer(ICMP):
            protocol = "ICMP"
            icmp_layer = packet[ICMP]
            # ICMP uses type/code instead of ports
            src_port = icmp_layer.type
            dst_port = icmp_layer.code

        return PacketInfo(
            timestamp=datetime.now(),
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            size=size,
            flags=flags,
            ttl=ttl,
            payload_preview=payload_preview,
        )

    def _parse_tcp_flags(self, tcp_layer) -> str:
        """
        Parse TCP flags into human-readable format.

        TCP Flags (bits in the TCP header):
        - URG: Urgent pointer field is valid
        - ACK: Acknowledgment field is valid
        - PSH: Push function
        - RST: Reset the connection
        - SYN: Synchronize sequence numbers
        - FIN: No more data from sender
        """
        flags = []
        flag_str = str(tcp_layer.flags)

        if "S" in flag_str:
            flags.append("SYN")
        if "A" in flag_str:
            flags.append("ACK")
        if "F" in flag_str:
            flags.append("FIN")
        if "R" in flag_str:
            flags.append("RST")
        if "P" in flag_str:
            flags.append("PSH")
        if "U" in flag_str:
            flags.append("URG")

        return "-".join(flags) if flags else "NONE"

    @staticmethod
    def parse_raw_ip_header(raw_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Manually parse an IP header from raw bytes.

        This demonstrates understanding of IP header structure:

        0                   1                   2                   3
        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |Version|  IHL  |Type of Service|          Total Length         |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |         Identification        |Flags|      Fragment Offset    |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |  Time to Live |    Protocol   |         Header Checksum       |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                       Source Address                          |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                    Destination Address                        |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        """
        if len(raw_bytes) < 20:  # Minimum IP header size
            return None

        # Version and IHL (Internet Header Length)
        version_ihl = raw_bytes[0]
        version = version_ihl >> 4
        ihl = (version_ihl & 0x0F) * 4  # IHL is in 32-bit words

        if version != 4:
            return None  # Only support IPv4

        # Total length
        total_length = int.from_bytes(raw_bytes[2:4], "big")

        # TTL and Protocol
        ttl = raw_bytes[8]
        protocol = raw_bytes[9]

        # Source IP (bytes 12-15)
        src_ip = ".".join(str(b) for b in raw_bytes[12:16])

        # Destination IP (bytes 16-19)
        dst_ip = ".".join(str(b) for b in raw_bytes[16:20])

        return {
            "version": version,
            "header_length": ihl,
            "total_length": total_length,
            "ttl": ttl,
            "protocol": protocol,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
        }
