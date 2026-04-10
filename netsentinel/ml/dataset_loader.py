"""
NSL-KDD Dataset Loader - Standard benchmark dataset for network IDS evaluation.

The NSL-KDD dataset is derived from the KDD Cup 1999 dataset and is a widely
used benchmark for evaluating Network Intrusion Detection Systems.

Dataset info:
    - 41 original features per connection record
    - 5 attack categories: Normal, DoS, Probe, R2L, U2R
    - Train set: KDDTrain+.txt (~125,973 records)
    - Test set : KDDTest+.txt  (~22,544 records)

Reference:
    Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
    "A detailed analysis of the KDD CUP 99 data set."
    IEEE Symposium on Computational Intelligence for Security and Defense Applications.
"""

import csv
import os
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

# NSL-KDD column names (41 features + class + difficulty)
NSL_KDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

# Mapping from NSL-KDD attack label → high-level category
ATTACK_CATEGORY: Dict[str, str] = {
    "normal": "normal",
    # DoS attacks
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "udpstorm": "dos",
    "processtable": "dos",
    "mailbomb": "dos",
    "worm": "dos",
    # Probe attacks
    "satan": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "mscan": "probe",
    "saint": "probe",
    # R2L attacks
    "guess_passwd": "r2l",
    "ftp_write": "r2l",
    "imap": "r2l",
    "phf": "r2l",
    "multihop": "r2l",
    "warezmaster": "r2l",
    "warezclient": "r2l",
    "spy": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "snmpguess": "r2l",
    "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    # U2R attacks
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "rootkit": "u2r",
    "perl": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r",
}

# Nominal (categorical) feature indices
NOMINAL_INDICES = [1, 2, 3]  # protocol_type, service, flag

# Protocol type values
PROTOCOL_TYPES = ["tcp", "udp", "icmp"]

# Service values (top services)
SERVICE_TYPES = [
    "http",
    "smtp",
    "finger",
    "domain_u",
    "auth",
    "telnet",
    "ftp",
    "ftp_data",
    "other",
    "private",
    "pop_3",
    "ntp_u",
    "eco_i",
    "ecr_i",
    "ssh",
    "imap4",
    "ctf",
    "login",
    "dns",
]

# Flag values
FLAG_TYPES = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"]


@dataclass
class NSLKDDRecord:
    """A single NSL-KDD connection record."""

    features: List[float]
    label: str  # "normal" or attack name
    category: str  # "normal", "dos", "probe", "r2l", "u2r"
    difficulty: int  # Difficulty level (1-21)


class NSLKDDLoader:
    """
    Loads and preprocesses the NSL-KDD dataset for IDS model training.

    The loader supports:
    - Parsing the standard CSV/TXT format
    - One-hot encoding of categorical features
    - Binary classification (normal vs. attack)
    - Multi-class classification (5 categories)
    - Mapping NSL-KDD features to NetSentinel's internal feature format

    Usage:
        loader = NSLKDDLoader()
        X_train, y_train = loader.load("path/to/KDDTrain+.txt", binary=True)
        X_test, y_test   = loader.load("path/to/KDDTest+.txt",  binary=True)
    """

    def __init__(self):
        self._protocol_map = {p: i for i, p in enumerate(PROTOCOL_TYPES)}
        self._service_map = {s: i for i, s in enumerate(SERVICE_TYPES)}
        self._flag_map = {f: i for i, f in enumerate(FLAG_TYPES)}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def load(
        self,
        filepath: str,
        binary: bool = True,
        max_records: Optional[int] = None,
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Load an NSL-KDD dataset file and return (X, y).

        Parameters
        ----------
        filepath : str
            Path to KDDTrain+.txt or KDDTest+.txt.
        binary : bool
            If True labels are "normal" / "anomaly".
            If False labels are one of the five categories.
        max_records : int | None
            Limit the number of records loaded (useful for quick tests).

        Returns
        -------
        X : list of float vectors
        y : list of label strings
        """
        records = self._parse_file(filepath, max_records)
        X = [r.features for r in records]
        if binary:
            y = ["normal" if r.category == "normal" else "anomaly" for r in records]
        else:
            y = [r.category for r in records]
        return X, y

    def load_records(
        self,
        filepath: str,
        max_records: Optional[int] = None,
    ) -> List[NSLKDDRecord]:
        """Load file and return structured records."""
        return self._parse_file(filepath, max_records)

    def map_to_netsentinel_features(
        self,
        record: NSLKDDRecord,
    ) -> List[float]:
        """
        Map a single NSL-KDD record to the 22-feature vector used by
        NetSentinel's TrafficFeatures.to_vector().

        Mapping heuristics (NSL-KDD → NetSentinel feature):
            src_bytes       → avg_packet_size, max_packet_size
            dst_bytes       → avg_bytes_per_connection
            duration        → avg_inter_arrival_time
            count           → packets_per_second  (approx.)
            protocol_type   → tcp_ratio, udp_ratio, icmp_ratio
            srv_count       → unique_dst_ports    (approximation)
            flag features   → syn_ratio, ack_ratio, fin_ratio, rst_ratio
        """
        raw = record.features
        # Original feature order after encoding:
        # 0: duration
        # 1-3: one-hot protocol (tcp, udp, icmp)
        # Next comes service one-hot, flag one-hot, then numeric features

        # For simplicity, use the _raw values kept in the record's features
        # The first numeric features up to index ~40 are:
        # We have already one-hot encoded, so map from the encoded vector.

        n_proto = len(PROTOCOL_TYPES)
        n_svc = len(SERVICE_TYPES)
        n_flag = len(FLAG_TYPES)
        offset = 1 + n_proto + n_svc + n_flag  # after categorical one-hots

        duration = raw[0]
        tcp_r = raw[1]  # one-hot tcp
        udp_r = raw[2]  # one-hot udp
        icmp_r = raw[3]  # one-hot icmp

        # Numeric block starts at 'offset' and maps to original cols 4-40
        def _num(idx: int) -> float:
            pos = offset + idx
            return raw[pos] if pos < len(raw) else 0.0

        src_bytes = _num(0)  # col 4 → src_bytes
        dst_bytes = _num(1)  # col 5 → dst_bytes

        count = _num(18)  # col 22 → count
        srv_count = _num(19)  # col 23 → srv_count

        # NetSentinel 22-dim vector mapping
        vector = [
            src_bytes,  # avg_packet_size
            0.0,  # std_packet_size
            0,  # min_packet_size
            src_bytes,  # max_packet_size
            duration,  # avg_inter_arrival_time
            0.0,  # std_inter_arrival_time
            count / max(duration, 1.0),  # packets_per_second
            tcp_r,  # tcp_ratio
            udp_r,  # udp_ratio
            icmp_r,  # icmp_ratio
            1,  # unique_src_ports
            int(srv_count),  # unique_dst_ports
            1.0 if srv_count < 1024 else 0.0,  # well_known_port_ratio
            int(count),  # unique_destinations
            1,  # unique_sources
            dst_bytes,  # avg_bytes_per_connection
            (
                raw[offset + 20] if offset + 20 < len(raw) else 0.0
            ),  # syn_ratio (serror_rate approx)
            1.0 if tcp_r else 0.0,  # ack_ratio
            0.0,  # fin_ratio
            (
                raw[offset + 22] if offset + 22 < len(raw) else 0.0
            ),  # rst_ratio (rerror_rate approx)
            12.0,  # hour_of_day (unknown, default noon)
            1.0,  # is_business_hours (unknown, default yes)
        ]
        return vector

    @staticmethod
    def get_dataset_info() -> Dict[str, Any]:
        """Return metadata about the NSL-KDD dataset."""
        return {
            "name": "NSL-KDD",
            "description": (
                "An improved version of the KDD Cup 1999 dataset for evaluating "
                "Network Intrusion Detection Systems. Redundant records are removed "
                "and difficulty levels are added."
            ),
            "num_features": 41,
            "num_classes_binary": 2,
            "num_classes_multi": 5,
            "categories": ["normal", "dos", "probe", "r2l", "u2r"],
            "train_file": "KDDTrain+.txt",
            "test_file": "KDDTest+.txt",
            "reference": (
                "Tavallaee et al. (2009). 'A detailed analysis of the KDD CUP 99 "
                "data set.' IEEE Symposium on Computational Intelligence for "
                "Security and Defense Applications."
            ),
            "download_url": ("https://www.unb.ca/cic/datasets/nsl.html"),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _parse_file(
        self, filepath: str, max_records: Optional[int] = None
    ) -> List[NSLKDDRecord]:
        """Parse a KDDTrain+.txt or KDDTest+.txt file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"NSL-KDD dataset file not found: {filepath}\n"
                f"Download from: https://www.unb.ca/cic/datasets/nsl.html"
            )

        records: List[NSLKDDRecord] = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if max_records is not None and i >= max_records:
                    break
                if len(row) < 42:
                    continue

                features = self._encode_row(row)
                label = row[41].strip().lower()
                difficulty = int(row[42]) if len(row) > 42 else 0
                category = ATTACK_CATEGORY.get(label, "unknown")

                records.append(
                    NSLKDDRecord(
                        features=features,
                        label=label,
                        category=category,
                        difficulty=difficulty,
                    )
                )

        return records

    def _encode_row(self, row: List[str]) -> List[float]:
        """
        Encode a raw CSV row into a numeric feature vector.

        Categorical features (protocol_type, service, flag) are one-hot encoded.
        Numeric features are converted to float.
        """
        features: List[float] = []

        for i, val in enumerate(row[:41]):
            val = val.strip()
            if i == 1:  # protocol_type → one-hot
                features.extend(self._one_hot(val, PROTOCOL_TYPES))
            elif i == 2:  # service → one-hot
                features.extend(self._one_hot(val, SERVICE_TYPES))
            elif i == 3:  # flag → one-hot
                features.extend(self._one_hot(val, FLAG_TYPES))
            else:
                try:
                    features.append(float(val))
                except ValueError:
                    features.append(0.0)

        return features

    @staticmethod
    def _one_hot(value: str, categories: List[str]) -> List[float]:
        """Create a one-hot encoded vector."""
        vec = [0.0] * len(categories)
        value_lower = value.lower()
        for idx, cat in enumerate(categories):
            if cat.lower() == value_lower:
                vec[idx] = 1.0
                break
        return vec
