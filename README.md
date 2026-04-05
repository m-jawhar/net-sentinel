# 🔍 NetSentinel: AI-Driven Network Traffic Analyzer & Anomaly Detector

A comprehensive real-time network monitoring system that captures traffic, visualizes network topology as a graph, stores data in a database, and uses Machine Learning to detect anomalies.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Technical Deep Dive](#-technical-deep-dive)
- [Academic Concepts Applied](#-academic-concepts-applied)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

## ✨ Features

### 📡 Real-Time Packet Capture

- Capture network packets using Scapy
- Parse TCP, UDP, and ICMP protocols
- Extract source/destination IPs, ports, flags, and payload
- Thread-safe packet processing with queues

### 💾 Database Storage

- SQLite database for persistent storage
- Optimized schema with proper indexing
- Bulk insert for performance
- Aggregation queries for analytics

### 🕸️ Network Graph Visualization

- Represent network as vertices (IPs) and edges (connections)
- Calculate degree centrality and clustering coefficients
- Detect hub nodes and isolated components
- Export to vis.js format for web visualization

### 🤖 ML-Powered Anomaly Detection

- **Statistical Detection**: Z-score based outlier detection
- **Rule-Based Detection**: Security heuristics (port scan, DDoS, SYN flood patterns)
- **Machine Learning**: Decision Tree (ID3), K-Means Clustering, Naive Bayes, Linear Regression
- **Ensemble Scoring**: Weighted combination of all 3 methods

### 📊 Interactive Dashboard

- Streamlit-based web interface with four tabs
- Live traffic statistics with auto-refresh
- Real-time capture-to-database pipeline (thread-safe)
- Network topology visualization (interactive vis.js graph)
- Alert management with severity filtering
- Analytics with per-IP analysis and protocol breakdown

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NetSentinel Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   Sniffer    │────▶│   Database  │────▶│  Dashboard   │     │
│  │  (Capture)   │     │   (Storage)  │     │    (UI)      │     │
│  └──────┬───────┘     └──────────────┘     └──────────────┘     │
│         │                     │                    ▲            │
│         ▼                     ▼                    │            │
│  ┌──────────────┐     ┌──────────────┐             │            │
│  │    Graph     │────▶│   ML Engine  │────────────┘            │
│  │  (Topology)  │     │  (Anomaly)   │                          │
│  └──────────────┘     └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Administrator/root privileges (for real packet capture)

### Setup

```bash
# Clone or navigate to the project
cd C:\Projects\net-sentinel

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Install (Minimal)

```bash
pip install scapy streamlit
```

## 🚀 Quick Start

### 1. Start the Dashboard

```bash
python main.py --mode dashboard
```

Open http://localhost:8501 in your browser.

### 2. Capture Traffic (CLI)

```bash
python main.py --mode capture
```

**Note**: Uses simulated traffic by default. For real capture, run as administrator.

### 3. Train ML Model

```bash
python main.py --mode train
```

### 4. Analyze Data

```bash
python main.py --mode analyze
```

## 📖 Usage

### Command Line Options

```bash
python main.py --help

Options:
  --mode, -m       Operation mode: dashboard, capture, train, analyze
  --config, -c     Path to configuration file
  --interface, -i  Network interface for capture
  --db, -d         Database path
  --port, -p       Dashboard port
```

### Configuration

Create a `config.json` file:

```json
{
  "database_path": "data/netsentinel.db",
  "sniffer_interface": null,
  "enable_statistical_detection": true,
  "enable_rule_detection": true,
  "enable_ml_detection": true,
  "zscore_threshold": 3.0,
  "dashboard_port": 8501,
  "log_level": "INFO"
}
```

### Python API Usage

```python
from netsentinel.sniffer import PacketSniffer, SimulatedSniffer
from netsentinel.database import DatabaseManager
from netsentinel.visualizer import NetworkGraph, GraphAnalyzer
from netsentinel.ml import AnomalyDetector

# Initialize components
db = DatabaseManager("data/traffic.db")
graph = NetworkGraph()
detector = AnomalyDetector()

# Start capture
sniffer = SimulatedSniffer()

def on_packet(packet_info):
    # Add to graph
    graph.add_edge(packet_info.src_ip, packet_info.dst_ip, packet_info.size)

    # Check for anomalies
    anomalies = detector.check_packet(packet_info)
    for anomaly in anomalies:
        print(f"ALERT: {anomaly.detection_method} - {anomaly.score}")

sniffer.add_callback(on_packet)
sniffer.start()
```

## 🔬 Technical Deep Dive

### Packet Parsing

The sniffer extracts these fields from each packet:

| Field            | Description                     | Layer               |
| ---------------- | ------------------------------- | ------------------- |
| Source IP        | Sender's IP address             | Network (IP)        |
| Destination IP   | Receiver's IP address           | Network (IP)        |
| Source Port      | Sender's port                   | Transport (TCP/UDP) |
| Destination Port | Receiver's port                 | Transport (TCP/UDP) |
| Protocol         | TCP, UDP, or ICMP               | Transport           |
| Flags            | TCP flags (SYN, ACK, FIN, etc.) | Transport (TCP)     |
| TTL              | Time To Live                    | Network (IP)        |
| Size             | Packet size in bytes            | All                 |

### Graph Analysis

The network graph uses these metrics:

- **Degree Centrality**: `C(v) = degree(v) / (n-1)`
- **Clustering Coefficient**: `C(v) = 2|e| / (k(k-1))` where k = degree
- **Graph Density**: `D = 2E / (V(V-1))`

### Anomaly Detection Methods

1. **Statistical (Z-Score)**
   - Calculates mean and standard deviation of packet sizes
   - Flags packets with Z-score > 3σ

2. **Rule-Based**
   - Port scan: > 20 unique destination ports
   - DDoS: > 100 packets/minute from single source
   - Suspicious ports: Known malware ports (4444, 31337, etc.)

3. **Machine Learning**
   - Decision Tree (ID3) with information gain
   - K-Means clustering for unsupervised anomaly detection
   - Naive Bayes for probabilistic classification

### Mathematical Foundation — Entropy & Information Gain

The Decision Tree (ID3) algorithm selects the best feature to split on at each node
by maximising **Information Gain**, which is derived from **Shannon Entropy**.

#### Shannon Entropy

Entropy measures the impurity (uncertainty) of a set _S_ of labelled samples:

```
H(S) = − Σ p_i · log₂(p_i)       for each class i in S
```

Where `p_i = |S_i| / |S|` is the proportion of samples belonging to class _i_.

**Worked example** — Given 100 traffic windows: 70 normal, 30 anomaly:

```
p_normal  = 70 / 100 = 0.70
p_anomaly = 30 / 100 = 0.30

H(S) = −(0.70 × log₂ 0.70 + 0.30 × log₂ 0.30)
     = −(0.70 × (−0.5146) + 0.30 × (−1.7370))
     = −(−0.3602 + (−0.5211))
     = 0.8813 bits
```

A perfectly pure set has entropy 0; a 50/50 split has entropy 1.0 (maximum for
binary classification).

#### Information Gain

Information Gain (IG) for an attribute _A_ measures how much entropy is reduced
after splitting _S_ on _A_:

```
IG(S, A) = H(S) − Σ ( |S_v| / |S| ) · H(S_v)       for each value v of A
```

The ID3 algorithm greedily picks the attribute with the **highest IG** at each
node. In our implementation the algorithm also supports a `max_depth` limit and
employs a majority-vote leaf when all features are exhausted.

#### Feature Importance

After tree construction, each feature's importance is computed as the total
information gain attributed to splits on that feature, normalised to sum to 1:

```
Importance(f) = Σ  IG_node(f) · |S_node| / |S_root|
                 nodes that split on f
```

### Benchmark Dataset — NSL-KDD

NetSentinel includes a built-in loader for the **NSL-KDD** benchmark dataset
(`netsentinel/ml/dataset_loader.py`), the de-facto standard for evaluating IDS
models. Use the `--nsl-kdd` CLI flag to train on the benchmark data:

```bash
python main.py --mode train --nsl-kdd path/to/KDDTrain+.txt
```

The loader one-hot encodes categorical features, maps records to NetSentinel's
22-dimensional feature vector, and supports both binary (normal/anomaly) and
multi-class (normal, dos, probe, r2l, u2r) labelling.

### SQL Triggers for Auto-Alerting

The database schema includes two SQLite triggers that fire automatically on
INSERT operations, enabling zero-latency alert escalation:

| Trigger                   | Event                          | Action                                                            |
| ------------------------- | ------------------------------ | ----------------------------------------------------------------- |
| `trg_alert_high_priority` | Alert with severity ≥ 3 (HIGH) | Appends `[AUTO-ESCALATED at <datetime>]` to the alert description |
| `trg_large_packet_flag`   | Traffic log with size > 10 000 | Sets `is_anomaly = 1` for the inserted record                     |

## 📚 Academic Concepts Applied

This project demonstrates concepts from multiple courses:

### Computer Networks (Semester 3)

- TCP/IP protocol stack
- Packet structure and headers
- Transport layer protocols (TCP, UDP)
- Network layer addressing (IP)

### Data Structures (Semester 3)

- Hash Maps for connection tracking
- Priority Queues for top-N queries
- Graph data structure (adjacency list)

### DBMS (Semester 4)

- Relational schema design
- SQL queries (DDL, DML)
- Indexing for query optimization
- Transaction management

### Graph Theory (Semester 4)

- Vertices and edges
- Degree and centrality measures
- Adjacency matrix representation
- Connected components (BFS/DFS)

### Machine Learning (Semester 4)

- Feature extraction and engineering
- Decision Trees (ID3 algorithm with entropy)
- K-Means clustering
- Classification metrics (accuracy, precision, recall, F1)

### Operating Systems (Semester 4)

- Multithreading for concurrent capture
- Thread-safe queues
- Process synchronization

#### Process Synchronization — Deep Dive

NetSentinel uses the classic **Producer-Consumer pattern** implemented with Python's
`threading` and `queue` modules:

```
┌──────────────┐    queue.Queue     ┌──────────────────┐
│   Sniffer    │ ──── put() ────▶  │   Main Thread     │
│  (Producer)  │                    │   (Consumer)      │
│  daemon=True │ ◀── get()  ────   │   processes pkts  │
└──────────────┘                    └──────────────────┘
```

**Why `queue.Queue` instead of explicit mutexes?**

Python's `queue.Queue` is internally protected by a `threading.Lock` (mutex) and
a `threading.Condition` for blocking `get()` / `put()` calls. By using the
high-level Queue API the application gains:

| Concern           | How it is solved                                                           |
| ----------------- | -------------------------------------------------------------------------- |
| Mutual Exclusion  | `Queue` wraps `deque` with an internal `Lock`                              |
| Bounded Buffer    | `maxsize` parameter limits memory usage                                    |
| Blocking / Wakeup | `Condition.wait()` / `Condition.notify()` inside `get()`/`put()`           |
| Graceful Shutdown | `stop_event` (`threading.Event`) signals the producer thread               |
| Thread Cleanup    | Producer is a **daemon thread** — it exits when the main thread terminates |

**Race-condition avoidance:** The sniffer thread only calls `queue.put()` and
never reads shared state; the main thread only calls `queue.get()`. The two
threads share no mutable data structures outside the queue, so no additional
locking is needed.

**Database thread safety:** `DatabaseManager` uses thread-local storage
(`threading.local()`) to give each thread its own SQLite connection, with WAL
(Write-Ahead Logging) mode enabled for concurrent readers.

## 📁 Project Structure

```
net-sentinel/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
├── config.json            # Configuration (optional)
│
├── netsentinel/              # Main package
│   ├── __init__.py
│   │
│   ├── sniffer/           # Packet capture module
│   │   ├── __init__.py
│   │   ├── packet_sniffer.py
│   │   └── packet_parser.py
│   │
│   ├── database/          # Data storage module
│   │   ├── __init__.py
│   │   ├── db_manager.py
│   │   └── models.py
│   │
│   ├── visualizer/        # Graph visualization module
│   │   ├── __init__.py
│   │   ├── network_graph.py
│   │   └── graph_analyzer.py
│   │
│   ├── ml/                # Machine learning module
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   ├── model_trainer.py
│   │   ├── anomaly_detector.py
│   │   └── dataset_loader.py   # NSL-KDD benchmark dataset loader
│   │
│   ├── dashboard/         # Web interface
│   │   ├── __init__.py
│   │   └── app.py
│   │
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
│
├── data/                  # Database storage
│   └── netsentinel.db
│
├── models/                # Trained ML models
│   ├── decision_tree.pkl
│   ├── kmeans.pkl
│   ├── naive_bayes.pkl
│   └── linear_regression.pkl
│
├── logs/                  # Log files
│   └── netsentinel.log
│
└── tests/                 # Unit tests (83 tests across 4 files)
    └── ...
```

## 🧪 Testing

```bash
# Run all tests (83 tests)
pytest tests/

# Run with coverage
pytest tests/ --cov=netsentinel --cov-report=html
```

## 🔒 Security Considerations

- **Packet capture requires administrator privileges** on most systems
- The simulated sniffer mode is safe for development/demo
- Be mindful of privacy when capturing real network traffic
- Store captured data securely

## 🚧 Future Enhancements

- [x] Real-time web-based graph visualization with vis.js
- [ ] Deep learning models (LSTM, Autoencoder)
- [ ] Integration with threat intelligence feeds
- [ ] Export to PCAP format
- [ ] REST API for external integrations
- [ ] Docker containerization

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Scapy library for packet manipulation
- Streamlit for the dashboard framework
- Network security research community

---

**Built with ❤️ for learning network security and machine learning**
