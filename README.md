# WarehouseGPT — Physical AI Operating System for Warehouses

WarehouseGPT is a multi-modal AI platform for autonomous warehouse operations. It combines NVIDIA Isaac Sim 5.1 for simulation, a real-time digital twin with live Bird's-Eye-View rendering, a GPT-4o-powered natural language agent, safety AI (fire, near-miss, zone violation), and a full Prometheus + Grafana monitoring stack.

---

## What Is Running Right Now

| Service | URL | Description |
|---|---|---|
| WarehouseGPT Chat UI | `http://localhost:8080` | GPT-4o agent with real slot data |
| Digital Twin API | `http://localhost:8003/docs` | FastAPI — state, BEV, incidents, metrics |
| Live BEV Map | `http://localhost:8003/bev/live` | Auto-refreshing overhead map |
| Warehouse State | `http://localhost:8003/state` | Live agent positions (JSON) |
| Incidents Feed | `http://localhost:8003/incidents` | Active safety incidents |
| Prometheus | `http://localhost:9090` | Metrics scraping + alerting |
| Grafana | `http://localhost:3001` | Operations dashboard (`admin` / `change_me_strong_grafana_password`) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OPERATOR LAYER                                │
│   WarehouseGPT Chat UI (WebSocket)  │  REST API (FastAPI)               │
│          http://localhost:8080      │  http://localhost:8003             │
└────────────────────┬────────────────────────────┬───────────────────────┘
                     │                            │
        ┌────────────▼────────────┐   ┌───────────▼──────────────────────┐
        │    AGENT (GPT-4o)       │   │   DIGITAL TWIN API               │
        │  warehouse_agent/       │   │   digital_twin/api/main.py        │
        │  ├─ agent.py            │   │   ├─ GET  /state                  │
        │  ├─ tools/              │   │   ├─ GET  /state/bev              │
        │  │   ├─ inventory       │   │   ├─ GET  /bev/live               │
        │  │   ├─ find_item       │   │   ├─ GET  /incidents              │
        │  │   └─ slot_contents   │   │   ├─ GET  /metrics/               │
        │  ├─ memory/episodic     │   │   ├─ POST /inject                 │
        │  └─ rag/knowledge_base  │   │   ├─ POST /inject/camera          │
        └─────────────────────────┘   │   └─ WS   /stream                │
                     │                └───────────┬──────────────────────┘
        ┌────────────▼────────────┐               │ HTTP POST /inject
        │    KNOWLEDGE PLANE      │   ┌───────────▼──────────────────────┐
        │  ChromaDB (embeddings)  │   │   ISAAC SIM 5.1                  │
        │  Neo4j (scene graph)    │   │   scripts/isaac_sim_bridge.py     │
        │  user_data/info_slots   │   │   ForkliftB + Dingo AMR assets    │
        └─────────────────────────┘   │   Viewport capture → /inject/camera│
                                      └──────────────────────────────────┘
                     │
        ┌────────────▼────────────────────────────────────────────────────┐
        │                     MONITORING STACK                             │
        │  Prometheus :9090 ←── scrapes /metrics/ ──→ Digital Twin API    │
        │  Grafana    :3001 ←── queries Prometheus ──→ Operations Dashboard│
        │  Alerts: HighIncidentRate, FireDetected, NoAgentsVisible, ...    │
        └─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation | NVIDIA Isaac Sim 5.1 (ForkliftB + Dingo AMR USD assets) |
| Language Agent | OpenAI GPT-4o (function calling) |
| API Gateway | FastAPI + Uvicorn |
| Safety AI | OpenCV fire/near-miss/zone-violation pipeline |
| Metrics | Prometheus + `prometheus_client` |
| Dashboards | Grafana 11 (auto-provisioned) |
| Tracing | OpenTelemetry Collector (OTLP → Prometheus) |
| Vector Store | ChromaDB |
| Graph Database | Neo4j (stub mode if not running) |
| Relational DB | PostgreSQL 15 (TimescaleDB optional) |
| Cache / Pub-Sub | Redis |
| Computer Vision | OpenCV (BEV rendering, safety detection) |
| GPU | RTX 4090, CUDA 12.8, Driver 570 |

---

## Repository Layout

```
warehousegpt/
├── warehouse_agent/
│   ├── agent.py                      # GPT-4o agentic loop (function calling)
│   ├── api/
│   │   ├── main.py                   # FastAPI app — WebSocket /chat, POST /query
│   │   └── static/index.html         # Chat UI (dark industrial theme)
│   ├── tools/
│   │   └── warehouse_tools.py        # Tools: inventory, find_item, slot_contents
│   ├── memory/
│   │   ├── episodic.py               # ChromaDB semantic memory
│   │   └── working.py                # Sliding conversation window
│   └── rag/
│       ├── knowledge_base.py         # ChromaDB RAG over safety docs
│       └── knowledge_graph.py        # Neo4j causal chain queries
├── digital_twin/
│   ├── api/main.py                   # Digital twin FastAPI gateway + Prometheus metrics
│   ├── state_estimation/
│   │   └── warehouse_state.py        # WarehouseState, BEV renderer
│   ├── sensor_fusion/
│   │   ├── multi_object_tracker.py
│   │   └── kalman_filter.py
│   ├── ingestion/
│   │   ├── ros2_bridge.py            # ROS 2 bridge (optional)
│   │   └── camera_stream.py
│   ├── analytics/
│   │   └── throughput.py             # KPI metrics
│   ├── storage/
│   │   ├── state_store.py            # Redis state cache
│   │   └── event_store.py            # PostgreSQL event log
│   └── world_model_sync.py           # Prediction + blending loop
├── safety_ai/                        # Fire, near-miss, zone violation detectors
├── world_model/                      # VQ-VAE + transformer world model
├── forklift_rl/                      # PPO / DreamerV3 RL environments
├── scripts/
│   └── isaac_sim_bridge.py           # Isaac Sim bridge (real USD assets, camera capture)
├── infra/
│   └── monitoring/
│       ├── prometheus.yml            # Scrape config (DT API + self)
│       ├── rules/
│       │   └── warehouse_alerts.yml  # 6 alert rules
│       ├── otel-collector.yaml       # OTLP → Prometheus exporter
│       └── grafana/
│           ├── provisioning/
│           │   ├── datasources/      # Auto-provision Prometheus datasource
│           │   └── dashboards/       # File-based dashboard provider
│           └── dashboards/
│               └── warehouse_ops.json # WarehouseGPT Operations dashboard
├── user_data/
│   └── info_slots.json               # Real warehouse slot → box mapping
├── tests/                            # 40 tests, all passing
├── docker-compose.yml                # Postgres, Redis, Neo4j, Prometheus, Grafana
├── .env                              # API keys and config
└── pyproject.toml
```

---

## Quick Start

### Prerequisites

- Ubuntu 22.04 LTS
- NVIDIA GPU, CUDA 12.x (RTX 4090 tested)
- Python 3.10+
- Docker + Docker Compose
- NVIDIA Isaac Sim 5.1 at `~/isaacsim_5_1/` (optional)
- OpenAI API key

### 1. Start infrastructure (Postgres, Redis, Prometheus, Grafana)

```bash
cd warehousegpt
docker compose up -d
# Grafana → http://localhost:3001  (admin / change_me_strong_grafana_password)
# Prometheus → http://localhost:9090
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn openai chromadb sentence-transformers \
            opencv-python-headless pydantic-settings asyncpg redis \
            prometheus_client python-multipart \
            pytest pytest-asyncio pytest-mock
```

### 3. Configure API keys

```bash
# Edit .env and set:
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
```

### 4. Start the Digital Twin API

```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn digital_twin.api.main:app --port 8003
# Open http://localhost:8003/bev/live
# Metrics at http://localhost:8003/metrics/
```

### 5. Seed demo data

```bash
curl -X POST http://localhost:8003/demo/seed
# Then open http://localhost:8003/bev/live
# And http://localhost:3001/d/warehousegpt-ops for the Grafana dashboard
```

### 6. Start the WarehouseGPT chat agent

```bash
# In a second terminal:
source .venv/bin/activate
uvicorn warehouse_agent.api.main:app --port 8080 --reload
# Open http://localhost:8080
```

### 7. Connect Isaac Sim (optional)

1. Open Isaac Sim 5.1
2. **Window → Script Editor**
3. Open and run `scripts/isaac_sim_bridge.py`
4. The bridge auto-loads the Isaac Warehouse scene if the stage is empty, then spawns ForkliftB + Dingo AMR on top
5. Forklifts and AMRs appear on the BEV map; viewport frames are sent to fire detection every 10 ticks

You can also pre-load the scene manually before running the script:
```
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Stages/IsaacWarehouse.usd
```

No ROS 2 required — uses HTTP injection (`POST /inject`).

---

## Digital Twin API

| Endpoint | Description |
|---|---|
| `GET  /health` | Liveness probe |
| `GET  /state` | Current warehouse state (JSON) |
| `GET  /state/bev` | Bird's-Eye-View map (JPEG) |
| `GET  /bev/live` | Auto-refreshing live BEV viewer (HTML) |
| `GET  /incidents` | Active safety incidents |
| `GET  /analytics/throughput` | KPI metrics |
| `GET  /metrics/` | Prometheus metrics endpoint |
| `POST /inject` | Push agent state from Isaac Sim or any source |
| `POST /inject/camera` | Push camera frame for fire detection (multipart JPEG/PNG) |
| `POST /demo/seed` | Inject mock data for testing |
| `WS   /stream` | Real-time state stream at 10 Hz |

### Camera endpoint example

```bash
curl -X POST http://localhost:8003/inject/camera \
  -F "frame=@/path/to/frame.jpg" \
  -F "camera_id=cam_0" \
  -F "frame_index=42"
```

---

## Grafana Operations Dashboard

The `WarehouseGPT — Operations` dashboard (`/d/warehousegpt-ops`) is auto-provisioned and contains:

| Panel | Query |
|---|---|
| Active Forklifts / Workers / AMRs | `warehouse_agents_total` gauge by type |
| Incidents Last Hour | `round(sum(increase(warehouse_incidents_total[1h])))` |
| Fires Detected (24 h) | `sum(increase(warehouse_fires_detected_total[24h]))` |
| Inject Rate (frames/s) | `rate(warehouse_inject_requests_total[1m])` |
| Agent Count Over Time | Timeseries per agent type |
| Incident Rate by Type | Near-miss / collision / zone-violation / fire per minute |
| Incident Severity Breakdown | Pie chart: critical / high / medium |
| /inject Latency p50/p95/p99 | Histogram quantiles |
| Camera Frames / Fire Events | Camera throughput vs fire detection rate |

### Alert rules

| Alert | Condition |
|---|---|
| `HighIncidentRate` | > 5 incidents/min for 2 min |
| `CriticalIncident` | Any critical-severity incident |
| `FireDetected` | Any fire event in 5 min |
| `NoAgentsVisible` | 0 agents for 5 min |
| `DigitalTwinAPIDown` | Scrape target down for 1 min |
| `HighInjectLatency` | p95 > 500 ms |

---

## Isaac Sim Integration

The bridge script (`scripts/isaac_sim_bridge.py`) runs inside Isaac Sim's Script Editor and:

- Loads the official NVIDIA Isaac Warehouse scene automatically if the stage is empty:
  - **Scene**: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Stages/IsaacWarehouse.usd`
- Spawns real robot USD assets from S3 on top of the warehouse scene:
  - **Forklift**: `ForkliftB/forklift_b_sensor.usd` (Isaac/5.1 S3)
  - **AMR**: `Clearpath/Dingo/dingo.usd` (Isaac/5.1 S3)
- Animates agents along waypoint paths at configurable speeds
- Posts positions to `POST /inject` every 100 ms
- Captures the viewport as JPEG every 10 ticks and posts to `POST /inject/camera` for fire detection

Agent configuration (`SPECS` in the bridge):

| Prim | Asset | Start X | Start Y | Speed |
|---|---|---|---|---|
| `/World/Forklifts/Forklift_01` | ForkliftB | 8 | 25 | 2.5 m/s |
| `/World/AMRs/AMR_01` | Dingo | 30 | 22 | 0.5 m/s |
| `/World/AMRs/AMR_02` | Dingo | 20 | 10 | 1.5 m/s |

---

## Agent Capabilities

The GPT-4o agent has access to real warehouse data via function calling:

| Tool | Description |
|---|---|
| `get_inventory_status` | Zone-level inventory summary from `info_slots.json` |
| `find_item_location` | Look up which slot a box ID is in |
| `get_slot_contents` | List all boxes in a specific slot |
| `get_robot_fleet_status` | Fleet health and active alerts |
| `get_pending_orders` | Order queue and priorities |
| `create_maintenance_ticket` | Log a maintenance request |

Example queries:
```
Where is box 1234?
How many boxes are in zone A?
What is in slot B/03/1/2/04?
Which robots need maintenance?
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
# 40 tests, all passing
```

---

## Current Status

| Component | Status |
|---|---|
| GPT-4o agent with function calling | Working |
| WebSocket streaming chat UI | Working |
| Real warehouse slot data (`info_slots.json`) | Connected |
| Digital twin API | Working |
| Live BEV map (auto-refresh) | Working |
| Isaac Sim bridge (ForkliftB + Dingo USD assets) | Working |
| Camera frame injection (`/inject/camera`) | Working |
| Safety AI — fire detection | Working (colour heuristic + CNN pipeline) |
| Safety AI — near-miss / zone violation | Working |
| Prometheus metrics (`/metrics/`) | Working |
| Grafana operations dashboard | Working (auto-provisioned) |
| Prometheus alert rules | Configured (6 alerts) |
| PostgreSQL event store | Working (TimescaleDB optional) |
| Redis state cache | Working |
| World model (VQ-VAE + transformer) | Code written, training not started |
| Forklift RL environments | Code written, policies untrained |
| Neo4j knowledge graph | Stub mode (no DB running) |

---

## Environment Variables

Key variables in `.env`:

```bash
OPENAI_API_KEY=sk-proj-...          # Required
OPENAI_MODEL=gpt-4o                 # Required
NEO4J_URI=bolt://localhost:7687     # Optional
REDIS_URL=redis://localhost:6380    # Mapped to 6380 (avoid conflicts)
DATABASE_URL=postgresql://...       # Mapped to port 5434
GF_SECURITY_ADMIN_PASSWORD=change_me_strong_grafana_password
```

See `.env` for the full list.
