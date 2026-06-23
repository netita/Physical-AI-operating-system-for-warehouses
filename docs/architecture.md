# WarehouseGPT — Technical Architecture

## 1. System Overview

WarehouseGPT is a layered Physical AI operating system that wraps a real warehouse in a continuously synchronized digital twin, runs perception and safety inference at the edge, orchestrates autonomous forklifts via reinforcement-learned policies, and exposes a natural-language control plane powered by Claude Sonnet 4.6.

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                            OPERATOR / INTEGRATION LAYER                             ║
║                                                                                      ║
║   WebApp Dashboard (React)     REST/WebSocket (FastAPI :8000)     ERP/WMS Webhooks  ║
╚═══════════════════════════════════════════╤══════════════════════════════════════════╝
                                            │ JWT-authenticated requests
╔═══════════════════════════════════════════▼══════════════════════════════════════════╗
║                              WAREHOUSEGPT AGENT PLANE                                ║
║                                                                                      ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐    ║
║  │              WarehouseGPT Agent (Claude Sonnet 4.6 + LangGraph)             │    ║
║  │                                                                             │    ║
║  │  NL Instruction → Task Decomposition → Tool Dispatch → Response Generation │    ║
║  │                                                                             │    ║
║  │  Tools: query_scene_graph | dispatch_forklift | run_safety_check |          │    ║
║  │         get_inventory | create_mission | query_rag | get_twin_state         │    ║
║  └───────────────────────────────┬─────────────────────────────────────────────┘    ║
║                                  │ LangChain tool calls                             ║
║  ┌────────────────────────────────▼────────────────────────────────────────────┐    ║
║  │              ORCHESTRATION CORE (Mission Scheduler + Fleet Manager)          │    ║
║  │                                                                             │    ║
║  │  Task Planner → Mission Graph → AMR Assignment → Execution Monitor          │    ║
║  │  Conflict Resolution | Priority Queue (Redis) | Deadlock Detection          │    ║
║  └───────┬────────────────────┬───────────────────────────┬─────────────────────┘   ║
╚══════════╪════════════════════╪═══════════════════════════╪═════════════════════════╝
           │                    │                           │
╔══════════▼════════╗  ╔═══════▼══════════════╗  ╔═════════▼═══════════════════════╗
║   KNOWLEDGE PLANE ║  ║   WORLD MODEL PLANE  ║  ║     ROBOT CONTROL LAYER         ║
║                   ║  ║                      ║  ║                                 ║
║  ChromaDB         ║  ║  VQ-VAE Tokenizer    ║  ║  ROS 2 Humble (rclpy)           ║
║  (RAG embeddings) ║  ║  Transformer (GPT-J) ║  ║  Nav2 — global/local planners   ║
║                   ║  ║  Occupancy Pred.     ║  ║  MoveIt2 — arm manipulation     ║
║  Neo4j            ║  ║  Trajectory Pred.    ║  ║  ros2_control — joint drivers   ║
║  (scene graph)    ║  ║  Temporal Fusion     ║  ║  Safety Monitor (E-stop arbiter) ║
║                   ║  ║                      ║  ║                                 ║
║  PostgreSQL       ║  ║  Digital Twin Sync   ║  ║  DDS — Cyclone DDS transport    ║
║  (WMS / history)  ║  ║  (real ↔ sim state)  ║  ║                                 ║
╚═══════════════════╝  ╚══════════╤═══════════╝  ║  AMR Fleet (n vehicles)         ║
                                  │               ║  Fixed Manipulators (m arms)    ║
╔═════════════════════════════════▼══════════════╗║  Camera Network                 ║
║          PERCEPTION & INFERENCE PLANE          ║╚═════════════════════════════════╝
║                                                ║           ▲ sensor data
║  NVIDIA Triton Inference Server 2.x           ║           │
║  ┌────────────────┬────────────────────────┐  ║  ╔════════╧════════════════════╗
║  │ Detection      │ Pose Estimation (6DoF) │  ║  ║  SIMULATION / DATA PLANE   ║
║  │ (YOLO-World)   │ FoundationPose         │  ║  ║                            ║
║  ├────────────────┼────────────────────────┤  ║  ║  NVIDIA Isaac Sim 4.x      ║
║  │ Seg / Depth    │ Grasp Prediction       │  ║  ║  Omniverse Replicator      ║
║  │ (Mask2Former)  │ (Contact-GraspNet)     │  ║  ║  Domain Randomizer         ║
║  └────────────────┴────────────────────────┘  ║  ║  Synthetic Dataset Gen     ║
║                                                ║  ║  Fire / Near-Miss Sim      ║
║  TensorRT 10.x  |  CUDA Streams  |  FP16/INT8 ║  ╚════════════════════════════╝
╚════════════════════════════════════════════════╝
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                            OBSERVABILITY PLANE                                       ║
║                                                                                      ║
║  Prometheus 2.x  │  Grafana 11  │  OpenTelemetry Collector  │  Structlog / JSON    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       REQUEST LIFECYCLE — OPERATOR INSTRUCTION                   │
│                                                                                  │
│  1. Operator sends NL message: "Move all pallets from A3 to dock 7 by shift end" │
│                    │                                                             │
│                    ▼                                                             │
│  2. FastAPI Gateway ──JWT verify──► Rate limiter ──► WebSocket / REST response  │
│                    │                                                             │
│                    ▼                                                             │
│  3. WarehouseGPT Agent (Claude Sonnet 4.6)                                       │
│     ├── Embed instruction in conversation context                                │
│     ├── TOOL: query_scene_graph(zone="A3") → Neo4j returns pallet list           │
│     ├── TOOL: get_twin_state() → Digital Twin returns AMR positions/loads        │
│     ├── TOOL: query_rag("dock 7 clearance policy") → ChromaDB returns SOP        │
│     └── TOOL: create_mission(pallets=[...], target="dock7", deadline="shift_end")│
│                    │                                                             │
│                    ▼                                                             │
│  4. Mission Scheduler                                                            │
│     ├── Decomposes into sub-tasks: pick(pallet_i, A3) → transport → place(dock7) │
│     ├── Pushes sub-tasks to Redis priority queue                                 │
│     └── Returns mission_id to agent                                              │
│                    │                                                             │
│                    ▼                                                             │
│  5. Fleet Manager                                                                │
│     ├── Pulls tasks from Redis queue                                             │
│     ├── Assigns available AMRs using Hungarian algorithm                         │
│     ├── Sends ROS 2 action goals via ros2_bridge                                 │
│     └── Subscribes to /amr/N/status topics for progress updates                 │
│                    │                                                             │
│                    ▼                                                             │
│  6. AMR (Nav2 + MoveIt2)                                                         │
│     ├── Nav2 plans path A3→dock7, avoids dynamic obstacles via Triton DNN       │
│     ├── Triton serves detection + depth at 30 Hz (TensorRT INT8, <5 ms latency) │
│     ├── MoveIt2 executes grasp using predicted grasp pose                        │
│     └── Publishes completion to /amr/N/status                                   │
│                    │                                                             │
│                    ▼                                                             │
│  7. Digital Twin Sync                                                            │
│     ├── Receives real sensor updates over MQTT/WebSocket                         │
│     ├── Updates Isaac Sim scene state                                            │
│     └── Pushes delta to Neo4j scene graph + PostgreSQL WMS records              │
│                    │                                                             │
│                    ▼                                                             │
│  8. Agent polls mission status and streams update to operator                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Diagram

### 3.1 Synthetic Data Pipeline: Isaac Sim → World Model → Digital Twin

```
Isaac Sim 4.x (headless)
│
│  Warehouse scene: aisles, shelves, pallets, forklifts, workers, lighting variants
│  Domain randomization: textures, friction, lighting, pallet weight, sensor noise
│
├─── RGB cameras (8 × 1920×1080 @ 30 fps)
├─── Depth sensors (LiDAR point clouds, 128-beam)
├─── Semantic segmentation masks (per-pixel class labels)
├─── Bounding boxes (COCO format)
├─── 6-DoF object poses (ground-truth from simulation state)
├─── Worker keypoints (skeleton estimation ground truth)
└─── Event stream (collision events, fire triggers, zone violations)
         │
         ▼ Omniverse Replicator → HDF5 / TFRecord shards
         │   ~50 TB synthetic dataset
         │   Stored: NFS / S3-compatible object store
         │
         ▼ Data Pipeline (synthetic_data/)
         │   preprocessing.py   — normalize, resize, augment
         │   augmentation.py    — CutMix, MixUp, color jitter, fog/rain sim
         │   dataset_loader.py  — PyTorch Dataset with streaming reads
         │   stats.py           — dataset statistics, class balance report
         │
         ├─── Branch A: Perception Models (Triton)
         │    YOLOv10 detection → ONNX → TensorRT INT8 plan
         │    Mask2Former segmentation → TensorRT FP16 plan
         │    FoundationPose → TensorRT FP16 plan
         │    Contact-GraspNet → TensorRT FP16 plan
         │
         ├─── Branch B: World Model Training (world_model/)
         │    │
         │    ▼ VQ-VAE Tokenizer (vqvae.py)
         │    │   Input: 256×256 RGB frame + depth channel
         │    │   Encoder: ResNet-50 backbone → 16×16 feature grid
         │    │   Vector Quantization: codebook size 8192, dim 256
         │    │   Decoder: transposed conv → reconstruction loss + perceptual loss
         │    │   Output: 256 discrete tokens per frame
         │    │
         │    ▼ Latent Tokenizer (latent_tokenizer.py)
         │    │   Combines frame tokens + action tokens + sensor tokens
         │    │   Sequence: [frame_t, action_t, sensor_t, frame_{t+1}, ...]
         │    │
         │    ▼ GPT-J style Transformer (transformer/model.py)
         │    │   24 layers, 16 heads, 1024 embedding dim
         │    │   Context window: 512 tokens (≈ 2 seconds at 30 fps)
         │    │   Training: next-token prediction on latent sequences
         │    │   Hardware: 4× A100 80 GB, DeepSpeed ZeRO-2
         │    │   Duration: ~7 days per training run
         │    │
         │    ▼ Prediction Heads (world_model/prediction/)
         │        occupancy.py     — 3D occupancy grid prediction (50 ms horizon)
         │        trajectory.py   — AMR/worker trajectory prediction (5 s horizon)
         │        temporal.py     — temporal consistency via LSTM fusion
         │
         └─── Branch C: Safety AI Training (safety_ai/)
              │
              ▼ Labeling Strategy (labeling/strategy.py)
              │   Auto-label from simulation events
              │   Human-in-the-loop review for edge cases
              │   CVAT integration for annotation management
              │
              ▼ Safety Detectors (detectors/)
                  fire.py          — YOLOv10 fine-tuned on fire/smoke frames
                  near_miss.py     — spatio-temporal distance model
                  collision.py     — contact event classifier
                  worker_safety.py — PPE detection + zone proximity
                  zone_violation.py — semantic zone rule enforcement

         ┌─── World Model outputs feed into:
         │
         ▼ Digital Twin (apps/digital_twin_sync/)
             Isaac Sim scene is updated continuously from:
             (a) Real sensor data — MQTT / WebSocket ingestion
             (b) World model predictions — fill gaps between sensor updates
             (c) RL policy rollouts — simulate forklift actions before execution
             State sync: Neo4j (spatial graph) + PostgreSQL (WMS state)
             Latency target: < 100 ms end-to-end (sensor → twin update)
```

### 3.2 Real-Time Inference Pipeline

```
Camera frame (1920×1080 JPEG)
         │ MQTT → Redis stream pub/sub
         ▼
Triton HTTP/gRPC :8000/:8001
         ├── detection model  → [N × {class, bbox, score}]   @ 30 Hz, < 5 ms
         ├── segmentation     → [H×W mask]                   @ 10 Hz, < 15 ms
         ├── depth estimation → [H×W float32]                @ 30 Hz, < 8 ms
         └── pose estimation  → [M × {R|t 6DoF}]            @ 10 Hz, < 12 ms
         │
         ▼ apps/fleet_manager + safety_ai detectors consume inference results
         ├── Safety alerts → Redis pub/sub "safety:alerts" channel → WebSocket push
         └── Scene updates → Neo4j Cypher MERGE queries → Digital Twin delta
```

---

## 4. Technology Choices Rationale

### 4.1 Simulation: NVIDIA Isaac Sim 4.x

**Why:** Isaac Sim is the only photorealistic warehouse simulator with physics-accurate forklifts, articulated shelving, conveyor belts, and a built-in Replicator synthetic data pipeline. The USD scene format gives us asset portability. Alternative (Gazebo) lacks photorealism needed for sim-to-real transfer.

### 4.2 VQ-VAE + GPT Transformer World Model

**Why:** Discrete tokenization of continuous visual observations allows applying language-model-scale training infrastructure (DeepSpeed, FlashAttention) to physical world modeling. VQ-VAE codebook forces the model to learn compact, reusable scene representations. Competitor approach (latent diffusion) is slower at inference time and harder to condition on discrete actions.

### 4.3 NVIDIA Triton Inference Server

**Why:** Triton provides concurrent model execution, dynamic batching, and TensorRT backend with INT8 quantization out of the box. Benchmarked at 2× throughput vs. direct TensorRT serving due to CUDA multi-stream concurrency. Production deployable on both bare-metal GPUs and Kubernetes with NVIDIA MIG.

### 4.4 ROS 2 Humble with Nav2 + MoveIt2

**Why:** ROS 2 is the de facto standard for robot middleware. Nav2 ships production-grade Dijkstra/A*/DWA planners with lifecycle management. MoveIt2 supports 6-DoF arm planning with collision-aware trajectory optimization. DDS (Cyclone) provides deterministic, low-latency pub/sub without a central broker. Alternative (custom middleware) would require rebuilding years of ecosystem work.

### 4.5 Claude Sonnet 4.6 as Agent Brain

**Why:** Claude's extended context window (200K tokens), tool-use capability, and strong instruction-following on structured JSON outputs make it suitable for decomposing complex warehouse operator instructions into typed tool calls. The LangGraph framework adds stateful multi-turn conversation and robust error handling for tool failures.

### 4.6 Neo4j Scene Graph

**Why:** Warehouse semantics are inherently relational: Zone CONTAINS Shelf, Shelf STORES Pallet, Pallet HAS_SKU Item, Worker IS_NEAR Forklift. Property graph queries (Cypher) express these spatial and semantic relationships naturally. PostgreSQL's relational schema would require expensive JOINs; a vector store alone cannot answer structural queries like "find all forklifts within 5 meters of worker W in zone A3."

### 4.7 Redis for Task Queue and Cache

**Why:** Redis Streams provide ordered, replay-capable event logs for mission state. Celery uses Redis as broker for background task processing. Redis pub/sub delivers sub-millisecond latency for real-time safety alerts. The LRU cache stores inference results and digital-twin state snapshots.

---

## 5. Scalability Considerations

### 5.1 Horizontal Scaling

| Component | Scaling Strategy | Constraint |
|---|---|---|
| FastAPI Gateway | Horizontal pod autoscaling (HPA) on CPU/RPS | Stateless; session via JWT |
| WarehouseGPT Agent | 1 replica per warehouse site or HPA | Claude API rate limits (per-org) |
| Triton Server | NVIDIA MIG + multi-GPU node pools | GPU VRAM per model |
| Fleet Manager | Partition by warehouse floor zone | Neo4j write throughput |
| Celery Workers | HPA on Redis queue depth | Redis memory |
| PostgreSQL | Read replicas + PgBouncer connection pooling | Write throughput via WAL |
| Neo4j | Causal clustering (3-node) for HA | Write scaling limited in Community |
| ChromaDB | Horizontal sharding (planned in Chroma 0.6) | Current: single-node |

### 5.2 Multi-Warehouse Architecture

Each warehouse site runs its own edge stack (Triton + ROS 2 bridge + Redis), which synchronizes with the central cloud control plane over mTLS-secured WebSocket tunnels. This keeps perception inference latency local (<10 ms) while allowing centralized fleet analytics and model updates.

```
                    Cloud Control Plane
                    ┌──────────────────────────────────┐
                    │  WarehouseGPT Agent (multi-tenant) │
                    │  Central PostgreSQL (analytics)    │
                    │  Central Neo4j (fleet-wide graph)  │
                    │  Model Registry (MLflow)           │
                    └────────┬──────────────────────────┘
                             │ mTLS WebSocket
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
      Site A Edge       Site B Edge       Site C Edge
      (Triton+ROS2)    (Triton+ROS2)    (Triton+ROS2)
      Local Redis      Local Redis      Local Redis
      Local Neo4j      Local Neo4j      Local Neo4j
```

### 5.3 Model Serving Throughput Targets

| Model | Resolution | Latency (P99) | Throughput |
|---|---|---|---|
| YOLOv10 detection | 640×640 | < 5 ms | 200 FPS/GPU |
| Mask2Former segmentation | 512×512 | < 15 ms | 67 FPS/GPU |
| FoundationPose | 256×256 RoI | < 12 ms | 83 FPS/GPU |
| Contact-GraspNet | 1024 pts | < 20 ms | 50 FPS/GPU |
| Safety classifier | 224×224 | < 3 ms | 333 FPS/GPU |

All latency targets measured on NVIDIA A10G (24 GB VRAM), TensorRT INT8, batch size 1.

### 5.4 Data Volume Estimates

| Data Type | Volume/Day | Retention | Storage |
|---|---|---|---|
| Camera streams (raw) | 8 TB | 7 days | Local NVMe |
| Camera streams (compressed JPEG) | 400 GB | 30 days | NFS / S3 |
| Inference results (JSON) | 5 GB | 90 days | PostgreSQL |
| Scene graph updates (Neo4j) | 200 MB | Indefinite | Neo4j |
| Prometheus metrics | 2 GB | 30 days | Prometheus TSDB |
| Mission/audit logs | 500 MB | 7 years (OSHA) | PostgreSQL → S3 |
