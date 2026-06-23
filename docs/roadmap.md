# WarehouseGPT Implementation Roadmap

## Executive Timeline

```
Month   1     2     3     4     5     6     7     8     9    10    11    12
        ├─────┤─────┤─────┤─────┤─────┤─────┤─────┤─────┤─────┤─────┤─────┤
MVP     ████████████████████
Alpha         ████████████████████
Beta                ████████████████████
Pilot                         ████████████████████████
Series A                                              ████████████████████████
        └─────────────────────────────────────────────────────────────────────┘
        $50K  $150K $350K                             $1.2M               $4M
              (cumulative cost estimates)
```

---

## Phase 1: MVP — 30 Days

**Goal:** Demonstrate end-to-end pipeline from Isaac Sim data generation through WarehouseGPT agent interaction. Show investors a working demo.

**Cost Estimate:** $50,000

**Hardware:** 4× NVIDIA A100 80 GB (cloud, on-demand ~$12/hr each → ~$35k compute + $15k salaries)

**Team:** 3 ML engineers, 1 DevOps engineer, 1 PM (5 people)

### Week 1: Isaac Sim Setup and Dataset Foundation

**ML Engineer 1 (Simulation):**
- Install NVIDIA Isaac Sim 4.x on A100 node; validate GPU rendering pipeline.
- Implement `warehouse_generator.py`: parameterized warehouse scene (aisle width, shelf height, zone layout) using USD Python API.
- Integrate Omniverse Replicator: configure 8 camera placements covering all zones.
- Validate synthetic frame output quality: confirm semantic segmentation masks, bounding boxes, and depth maps are correct.

**ML Engineer 2 (Data Pipeline):**
- Stand up NFS storage mount for synthetic dataset output (target: 5 TB capacity).
- Implement `data_pipeline.py`: frame extraction, COCO JSON annotation writer, HDF5 shard writer.
- Implement `preprocessing.py`: resize to 640×640, normalize, handle missing frames.
- Set up Weights & Biases (W&B) project for experiment tracking.

**ML Engineer 3 (Infrastructure):**
- Docker Compose stack: PostgreSQL, Redis, Neo4j, ChromaDB — all services healthy.
- CI pipeline (GitHub Actions): lint (Ruff), type check (Mypy), unit tests.
- `.env.example` and secrets management (AWS Secrets Manager or local .env).

**DevOps Engineer:**
- Provision 4× A100 instances on AWS (p4d.24xlarge or similar).
- Configure NFS share between nodes.
- Set up monitoring: Prometheus + Grafana with basic GPU utilization dashboards.

**PM:**
- Finalize demo scenario: "move pallets A3 to dock 7" use case.
- Stakeholder communication plan.
- Milestone sign-off criteria document.

**Deliverable:** 100K synthetic frames generated, pipeline running, infra healthy.

---

### Week 2: VQ-VAE Training on Synthetic Data

**ML Engineer 1:**
- Implement VQ-VAE tokenizer (`world_model/tokenizer/vqvae.py`):
  - ResNet-50 encoder (pretrained ImageNet weights as init).
  - Vector quantization layer: codebook 8192 entries, dim 256.
  - Transposed-conv decoder.
  - Loss: reconstruction (L1) + VQ commitment + perceptual (VGG).
- Launch training: 4× A100, DeepSpeed ZeRO-1, batch 64, 50 epochs.
- W&B logging: reconstruction loss, codebook usage (target: >80% of codebook active).

**ML Engineer 2:**
- Implement `latent_tokenizer.py`: combine VQ-VAE frame tokens with action tokens.
- Implement `world_model/training/dataset.py`: streaming HDF5 dataset with sequence sampling.
- Implement `world_model/training/config.py`: YAML-based training configuration.

**ML Engineer 3:**
- Download and set up pre-trained YOLOv10 (PyPI: ultralytics) for baseline detection.
- Convert YOLOv10 to ONNX, then TensorRT FP16 plan.
- Stand up Triton with YOLOv10 model; validate inference via REST API.

**Deliverable:** VQ-VAE training converging (reconstruction loss < 0.05), Triton serving YOLOv10 at >100 FPS.

---

### Week 3: Safety AI Prototype (Fire + Near-Miss)

**ML Engineer 1 + 2 (joint):**
- Generate safety scenario dataset: 50K frames with fire/smoke events (`isaac_sim/fire_simulation.py`), 30K frames with near-miss events (`isaac_sim/worker_behavior.py` + `forklift_behavior.py`).
- Implement `safety_ai/labeling/strategy.py`: auto-label from simulation event log.
- Implement `safety_ai/detectors/fire.py`: YOLOv10 fine-tuned on fire/smoke class; 3-epoch fine-tune, FP16 TRT export.
- Implement `safety_ai/detectors/near_miss.py`: spatio-temporal distance model using detection tracks + kalman filter; threshold at 2 m.
- Implement `safety_ai/training/pipeline.py`: training loop, validation loop, DANN loss stub (full DANN in alpha).
- `safety_ai/training/metrics.py`: recall@95, precision@recall curves, alert latency measurement.

**ML Engineer 3:**
- FastAPI skeleton `apps/api_gateway/`: `/health`, `/safety/alerts` (stubbed), `/twin/state` (stubbed).
- Redis pub/sub integration: safety alerts published to `safety:alerts` channel.
- Celery worker with basic task queue.

**Deliverable:** Fire detector > 90% recall on synthetic test set. Near-miss alert fires within 30 ms of event.

---

### Week 4: WarehouseGPT Agent MVP with Mock Tools

**ML Engineer 2 (Agent):**
- Implement `apps/agent/` with Claude Sonnet 4.6 via Anthropic SDK.
- LangGraph tool nodes: `query_scene_graph`, `dispatch_forklift`, `create_mission`, `get_inventory`, `run_safety_check`.
- Tools use mock responses (hardcoded or from PostgreSQL seed data).
- `/agent/chat` and `/agent/chat/stream` endpoints wired up.
- Conversation history stored in PostgreSQL.

**ML Engineer 1 (Neo4j Scene Graph):**
- `scripts/seed_knowledge_graph.py`: populate Neo4j with warehouse schema (zones, shelves, pallets, forklifts).
- Implement `packages/knowledge/` Neo4j adapter with Cypher query helpers.
- Connect `query_scene_graph` tool to real Neo4j data.

**ML Engineer 3 (Digital Twin MVP):**
- `apps/digital_twin_sync/`: poll Isaac Sim via Python API, push scene state to Neo4j.
- `/twin/state` returns real data from Neo4j.
- WebSocket `/twin/stream` streams state deltas.

**DevOps:**
- Full docker-compose stack running all services.
- Demo environment with pre-loaded scenario.

**PM:**
- Coordinate investor demo (Week 4 Friday).
- Prepare demo script: operator types "Move pallets A3 to dock 7" → agent responds → mock forklifts assigned → safety alert triggers.

**Deliverable:** Live demo: operator NL instruction → agent tool calls → mission creation → safety monitoring dashboard.

---

## Phase 2: Alpha — 60 Days

**Goal:** World model trained to production quality. Real-time digital twin pipeline operational. Forklift RL basic navigation demonstrated in Isaac Sim.

**Cumulative Cost Estimate:** $150,000

**Hardware:** 8× A100 80 GB

**Team:** 3 ML engineers + 2 robotics engineers + 1 backend engineer + 1 DevOps + 1 PM = 8 people

### Key Milestones

**World Model (Days 31–45):**
- GPT-J transformer trained to completion: 24 layers, 16 heads, 1024 dim.
- Training data: 1M frame sequences from Isaac Sim.
- Validation: trajectory ADE < 0.3 m at 1 second, occupancy IoU > 0.85.
- Inference server: TensorRT export, < 50 ms prediction latency.

**Digital Twin Real-Time Pipeline (Days 31–50):**
- Real-time MQTT/WebSocket ingestion from camera network (simulated cameras in Isaac Sim for alpha).
- Isaac Sim scene state synchronized every 100 ms.
- Neo4j scene graph updated with sub-second latency.
- Digital twin viewer in Grafana (position overlays on floor plan).
- State reconciliation: real sensor data + world model predictions for occluded regions.

**Forklift RL Basic Navigation (Days 40–60):**
- Isaac Sim environment: `forklift_rl/environments/navigation.py`.
- State space: lidar scan (128 beams), goal vector, obstacle map.
- Action space: linear velocity (0–2 m/s), angular velocity (±0.5 rad/s).
- Algorithm: PPO (Proximal Policy Optimization) via `forklift_rl/training/trainer.py`.
- Reward: progress toward goal + collision penalty + smooth velocity preference.
- Curriculum: empty aisle → light traffic → full warehouse.
- Metric: 95% success rate navigating 20 m aisle with 2 obstacles.

**Kubernetes Deployment (Days 50–60):**
- Helm charts for all services: `infra/k8s/`.
- Horizontal Pod Autoscaler on Celery workers.
- NVIDIA GPU operator for Triton.
- Secrets via Kubernetes Secrets (sealed with Bitnami Sealed Secrets).
- Staging cluster: 4 nodes (2× CPU + 2× GPU).

**Domain Adaptation (Days 35–55):**
- DANN training pipeline fully implemented.
- CycleGAN trained on Isaac Sim ↔ 10K real-world warehouse frames (sourced from public datasets: MVTec, Warehouse Product Dataset).
- Perception model mAP improved by > 5 points on real-world proxy set.

---

## Phase 3: Beta — 90 Days

**Goal:** Full safety AI suite validated. Multi-agent forklift coordination. Knowledge graph populated from real SOPs. First real warehouse pilot customer identified.

**Cumulative Cost Estimate:** $350,000

**Hardware:** 8× A100 (training) + 2× A10G per warehouse site (inference)

**Team:** 3 ML engineers + 3 robotics engineers + 2 backend engineers + 1 DevOps + 1 PM + 1 customer success = 11 people

### Key Milestones

**Full Safety AI Suite (Days 61–75):**
- All detectors operational and validated on real-world proxy data:
  - `fire.py`: > 99% recall, < 0.001 false alarm rate.
  - `near_miss.py`: > 97% recall at < 2 m, > 99.5% recall at < 1 m.
  - `collision.py`: > 95% recall on contact events.
  - `worker_safety.py`: PPE detection > 92% recall per class.
  - `zone_violation.py`: > 95% recall, < 3 s detection latency.
- Safety dashboard in Grafana with real-time alert feed.
- OSHA-compliant incident log stored in PostgreSQL (7-year retention).
- Alert-to-action loop: safety alert → automatic E-stop command → confirmation to operator.

**Multi-Agent Forklift Coordination (Days 65–85):**
- `forklift_rl/environments/multi_agent.py`: N-agent MARL environment.
- Algorithm: QMIX or MAPPO for cooperative multi-agent training.
- Reward shaping: team reward (total pallet throughput/hour) + individual penalty (collision, idle time).
- Policy network: shared backbone with individual heads per agent.
- Tested with 3 forklifts coordinating in a 50×30 m warehouse.
- Metric: 0 collision incidents, > 80% of single-agent throughput efficiency at 3× agent count.
- Fleet Manager integration: RL policy queried for action recommendations, overriding is possible.

**Knowledge Graph Populated (Days 70–85):**
- SOP ingestion pipeline: PDF → text extraction → chunking → embedding (Claude claude-haiku-4-5) → ChromaDB.
- Equipment manuals (generic models from manufacturer websites) → knowledge base.
- Warehouse layout graph: zones, aisles, shelves, dock doors modeled in Neo4j.
- `scripts/seed_knowledge_graph.py`: idempotent seeding from YAML layout file.
- RAG system: `/knowledge/search` endpoint returning cited results with confidence scores.

**First Real Warehouse Pilot Identified (Days 80–90):**
- Signed LOI with 1 design-partner warehouse (target: 3PL or eCommerce fulfillment center).
- Site survey completed: camera placement plan, network diagram, floor plan digitized.
- Shadow-mode deployment plan agreed with site manager.
- Safety protocol reviewed with site's EHS team.

---

## Phase 4: 6-Month Production Launch

**Goal:** Production-grade digital twin at 3 warehouse pilots. OSHA compliance certification. Customer dashboard for multiple operators.

**Cumulative Cost Estimate:** $1,200,000

**Team:** 20 people (see staffing breakdown below)

### Key Milestones

**Production Digital Twin (Months 4–5):**
- < 100 ms end-to-end latency (sensor → twin update) validated.
- 99.9% uptime SLA with automatic failover.
- Multi-site architecture deployed: central cloud + edge per site.
- Digital twin playback: operators can replay any 24-hour period.
- Anomaly detection: automatic alert when twin diverges > threshold from sensor data.

**3 Warehouse Pilots (Months 4–6):**
- Site 1: 30-day shadow mode → 30-day teleoperated → production release.
- Sites 2 and 3: accelerated 2-week shadow → production (with lessons from Site 1).
- Real forklift integration: at least one site with real autonomous forklift dispatch (constrained to low-speed, human-supervised operation).
- Customer success metrics: near-miss reduction > 30%, mission completion time improvement > 15%.

**OSHA Compliance Validation (Month 5):**
- Third-party safety audit (SGS or Bureau Veritas).
- OSHA 1910.178 (powered industrial trucks) compliance checklist.
- Incident log format validated against OSHA recordkeeping requirements.
- Safety system failure mode analysis (FMEA) documented.

**Customer Dashboard (Months 4–5):**
- React dashboard: real-time floor plan with forklift/worker positions.
- Mission management UI: create, monitor, cancel missions.
- Safety alert feed with acknowledgment workflow.
- KPI widgets: throughput, uptime, safety score, energy consumption.
- Role-based access: viewer, operator, safety officer, admin.
- Mobile-responsive for tablet use on the warehouse floor.

**Series A Preparation (Month 6):**
- Data room: financials, pilot results, safety audit report, IP filing.
- Pitch deck updated with production metrics.
- Target: $8M–$12M Series A, led by industrial tech or robotics-focused fund.

---

## Phase 5: 12-Month Scale

**Goal:** 50 warehouses contracted. Series A closed. Enterprise sales motion. Hardware partnerships with NVIDIA and Boston Dynamics.

**Cumulative Cost Estimate:** $4,000,000

**Team:** 45 people (see staffing breakdown below)

### Key Milestones

**50 Warehouses (Months 7–12):**
- Deployment playbook: new site go-live in < 3 weeks.
- Remote onboarding: site manager configures floor plan via drag-and-drop tool.
- Multi-site dashboard: single pane of glass for fleet of warehouses.
- Pricing model: SaaS per-warehouse per-month + hardware-as-a-service for camera kit.

**Enterprise Sales Motion (Months 7–10):**
- Target verticals: 3PL, eCommerce fulfillment, automotive parts, cold chain.
- POC-to-production conversion rate target: > 60%.
- Enterprise contract template: MSA, DPA, SLA with 99.9% uptime guarantee.
- Channel partner: systems integrators (Accenture Industry X, Deloitte Supply Chain).

**NVIDIA Partnership (Month 8):**
- Co-engineering: Isaac Sim warehouse scene library shared with NVIDIA developer program.
- Joint marketing: press release, NVIDIA GTC presentation.
- Hardware bundling: NVIDIA Jetson AGX Orin as edge inference node (replaces A10G in low-volume sites).
- Early access to Isaac Sim 5.x and Cosmos world foundation model.

**Boston Dynamics Partnership (Month 9):**
- Spot integration: WarehouseGPT agent can dispatch Spot for inventory scanning and floor inspection.
- Stretch integration: full manipulation pipeline (grasp prediction → MoveIt2 → Stretch arm).
- Joint case study: "fully autonomous fulfillment" demo at Automate 2027.

**IP and Moat (Months 7–12):**
- Patent filings: warehouse-specific world model architecture, sim-to-real calibration procedure, multi-agent safety arbiter.
- Proprietary dataset: 100M+ synthetic frames + 10M+ real frames (anonymized) — largest warehouse AI dataset.
- Model zoo: pre-trained weights for 20+ forklift models, 50+ SKU categories.

---

## Staffing Breakdown

### Month 1–2 (5 people, $150k/month burn)

| Role | Count | Focus |
|---|---|---|
| Senior ML Engineer | 2 | World model, safety AI |
| ML Engineer | 1 | Data pipeline, Triton |
| DevOps Engineer | 1 | Cloud infra, CI/CD |
| Product Manager | 1 | Roadmap, investor relations |

### Month 3 (8 people, $280k/month burn)

| Role | Count | New Additions |
|---|---|---|
| Robotics Engineer | 2 | ROS 2, RL training |
| Backend Engineer | 1 | API, fleet manager |

### Month 4–6 (20 people, $600k/month burn)

| Role | Count | New Additions |
|---|---|---|
| Senior ML Engineer | 1 | Sim-to-real, domain adaptation |
| ML Engineer | 2 | Safety suite, perception |
| Robotics Engineer | 2 | Multi-agent coordination |
| Backend Engineer | 2 | Customer dashboard, API |
| Frontend Engineer | 2 | React dashboard |
| Sales Engineer | 1 | Pilot customer technical lead |
| Customer Success | 1 | Pilot onboarding |
| Security Engineer | 1 | SOC 2 prep |

### Month 7–12 (45 people, $1.2M/month burn)

Additional hires across: ML (5), Robotics (5), Software (8), Sales (5), CS (4), Ops (3), Finance/Legal (2), HR/Recruiting (2).

---

## Hardware Bill of Materials (BOM)

### Training Cluster (Cloud)

| Item | Quantity | Unit Cost/Month | Total/Month |
|---|---|---|---|
| A100 80 GB GPU instance (p4d.24xlarge) | 4 (MVP) → 8 (alpha) | $9,200 | $36,800–$73,600 |
| NFS storage (20 TB) | 1 | $500 | $500 |
| Networking (egress) | — | ~$300 | $300 |

### Per-Warehouse Edge Deployment

| Item | Quantity | Unit Cost (one-time) | Description |
|---|---|---|---|
| NVIDIA A10G GPU server (edge inference) | 1–2 | $12,000 | Triton inference server |
| IP PoE cameras (4K, 30fps) | 8–16 | $400 each | Full warehouse coverage |
| PoE network switch (24-port) | 1 | $800 | Camera network |
| Edge server (CPU-only, management) | 1 | $3,000 | ROS 2 bridge, fleet manager |
| UPS (uninterruptible power supply) | 1 | $1,500 | Uptime guarantee |
| LiDAR units (optional, Ouster OS1-64) | 2–4 | $6,000 each | High-accuracy localization zones |

**Total per site hardware:** ~$25,000–$50,000 (depending on warehouse size and camera count).

### Cloud Control Plane (Multi-Tenant, Month 6+)

| Item | Quantity | Cost/Month |
|---|---|---|
| Kubernetes cluster (EKS, 6 nodes: 4 CPU, 2 GPU) | 1 | $8,000 |
| PostgreSQL RDS (Multi-AZ, db.r6g.2xlarge) | 1 | $1,200 |
| Redis ElastiCache (r6g.xlarge, cluster mode) | 1 | $600 |
| S3 storage (100 TB dataset + backups) | 1 | $2,300 |
| CloudFront CDN (dashboard assets) | 1 | $200 |

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Sim-to-real gap: perception models underperform on real cameras | High | High | DANN + CycleGAN adaptation; 4-week calibration period | ML Lead |
| World model hallucinations cause false safety alerts | Medium | Critical | Never use world model for safety decisions; sensor-only safety pipeline | ML Lead |
| Isaac Sim license / cost exceeds budget | Low | Medium | Budget $30k/yr; evaluate Blender+PyBullet as fallback | DevOps |
| Claude API rate limits constrain agent throughput | Medium | Medium | Request enterprise rate limit increase; implement local queue | Backend Lead |
| First pilot customer delays or cancels | Medium | High | 3 parallel LOIs; design fallback demo with simulated warehouse | PM |
| Forklift RL policy fails safety tests in real deployment | Medium | Critical | Shadow mode only until recall > 99.5%; human-in-the-loop dispatch | Robotics Lead |
| GPU cloud costs exceed budget | Medium | Medium | Reserved instances (60% saving vs. on-demand); spot instances for training | DevOps |
| OSHA audit fails | Low | Critical | Third-party pre-audit at Month 4; conservative safety thresholds | PM + Safety |
| Key engineer attrition | Medium | High | Equity vesting, competitive salaries; bus factor >2 on all systems | CEO/HR |
| Competitor (Gather AI, Symbotic) launches similar product | Medium | Medium | Focus on SMB 3PL market (underserved by enterprise robotics); faster deployment | CEO/PM |
