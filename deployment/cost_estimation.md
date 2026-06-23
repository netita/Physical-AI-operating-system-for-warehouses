# WarehouseGPT — Production Cost Estimation

**Prepared by:** WarehouseGPT Engineering  
**Date:** June 2026  
**Currency:** USD/month (1 USD = 1 USD; AWS us-east-1 on-demand list pricing unless noted)

---

## Deployment Tiers

| Attribute          | Small                  | Mid-Scale              | Enterprise              |
|--------------------|------------------------|------------------------|-------------------------|
| Warehouses         | 1                      | 5                      | 50                      |
| IP Cameras         | 10                     | 50                     | 500                     |
| Forklifts          | 5                      | 30                     | 300                     |
| Orders/day         | ~500                   | ~5,000                 | ~100,000                |
| Users (operators)  | 5–10                   | 50–100                 | 500–2,000               |

---

## Tier 1 — Small (1 Warehouse, 10 Cameras, 5 Forklifts)

### GPU Compute — Inference & World Model

| Service                          | Instance Type     | Count | $/hr  | Hours/mo | Monthly Cost |
|----------------------------------|-------------------|-------|-------|----------|--------------|
| Triton Inference Server          | p3.2xlarge (1×V100)| 1    | 3.06  | 720      | $2,203       |
| World Model Inference            | (shared node)     | —     | —     | —        | $0           |
| Safety Detector (co-located)     | (shared node)     | —     | —     | —        | $0           |
| **GPU Subtotal**                 |                   |       |       |          | **$2,203**   |

> Small tier runs all inference on a single p3.2xlarge. For production consider 1-year Reserved Instance at 40% discount: ~$1,320/mo.

### CPU Compute — Application Services

| Service                          | Instance Type     | Count | $/hr  | Hours/mo | Monthly Cost |
|----------------------------------|-------------------|-------|-------|----------|--------------|
| Agent API (warehousegpt-agent)   | c6i.xlarge (4 vCPU, 8 GB) | 2 | 0.17 | 720 | $245 |
| Digital Twin Service             | c6i.large (2 vCPU, 4 GB)  | 1 | 0.085 | 720 | $61 |
| Celery Workers                   | c6i.large              | 1 | 0.085 | 720 | $61 |
| **CPU Subtotal**                 |                        |   |       |          | **$367**     |

### Databases

| Service              | AWS Service            | Config                          | Monthly Cost |
|----------------------|------------------------|---------------------------------|--------------|
| TimescaleDB          | RDS PostgreSQL db.r6g.xlarge | 4 vCPU, 32 GB RAM, 500 GB gp3 | $520 |
| Redis                | ElastiCache r6g.large  | 2 vCPU, 13 GB RAM, Multi-AZ    | $218         |
| Neo4j                | EC2 r6i.large + 100 GB SSD | Self-managed single node   | $195         |
| ChromaDB             | EC2 c6i.large + 50 GB SSD  | Self-managed                | $100         |
| **Database Subtotal**|                        |                                 | **$1,033**   |

### Storage

| Type                             | Size    | $/GB/mo | Monthly Cost |
|----------------------------------|---------|---------|--------------|
| S3 — Model repository            | 100 GB  | 0.023   | $2           |
| S3 — Video clips (incidents)     | 500 GB  | 0.023   | $12          |
| S3 — Synthetic data archive      | 1 TB    | 0.023   | $23          |
| EBS gp3 — TimescaleDB            | 500 GB  | 0.08    | $40          |
| EBS gp3 — Neo4j                  | 100 GB  | 0.08    | $8           |
| CloudWatch Logs / log archive    | 50 GB   | 0.03    | $2           |
| **Storage Subtotal**             |         |         | **$87**      |

### Networking

| Component                        | Config                       | Monthly Cost |
|----------------------------------|------------------------------|--------------|
| EKS Control Plane                | 1 cluster                    | $73          |
| NAT Gateway                      | 1 AZ, ~500 GB egress         | $45          |
| Load Balancer (ALB)              | 1 ALB, 50 GB processed       | $22          |
| Camera stream ingress (RTSP)     | 10 cameras × 2 Mbps × 720h  | $20          |
| Data transfer out                | ~200 GB                      | $18          |
| **Networking Subtotal**          |                              | **$178**     |

### Third-Party APIs & Licences

| Service                          | Plan / Usage             | Monthly Cost |
|----------------------------------|--------------------------|--------------|
| Anthropic Claude (claude-sonnet-4-6) | ~500K tokens/day input + 100K output | $135 |
| Weights & Biases (MLOps)         | Team plan                | $150         |
| PagerDuty (on-call)              | Professional, 5 users    | $90          |
| **API/Licences Subtotal**        |                          | **$375**     |

### Tier 1 Monthly Total

| Category              | Cost     |
|-----------------------|----------|
| GPU Compute           | $2,203   |
| CPU Compute           | $367     |
| Databases             | $1,033   |
| Storage               | $87      |
| Networking            | $178     |
| APIs & Licences       | $375     |
| **TOTAL**             | **$4,243/mo** |
| Per-warehouse         | **$4,243/mo** |
| With 1yr Reserved (GPU -40%) | **~$3,360/mo** |

---

## Tier 2 — Mid-Scale (5 Warehouses, 50 Cameras, 30 Forklifts)

### GPU Compute

| Service                          | Instance Type         | Count | $/hr  | Hours/mo | Monthly Cost |
|----------------------------------|-----------------------|-------|-------|----------|--------------|
| Triton Inference Server (HA)     | p3.8xlarge (4×V100)   | 2     | 12.24 | 720      | $17,626      |
| World Model (dedicated)          | p3.2xlarge            | 2     | 3.06  | 720      | $4,406       |
| Safety Detector (high-throughput)| p3.2xlarge            | 1     | 3.06  | 720      | $2,203       |
| **GPU Subtotal**                 |                       |       |       |          | **$24,235**  |

> 1-year Reserved: ~$14,540/mo (-40%)

### CPU Compute

| Service                          | Instance Type         | Count | $/hr  | Hours/mo | Monthly Cost |
|----------------------------------|-----------------------|-------|-------|----------|--------------|
| Agent API (auto-scaled, avg)     | c6i.2xlarge           | 4     | 0.34  | 720      | $979         |
| Digital Twin (per warehouse)     | c6i.xlarge            | 5     | 0.17  | 720      | $612         |
| Celery Workers                   | c6i.xlarge            | 3     | 0.17  | 720      | $367         |
| ROS2 Bridge (edge simulation)    | c6i.large             | 5     | 0.085 | 720      | $306         |
| **CPU Subtotal**                 |                       |       |       |          | **$2,264**   |

### Databases

| Service              | AWS Service                  | Config                                | Monthly Cost |
|----------------------|------------------------------|---------------------------------------|--------------|
| TimescaleDB          | RDS PostgreSQL db.r6g.4xlarge | 16 vCPU, 128 GB RAM, Multi-AZ, 5 TB | $3,890       |
| Redis                | ElastiCache r6g.2xlarge      | Cluster mode, 3 shards × 2 replicas   | $1,310       |
| Neo4j Enterprise     | EC2 r6i.2xlarge × 3          | Causal cluster, 500 GB SSD × 3        | $2,100       |
| ChromaDB             | EC2 c6i.2xlarge × 2          | HA pair, 500 GB SSD × 2               | $680         |
| **Database Subtotal**|                              |                                       | **$7,980**   |

### Storage

| Type                             | Size    | $/GB/mo | Monthly Cost |
|----------------------------------|---------|---------|--------------|
| S3 — Model repository            | 500 GB  | 0.023   | $12          |
| S3 — Video clips (incidents)     | 5 TB    | 0.023   | $115         |
| S3 — Time-series archive (Parquet)| 10 TB  | 0.023   | $230         |
| S3 Glacier — Long-term archive   | 50 TB   | 0.004   | $200         |
| EBS gp3 — Databases              | 10 TB   | 0.08    | $800         |
| ECR — Container images           | 100 GB  | 0.10    | $10          |
| **Storage Subtotal**             |         |         | **$1,367**   |

### Networking

| Component                        | Config                             | Monthly Cost |
|----------------------------------|------------------------------------|--------------|
| EKS Control Plane                | 2 clusters (prod + dev/staging)    | $146         |
| NAT Gateways                     | 3 AZs, ~5 TB egress                | $400         |
| ALB (HTTPS + WebSocket)          | 2 ALBs, 500 GB processed           | $80          |
| Camera stream ingress            | 50 cameras × 2 Mbps               | $100         |
| Inter-warehouse VPN (WireGuard)  | 5 sites, 1 TB                      | $50          |
| CloudFront (dashboard CDN)       | 50 GB                              | $5           |
| Data transfer out                | ~2 TB                              | $180         |
| **Networking Subtotal**          |                                    | **$961**     |

### Third-Party APIs & Licences

| Service                                   | Plan / Usage                       | Monthly Cost |
|-------------------------------------------|------------------------------------|--------------|
| Anthropic Claude (claude-sonnet-4-6)      | ~3M tokens/day input, 600K output  | $810         |
| Weights & Biases (MLOps)                  | Business plan                      | $500         |
| PagerDuty (on-call)                       | Business, 20 users                 | $360         |
| Datadog (APM fallback / secondary)        | Pro, 10 hosts                      | $450         |
| Neo4j Enterprise licence (if not OEM)    | Annual ÷ 12                        | $1,500       |
| **API/Licences Subtotal**                 |                                    | **$3,620**   |

### Tier 2 Monthly Total

| Category              | Cost      |
|-----------------------|-----------|
| GPU Compute           | $24,235   |
| CPU Compute           | $2,264    |
| Databases             | $7,980    |
| Storage               | $1,367    |
| Networking            | $961      |
| APIs & Licences       | $3,620    |
| **TOTAL**             | **$40,427/mo** |
| Per-warehouse         | **$8,085/mo** |
| With 1yr Reserved (GPU -40%) | **~$30,140/mo** |

---

## Tier 3 — Enterprise (50 Warehouses, 500 Cameras, 300 Forklifts)

### GPU Compute

| Service                              | Instance Type                 | Count | $/hr  | Hours/mo | Monthly Cost  |
|--------------------------------------|-------------------------------|-------|-------|----------|---------------|
| Triton Inference Server (HA)         | p4d.24xlarge (8×A100 80 GB)   | 4     | 32.77 | 720      | $94,377       |
| World Model Inference                | p4d.24xlarge                  | 2     | 32.77 | 720      | $47,189       |
| Safety Detector (per-region cluster) | p3.8xlarge (4×V100) × 3 regions| 6    | 12.24 | 720      | $52,877       |
| RL Training cluster (weekly jobs)    | p4d.24xlarge × 4 (Spot, 40h/mo)| —    | 9.83  | 160      | $6,291        |
| **GPU Subtotal**                     |                               |       |       |          | **$200,734**  |

> 3-year Reserved (Convertible) GPU savings: up to 60% = **~$80,294/mo**  
> Spot instances for batch training only; no Spot for real-time inference.

### CPU Compute (EKS managed node groups)

| Service                          | Instance Type         | Count | $/hr   | Hours/mo | Monthly Cost |
|----------------------------------|-----------------------|-------|--------|----------|--------------|
| Agent API                        | c6i.4xlarge           | 20    | 0.68   | 720      | $9,792       |
| Digital Twin (per warehouse)     | c6i.2xlarge           | 50    | 0.34   | 720      | $12,240      |
| Celery Workers                   | c6i.4xlarge           | 10    | 0.68   | 720      | $4,896       |
| ROS2 Bridge + Edge Proxies       | c6i.xlarge            | 50    | 0.17   | 720      | $6,120       |
| Ingress + API Gateway nodes      | c6i.2xlarge           | 6     | 0.34   | 720      | $1,469       |
| Monitoring stack (Prometheus/Grafana/Loki/Tempo) | r6i.2xlarge | 6 | 0.504 | 720 | $2,177 |
| **CPU Subtotal**                 |                       |       |        |          | **$36,694**  |

### Databases

| Service               | AWS Service                      | Config                                            | Monthly Cost |
|-----------------------|----------------------------------|---------------------------------------------------|--------------|
| TimescaleDB           | RDS PostgreSQL db.r6g.16xlarge   | 64 vCPU, 512 GB RAM, Multi-AZ, 100 TB, IOPS 10K  | $28,600      |
| Redis                 | ElastiCache r6g.4xlarge          | 8 shards × 3 replicas (Global Datastore)          | $9,720       |
| Neo4j Enterprise      | EC2 r6i.8xlarge × 9             | 3 clusters (3 nodes each), 5 TB SSD each          | $31,500      |
| ChromaDB              | EC2 r6i.4xlarge × 6             | HA across 3 AZs, 5 TB SSD each                   | $12,960      |
| Aurora (metadata/control plane) | db.r6g.4xlarge Multi-AZ | Ops metadata, event log                   | $2,200       |
| **Database Subtotal** |                                  |                                                   | **$84,980**  |

### Storage

| Type                                    | Size     | $/GB/mo or rate | Monthly Cost |
|-----------------------------------------|----------|-----------------|--------------|
| S3 — Model repository + versioning      | 5 TB     | 0.023           | $115         |
| S3 — Video incident archive             | 200 TB   | 0.023           | $4,600       |
| S3 — Time-series Parquet (analytics)    | 500 TB   | 0.023           | $11,500      |
| S3 Glacier — 7-year regulatory archive  | 2 PB     | 0.004           | $8,000       |
| S3 Intelligent-Tiering (active data)    | 50 TB    | 0.023–0.0125    | $690         |
| EBS gp3 — Database volumes              | 300 TB   | 0.08            | $24,000      |
| EFS — Shared model cache                | 10 TB    | 0.30            | $3,000       |
| ECR — Container registry                | 500 GB   | 0.10            | $50          |
| Backup (AWS Backup)                     | 50 TB    | 0.05            | $2,500       |
| **Storage Subtotal**                    |          |                 | **$54,455**  |

### Networking

| Component                               | Config                                  | Monthly Cost |
|-----------------------------------------|-----------------------------------------|--------------|
| EKS Control Planes                      | 5 clusters (prod × 3 regions + dev/qa)  | $365         |
| NAT Gateways                            | 3 regions × 3 AZs, ~50 TB egress        | $4,250       |
| ALB / NLB                               | 10 load balancers, 5 TB processed        | $480         |
| AWS Direct Connect (warehouses to cloud)| 10 Gbps dedicated, 50 warehouses        | $15,000      |
| Camera stream ingress                   | 500 cameras × 2 Mbps                   | $1,000       |
| CloudFront (dashboard + static assets)  | 10 TB                                   | $600         |
| Route 53 (DNS, health checks)           | 50 hosted zones, 1M health checks       | $250         |
| VPC Transit Gateway (multi-region)      | 10 attachments, 50 TB                   | $2,500       |
| PrivateLink (internal API calls)        | 20 endpoints                            | $200         |
| Data transfer out (total)               | ~20 TB                                  | $1,800       |
| WAF + Shield Advanced (DDoS)            | 10 resources                            | $3,180       |
| **Networking Subtotal**                 |                                         | **$29,625**  |

### Third-Party APIs & Licences

| Service                                        | Plan / Usage                          | Monthly Cost  |
|------------------------------------------------|---------------------------------------|---------------|
| Anthropic Claude (claude-sonnet-4-6)           | ~30M tokens/day input, 6M output      | $8,100        |
| Anthropic API Enterprise (volume discount -20%)| Applied above                         | -$1,620       |
| NVIDIA AI Enterprise (Triton, CUDA X)          | Per-GPU licence, 24 GPUs             | $14,400       |
| Weights & Biases (Enterprise MLOps)            | Enterprise annual ÷ 12               | $3,500        |
| PagerDuty (Enterprise on-call)                 | Enterprise, 200 users                | $5,000        |
| Datadog (APM, Logs, Infrastructure)            | Enterprise, 200 hosts                | $12,000       |
| Neo4j Enterprise licence                       | Annual ÷ 12, 9 servers               | $13,500       |
| Grafana Enterprise                             | Annual ÷ 12                          | $2,000        |
| Security scanning (Snyk, Wiz)                  | Enterprise                           | $3,000        |
| SOC 2 audit tooling (Vanta)                    | Annual ÷ 12                          | $1,500        |
| **API/Licences Subtotal**                      |                                       | **$61,380**   |

### Tier 3 Monthly Total

| Category              | Cost        |
|-----------------------|-------------|
| GPU Compute           | $200,734    |
| CPU Compute           | $36,694     |
| Databases             | $84,980     |
| Storage               | $54,455     |
| Networking            | $29,625     |
| APIs & Licences       | $61,380     |
| **TOTAL**             | **$467,868/mo** |
| Per-warehouse         | **$9,357/mo** |
| Annual cost           | **~$5.6M/yr** |
| With 3yr Reserved + Savings Plans (GPU -60%) | **~$267,868/mo (~$3.2M/yr)** |

---

## Cost Comparison Summary

| Metric                   | Small         | Mid-Scale     | Enterprise     |
|--------------------------|---------------|---------------|----------------|
| Monthly total            | $4,243        | $40,427       | $467,868       |
| Per-warehouse/month      | $4,243        | $8,085        | $9,357         |
| Per-camera/month         | $424          | $808          | $936           |
| Per-forklift/month       | $849          | $1,348        | $1,559         |
| GPU % of total           | 52%           | 60%           | 43%            |
| Database % of total      | 24%           | 20%           | 18%            |
| Annual (no discounts)    | $50,916       | $485,124      | $5,614,416     |
| Annual (with Reserved)   | ~$40,320      | ~$361,680     | ~$3,214,416    |

---

## Cost Optimisation Strategies

### 1. Reserved Instances & Savings Plans

| Strategy                      | Saving  | Commitment |
|-------------------------------|---------|------------|
| 1-year Reserved (GPU)         | 40%     | 1 year     |
| 3-year Reserved (GPU)         | 60%     | 3 years    |
| Compute Savings Plans (CPU)   | 30–40%  | 1–3 years  |
| RDS Reserved                  | 30–55%  | 1–3 years  |
| ElastiCache Reserved          | 30–55%  | 1–3 years  |

### 2. Spot Instances for Batch Workloads
- RL training jobs: 70–90% saving
- Synthetic data generation: 70–90% saving
- Model fine-tuning: 70% saving
- **Do NOT use Spot for real-time inference** (latency/availability SLA)

### 3. Model Optimization (reduces GPU requirement)
| Technique           | GPU Cost Reduction | Quality Impact |
|---------------------|-------------------|----------------|
| TensorRT FP16       | 30–40%            | <1% accuracy loss |
| TensorRT INT8       | 50–60%            | 1–3% accuracy loss |
| Model distillation  | 40–60%            | 3–5% accuracy loss |
| Dynamic batching    | 20–30%            | None |

### 4. Storage Cost Reduction
- Move video clips > 30 days to S3 Glacier Instant Retrieval: saves 68%
- TimescaleDB continuous aggregation + compression: saves 60% on old data
- S3 Intelligent-Tiering for models > 30 days: saves 40%

### 5. Multi-Cloud / Hybrid Deployment
- On-premise GPU servers (NVIDIA DGX A100): 5-year TCO 60% lower than cloud for Enterprise tier
- AWS Outposts: run EKS workloads on-prem at consistent pricing
- Edge AI (NVIDIA Jetson AGX Orin at each forklift): reduces cloud inference cost by 40%

---

## Break-Even & ROI Analysis

### Value Delivered

| Metric                        | Industry Benchmark | WarehouseGPT Target |
|-------------------------------|-------------------|---------------------|
| Forklift collision reduction  | —                 | 85%                 |
| Near-miss reduction           | —                 | 70%                 |
| Fleet utilisation improvement | Baseline          | +25%                |
| Order fulfilment speed        | Baseline          | +40%                |
| Safety incident cost (avg)    | $50,000–$500,000  | Avoided             |
| Unplanned downtime reduction  | —                 | 60%                 |

### Small Warehouse ROI
- **Platform cost:** $4,243/month
- **Value from 1 avoided incident/year:** $50,000+ = $4,167/month
- **ROI break-even:** < 1 month of prevented incidents
- **Fleet efficiency gain (25% utilisation +):** $8,000–$15,000/month in labour savings

### Enterprise ROI
- **Platform cost:** $467,868/month
- **50 warehouses × 2 incidents avoided/mo × $100K avg:** $10,000,000/month avoided costs
- **ROI:** 21:1 (return:cost)
- **Annual net value at Enterprise:** ~$115M incident cost avoidance + $24M efficiency gains

---

## Notes & Assumptions

1. **Pricing:** AWS us-east-1 on-demand list prices as of June 2026. GCP pricing is within 10% for equivalent specs.
2. **GPU instances:** p3 = V100 (training-gen), p4d = A100 (inference-grade). If H100 instances (p5.48xlarge) become available, expect 2× throughput at 1.6× price — better economics for Enterprise.
3. **Anthropic API pricing:** claude-sonnet-4-6 at $3/M input tokens, $15/M output tokens (standard pricing).
4. **Network:** Does not include customer's existing warehouse LAN/WAN.
5. **Licences:** Neo4j Enterprise required for Causal Cluster (HA). Triton Inference Server is free; NVIDIA AI Enterprise bundle may be optional.
6. **Compliance:** Add 5–10% for AWS compliance services (CloudTrail, Config, GuardDuty, SecurityHub) if required for ISO 45001 / SOC 2.
7. **Development/Staging:** Add 20–30% of production cost for non-production environments.
8. **Support:** AWS Business Support: ~3% of monthly bill ($13–$14K for Enterprise).

---

*For a customised quote or architecture review, open a GitHub Discussion in this repository.*
