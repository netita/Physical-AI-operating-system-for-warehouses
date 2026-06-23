# MVP Recommendation

## What to Build First and Why

### The One Thing to Demo in 30 Days

**Build:** A WarehouseGPT agent that accepts a natural language instruction ("Move pallets from zone A3 to dock 7"), queries a live-seeded Neo4j scene graph, creates a structured mission plan, and displays the mission on a Grafana floor plan — with a safety alert firing in real time when a simulated near-miss event occurs in Isaac Sim.

**Why this specific slice:**

1. **Closes the investor confidence gap.** Investors in Industrial AI have seen too many demos that are slides or videos. A live demo where a human types a warehouse instruction and watches forklifts get assigned in real time is viscerally convincing in a way that benchmarks are not.

2. **Validates the hardest technical assumption early.** The most uncertain technical question is whether Claude can reliably decompose ambiguous warehouse instructions into typed tool calls with correct parameters. Testing this on Day 30 (not Day 90) gives us time to fix it.

3. **Creates a sales artifact.** The demo can be recorded and shown to pilot prospects immediately. We do not need a production-ready system to sign a design-partner LOI — we need a credible demo and a compelling safety narrative.

4. **All subsequent phases build on this foundation.** World model, full RL deployment, multi-site dashboard — every future milestone extends what the MVP establishes, rather than rebuilding it.

### What to Explicitly Not Build in 30 Days

| Component | Reason to defer |
|---|---|
| Real forklift integration (ROS 2 hardware) | Hardware procurement takes 4–8 weeks; use simulated forklifts |
| Multi-agent RL policy | Requires 4+ weeks of training alone; sim navigation demo suffices |
| Customer-facing web dashboard | Grafana serves demo purposes; React app is Month 3 |
| Full OSHA compliance documentation | Premature; first focus on the detector working, then the paperwork |
| DANN / CycleGAN adaptation | Unnecessary until first real camera data; add in alpha |
| Kubernetes production deployment | Docker Compose is sufficient for demo; K8s in month 2 |

---

## Key Risk Mitigations

### Risk 1: Agent Tool-Call Reliability

**Risk:** Claude returns malformed JSON tool arguments or fails to decompose complex instructions correctly.

**Mitigation:**
- Write a comprehensive system prompt with explicit tool schemas (JSON Schema format).
- Implement structured output validation with Pydantic models; re-prompt on validation failure (up to 3 retries).
- Start with simple instructions in the demo; add complexity after each release.
- Use Claude claude-sonnet-4-6's extended thinking mode for multi-step decomposition tasks.
- Log all tool calls and failures from day 1; build a golden test set from real operator queries.

**Fallback:** If tool-use reliability is insufficient, wrap Claude in a deterministic rule-based router for common command patterns (move_zone_to_dock, check_inventory, run_safety_scan) and use Claude only for ambiguous/novel instructions.

### Risk 2: Sim-to-Real Gap Killing Pilot

**Risk:** Safety detectors trained on Isaac Sim data fail to detect real fire or near-miss events in the pilot warehouse.

**Mitigation:**
- Run shadow mode for minimum 2 weeks before any autonomous actuation.
- Collect 500 labeled real frames during shadow mode; fine-tune detectors before going live.
- Set safety thresholds conservatively (recall-optimized, accept higher false alarms initially).
- Never inhibit real E-stop signals based on AI confidence alone; the AI augments but does not replace physical safety systems.
- Have an emergency override protocol: operators can disable WarehouseGPT and revert to manual WMS in < 30 seconds.

### Risk 3: Pilot Customer Pull-Out

**Risk:** The single design-partner warehouse is not ready or pulls out before Beta.

**Mitigation:**
- Pursue 3 simultaneous LOIs; sign whichever converts first.
- Offer the pilot at cost (no software fee) in exchange for labeled data and a reference customer agreement.
- Prepare a simulated "virtual pilot" — Isaac Sim running a digital replica of a public warehouse layout (e.g., Amazon Robotics reference design) — as a fallback demo for investors if no real pilot is secured by Month 3.

### Risk 4: GPU Cost Overrun

**Risk:** World model training takes longer than expected, driving cloud costs above the $50k MVP budget.

**Mitigation:**
- Use AWS Spot Instances for training (60–80% cheaper than on-demand; interrupt-tolerant training via DeepSpeed checkpoint resume).
- Profile before scaling: validate VQ-VAE converges on 100K frames before committing to 1M.
- Reserve one A100 instance for 1-month Reserved Instance pricing ($5,500/mo vs. $9,200/mo on-demand).
- Hard budget cap: auto-terminate training jobs if GPU-hours exceed monthly budget threshold (CloudWatch alarm).

---

## Go-to-Market Strategy

### Target Customer: Mid-Market 3PL (Third-Party Logistics)

**Why 3PLs first:**
- 20,000+ 3PL warehouses in the US, 90% are SMB with < 500,000 sq ft.
- 3PLs face extreme labor cost pressure ($22–28/hr warehouse workers) and high turnover (60% annual attrition).
- 3PLs are technology buyers — they already use WMS software and are comfortable with SaaS subscriptions.
- They are not building their own AI capability; they need a vendor.
- Large enterprise (Amazon, Walmart) is a 3-year sales cycle and will want to acquire, not buy.

**Avoid initially:**
- Automotive / aerospace: safety certification requirements (ISO 26262, IEC 61508) add 12–18 months.
- Cold chain: extreme temperature sensors add hardware complexity without additional value.
- Pharma: FDA 21 CFR Part 11 compliance.

### Sales Motion

**Phase 1 (Months 1–3): Design Partner**
- 1 pilot customer; free software in exchange for data and reference.
- Direct outreach to VP Operations at mid-market 3PLs in SE United States (lower labor costs, less union risk).
- Intro via WERC (Warehousing Education and Research Council) network.

**Phase 2 (Months 4–6): Land and Expand**
- Convert design partner to $5,000/month paid subscription.
- Use documented ROI (near-miss reduction, throughput gain) to close 2 additional pilots at $5,000/month each.
- Attend ProMat 2027 (March 2027) for industry visibility.

**Phase 3 (Months 7–12): Channel Scale**
- Recruit 2 systems integrator channel partners (VAR model, 20% margin).
- Inside sales team of 3 SDRs generating 20 qualified demos/month.
- Target: 50 sites at $5,000–$15,000/month (depending on site size) = $3M–$9M ARR.

### Pricing Model

| Tier | Price/Month | Included |
|---|---|---|
| Starter | $3,500 | 1 warehouse, 8 cameras, 5 forklifts, safety AI + basic agent |
| Professional | $7,500 | 1 warehouse, 16 cameras, 20 forklifts, full agent, dashboard |
| Enterprise | Custom ($15k+) | Multi-site, custom integrations, dedicated CSM, SLA |

Camera kit (hardware): leased at $800/month or purchased at $25,000.

---

## Investor Narrative

### The Problem

Warehouses are the most dangerous industrial workplaces in America. 95 workers die in warehouse accidents annually (BLS 2023). 340,000 forklift accidents occur each year. The average cost per incident is $42,000 (direct) to $150,000 (indirect, including OSHA fines, workers' comp, and productivity loss). Simultaneously, the US faces a structural warehouse labor shortage: 800,000 unfilled warehouse jobs (2025), with e-commerce demand growing at 15% annually.

Existing automation solutions (Symbotic, Ocado, Autostore) cost $20M–$80M per installation and require greenfield facility design. 90% of warehouses cannot afford to rebuild. They need intelligence layered on top of their existing forklifts, cameras, and workflows.

### The Opportunity

**Total Addressable Market:**
- 20,000 US mid-market warehouses × $90,000/year average contract = $1.8B US TAM.
- Global TAM (EU + APAC, 5× US): $9B.
- Adjacent: port terminals, manufacturing floors, airport baggage handling — same physics, same problems.

**Why Now:**
1. NVIDIA Isaac Sim makes photorealistic synthetic data generation commercially viable for the first time. We can train safety AI without putting workers at risk.
2. Claude Sonnet 4.6 and similar LLMs can reliably decompose NL instructions into structured robot commands — the "NL-to-robot" interface is finally ready.
3. OSHA enforcement is intensifying: proposed new powered industrial truck standard (2024) increases fines to $15,625 per serious violation.

### Why WarehouseGPT Wins

**Not just perception. Not just robotics. The OS.**

Most competitors solve one layer: Gather AI (inventory scanning drones), Locus Robotics (AMR dispatch), Vimaan (computer vision). WarehouseGPT integrates all layers into a single control plane with a natural language interface that any operator can use on day one.

**The flywheel:**
- More warehouses → more real data → better world model → better predictions → fewer accidents → more warehouses.
- Proprietary dataset compounds into an insurmountable moat. Competitors starting today would need 3 years to generate equivalent real-world safety event data.

**Key metrics for Series A pitch (Month 6):**
- 3 paying pilot customers.
- > 30% near-miss reduction documented at Site 1 (before/after comparison).
- > 95% safety detector recall on real warehouse data.
- NPS > 50 from warehouse operators (measure: ease of NL instruction use vs. legacy WMS UI).
- $180k ARR with clear path to $2M ARR at 20 sites.

---

## Technical Moats

### Moat 1: Synthetic Dataset at Scale

Isaac Sim generates 500K diverse, photo-realistic, labeled frames per day on 4× A100s. Generating this dataset cost us $50k in compute. Replicating it requires 6 months of engineering plus NVIDIA Isaac Sim expertise that is rare outside of robotics research labs. No publicly available warehouse safety dataset comes close in scale or label quality.

The dataset itself becomes a proprietary asset. Over 12 months we will accumulate 100M synthetic + 10M real labeled frames — more than any academic group or competitor has published.

### Moat 2: Warehouse-Specific World Model

A general-purpose world model (e.g., Google Genie 2, NVIDIA Cosmos) is not trained on forklift dynamics, pallet physics, or warehouse-specific spatial semantics. Our VQ-VAE + transformer is fine-tuned on warehouse data, making it dramatically more accurate for:
- Predicting whether a pallet stack will fall during pick.
- Forecasting whether a forklift and worker will intersect in 3 seconds.
- Detecting anomalies (unusual traffic patterns, objects in wrong zones) that indicate upcoming incidents.

This specificity cannot be replicated by fine-tuning a general model quickly; it requires the proprietary dataset.

### Moat 3: Sim-to-Real Calibration Playbook

Our DANN + CycleGAN calibration procedure can onboard a new warehouse site in 4 hours (vs. months of manual data collection for competitors). This operational efficiency directly translates to a lower cost of deployment and faster time-to-value for customers. The calibration playbook is internally documented and not reproducible from public papers alone.

### Moat 4: Multi-Agent Coordination Policy

Training a MARL policy that coordinates 10+ forklifts to maximize throughput while maintaining safety constraints is a 6-month research project. We are starting that research in Month 2. By Month 9 we will have a production-tested policy that no competitor has matched. This policy, combined with the world model's trajectory predictions, allows WarehouseGPT to safely schedule higher forklift density than any human dispatcher could manage.

### Moat 5: Safety Audit Trail and OSHA Compliance Integration

Every safety event, every E-stop, every mission is logged with full video evidence, AI confidence scores, and operator responses. This audit trail, stored for 7 years, is exactly what OSHA requires for their recordkeeping standard. Customers using WarehouseGPT can generate OSHA 300 and 301 logs automatically. This operational/compliance value is not about AI at all — it creates switching costs independent of model quality.
