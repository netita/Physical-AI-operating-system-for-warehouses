# Sim-to-Real Transfer Strategy

## Overview

The sim-to-real gap is the primary technical risk in WarehouseGPT. Models trained exclusively on Isaac Sim data will encounter distribution shift when deployed on real warehouse cameras, with real lighting, real dust, real lens distortion, and real sensor noise. This document describes our methodology to minimize and measure this gap.

---

## 1. Domain Randomization Methodology

Domain randomization (DR) is applied during data collection in Isaac Sim via `isaac_sim/domain_randomizer.py`. The philosophy is to generate such a wide distribution of synthetic conditions that real-world conditions fall within the training distribution.

### 1.1 Visual Randomization

| Parameter | Range | Distribution |
|---|---|---|
| Ambient light intensity | 100–3000 lux | Uniform |
| Directional light angle | ±60° azimuth, 20°–80° elevation | Uniform |
| Light color temperature | 2700K–6500K (warm to cool white) | Uniform |
| Shadow softness | 0.0–1.0 (hard to soft) | Uniform |
| Camera exposure | –2 to +2 EV | Gaussian (σ=0.5) |
| Motion blur kernel | 0–5 px | Exponential |
| JPEG compression quality | 60–100 | Uniform |
| Lens vignetting | 0–0.3 | Uniform |
| Chromatic aberration | 0–2 px | Exponential |
| Gaussian image noise (σ) | 0–0.03 (normalized) | Uniform |
| Salt-and-pepper noise rate | 0–0.001 | Uniform |

### 1.2 Texture and Material Randomization

| Asset Class | Randomization |
|---|---|
| Floor surface | 50+ texture variants (concrete, epoxy, painted lines) |
| Pallet wrap | Random wrap color (clear, blue, black, white, mixed) |
| Shelf metal | Rust, paint wear, reflectivity variation |
| Forklift livery | Random color + logo placement |
| Worker clothing | 20+ vest colors, random PPE combinations |
| Ambient objects (boxes) | Random label placement, wear level, color |

### 1.3 Geometric Randomization

| Parameter | Range |
|---|---|
| Camera pose (pitch) | ±5° from nominal |
| Camera pose (yaw) | ±3° from nominal |
| Camera height | ±0.2 m from nominal |
| Pallet stack height variation | 0–3 layers |
| Shelf slot occupancy | 20%–100% fill rate |
| Aisle width | 2.5 m – 4.5 m (parameterized scene) |

### 1.4 Physics Randomization

| Parameter | Range |
|---|---|
| Floor friction coefficient | 0.3–0.8 |
| Pallet weight | ±20% of nominal |
| Forklift acceleration profile | ±15% of nominal |
| Hydraulic lift speed | ±10% of nominal |
| Worker walking speed | 0.8–1.8 m/s |

### 1.5 Environmental Event Randomization

To train safety detectors:

| Event | Frequency | Severity Range |
|---|---|---|
| Fire / smoke | 0.5% of frames | Small flame to full bay fire |
| Forklift near-miss | 2% of episodes | 0.5 m – 2.5 m clearance |
| Worker zone violation | 3% of frames | 0–10 s duration |
| Spill / slip hazard | 1% of frames | Small puddle to large spill |
| PPE violation | 5% of frames | Missing 1–3 PPE items |
| Pallet drop | 0.1% of episodes | Full pallet collapse |

---

## 2. Domain Adaptation Techniques

Even with aggressive DR, a residual distribution shift typically remains. We apply two adaptation techniques to close the remaining gap.

### 2.1 Domain-Adversarial Neural Networks (DANN)

**Reference:** Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016).

**Architecture:**

```
Input Frame
     │
     ▼
Feature Extractor (ResNet-50 shared backbone)
     │
     ├──► Task Classifier (detection / segmentation head)
     │         Loss: task-specific (CE, IoU)
     │
     └──► Domain Classifier (binary: sim or real)
               Loss: binary CE with gradient reversal
               Gradient Reversal Layer (λ scheduled 0→1)
```

**Training procedure:**

1. Collect 10K–50K unlabeled real warehouse frames (no annotation needed).
2. Train feature extractor jointly:
   - Minimize task loss on labeled synthetic data.
   - Minimize domain classifier accuracy (via gradient reversal) on mixed sim+real batches.
3. λ schedule: `λ(p) = 2/(1 + exp(-10p)) - 1`, where `p` is training progress 0→1.

**Expected gain:** +3–8 mAP on real-world detection benchmark from DANN.

**Implementation location:** `safety_ai/training/pipeline.py` (DANN loss), `world_model/training/trainer.py` (domain mixing).

### 2.2 CycleGAN for Warehouse Imagery Translation

**Reference:** Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (ICCV 2017).

**Use case:** Translate synthetic Isaac Sim frames into the visual style of real camera feeds, providing additional training data that matches real sensor characteristics.

**Architecture:**

```
Synthetic Frame (S)                Real Frame (R)
       │                                  │
       ▼                                  ▼
   G: S → R                          F: R → S
(Generator: synthetic                (Generator: real
 to real style)                       to synthetic)
       │                                  │
       ▼                                  ▼
   Fake Real                          Fake Synth
       │                                  │
 D_R (discriminator)              D_S (discriminator)
       │                                  │
   Cycle: F(G(S)) ≈ S              Cycle: G(F(R)) ≈ R
         Identity: G(R) ≈ R               Identity: F(S) ≈ S
```

**Training data required:**

- 50K synthetic frames from Isaac Sim (unpaired).
- 10K real frames from production cameras (unpaired, unlabeled).
- Training time: ~24 hours on 1× A100 for a warehouse-specific model.

**Losses:**

- Adversarial loss (LSGAN variant): ensures style transfer.
- Cycle consistency loss (λ=10): preserves content.
- Identity loss (λ=5): prevents color shift when input already matches target domain.

**Augmentation:** Generated fake-real frames are mixed into the perception model training set at a 30% real / 70% synthetic ratio.

**Limitation:** CycleGAN can alter fine-grained textures (pallet labels, SKU text) which can degrade OCR tasks. Apply selectively to structural scene elements only; mask text regions.

**Implementation location:** `synthetic_data/augmentation.py` (`CycleGANAugmenter` class).

---

## 3. Real-World Calibration Procedure

Before deploying to a new warehouse site, the following calibration steps are executed:

### 3.1 Camera Intrinsic Calibration

1. Place a 9×7 checkerboard in 30+ positions covering the full FOV of each camera.
2. Run OpenCV `calibrateCamera` on the image set.
3. Store intrinsic matrix K and distortion coefficients per camera in PostgreSQL (`camera_calibrations` table).
4. Apply `cv2.undistort` to all frames before inference.

**Acceptance criterion:** Reprojection error < 0.5 px RMS.

### 3.2 Camera Extrinsic / World-Frame Calibration

1. Place AprilTag (36h11 family) or QR code fiducials at 8–12 known warehouse floor positions.
2. Run SolvePnP to compute camera-to-world transform for each camera.
3. Update `camera_extrinsics` table; Digital Twin uses these transforms to project detections into world coordinates.

**Acceptance criterion:** Localization error < 5 cm at 10 m range.

### 3.3 LiDAR-Camera Fusion Calibration

1. Place a calibration target (reflective board with AprilTag) visible to both LiDAR and camera.
2. Run ICP-based extrinsic calibration to align LiDAR point cloud with camera frame.
3. Validate by projecting LiDAR points onto camera image and verifying edge alignment.

**Acceptance criterion:** Reprojection RMSE < 2 px on calibration target edges.

### 3.4 Model Adaptation Run

After hardware calibration:

1. Collect 2K real frames from the new site cameras (15 minutes of footage, no annotation).
2. Run DANN fine-tuning for 1 epoch on the new site data (unlabeled) + original labeled synthetic data.
3. Run CycleGAN inference on the synthetic training set using a site-specific translator if color/lighting is very different from previously seen sites.
4. Re-export perception models to TensorRT INT8 (quantize on site-specific calibration data).

**Duration:** ~4 hours end-to-end on 1× A100.

### 3.5 Safety Threshold Calibration

Real detection confidence distributions may shift from simulation. Recalibrate thresholds:

1. Collect 500 labeled real frames (human annotation, ~2 hours).
2. Plot precision-recall curves for each safety detector on real data.
3. Set operating threshold at F1-optimal point per class.
4. Document thresholds in `safety_ai/configs/site_{site_id}.yaml`.

---

## 4. Validation Protocol: Simulation vs. Real Metrics

### 4.1 Perception Model Validation

Benchmark evaluated on three splits:

| Split | Composition | Size |
|---|---|---|
| Synth-Test | Unseen synthetic scenes | 5K images |
| Real-Labeled | Hand-labeled real frames from pilot site | 1K images |
| Real-Unlabeled-Proxy | CycleGAN-translated synthetic frames | 5K images |

**Metrics:**

| Metric | Definition | Synth-Test Target | Real-Labeled Target |
|---|---|---|---|
| mAP@50 (detection) | COCO mAP at IoU=0.5 | > 85% | > 72% |
| mAP@50-95 (detection) | COCO mAP at IoU=0.5:0.95 | > 60% | > 48% |
| mIoU (segmentation) | Mean IoU per class | > 78% | > 65% |
| Pose error (R) | Mean rotation error | < 3° | < 6° |
| Pose error (t) | Mean translation error at 2 m | < 2 cm | < 5 cm |
| Near-miss recall | True positive rate (safety-critical) | > 98% | > 95% |
| Near-miss precision | Precision (avoid false alarms) | > 90% | > 85% |

**Validation frequency:** After every model update, before deployment to production.

### 4.2 World Model Validation

| Metric | Definition | Target |
|---|---|---|
| Reconstruction FID | Frechet Inception Distance (synth ↔ reconstructed) | < 15 |
| Prediction SSIM | Structural similarity at t+1s | > 0.80 |
| Occupancy IoU | Grid cell match at 50 ms | > 0.85 |
| Trajectory ADE (1s) | Average displacement error at 1 second | < 0.3 m |
| Trajectory ADE (5s) | Average displacement error at 5 seconds | < 1.2 m |
| Trajectory FDE (5s) | Final displacement error at 5 seconds | < 2.0 m |

### 4.3 Safety Detector Validation

Safety detectors operate in a zero-tolerance regime. The validation criteria are:

| Detector | Target Recall (Real) | Max False Alarm Rate | Latency Target |
|---|---|---|---|
| Fire/smoke | > 99% | < 0.001 per frame | < 50 ms |
| Near-miss (< 2 m) | > 97% | < 0.005 per frame | < 30 ms |
| Near-miss (< 1 m) | > 99.5% | < 0.002 per frame | < 30 ms |
| PPE violation | > 92% | < 0.01 per frame | < 50 ms |
| Zone violation | > 95% | < 0.002 per frame | < 30 ms |

**Methodology:** For safety detectors, recall is the primary metric. We tune thresholds to achieve target recall on the real-labeled set, accepting the resulting precision.

### 4.4 End-to-End System Validation

Before first deployment at a new warehouse site:

1. **Shadow mode (2 weeks):** Run WarehouseGPT in parallel with existing WMS. Compare missions created, safety alerts raised, and inventory states. Do not actuate forklifts.
2. **Teleoperated validation (1 week):** Allow the system to dispatch forklifts on low-stakes tasks (single-zone moves) with a human operator shadowing each vehicle.
3. **Supervised autonomous (2 weeks):** Full autonomy with a safety officer available to E-stop. Monitor all alerts and anomalies.
4. **Production release:** Full autonomous operation for approved mission types.

---

## 5. Known Sim-to-Real Gaps and Mitigations

### 5.1 Lighting Realism

**Gap:** Isaac Sim global illumination is physically accurate but lacks real camera sensor effects: sensor noise at different ISO settings, auto-exposure hunting, bloom around point sources, and rolling-shutter artifacts.

**Mitigation:**
- Apply camera noise injection during synthetic data generation (modeled from real camera spectral response).
- Capture real dark/bright frames for DANN fine-tuning.
- Use GAN discriminator that operates on camera RAW-simulated artifacts.

**Residual risk:** Low. Most warehouse environments have controlled lighting.

### 5.2 Forklift Appearance

**Gap:** Synthetic forklift models are clean CAD meshes. Real forklifts accumulate dirt, dents, and non-standard attachments that confuse pose estimators.

**Mitigation:**
- Apply procedural wear-and-dirt texture randomization in Replicator.
- Add a "forklift identification" fine-tuning step at each new site with 50 labeled frames per forklift model.

**Residual risk:** Medium. Custom attachments (extended forks, side-shifters) require site-specific model additions.

### 5.3 Occlusion Patterns

**Gap:** Simulation generates clean inter-object occlusions. Real warehouses have semi-permanent occlusions (hanging cables, support columns, shrink-wrap curtains) not modeled in default scenes.

**Mitigation:**
- Add warehouse-specific occluder assets to the Isaac Sim scene during site setup.
- Train with heavy random object insertion as occluders.

**Residual risk:** Low–Medium. Fixed occluders can be added to the scene model during commissioning.

### 5.4 Human Behavior Distribution

**Gap:** Simulated worker trajectories are generated by simple procedural models. Real workers exhibit more varied, less predictable paths, including improper shortcuts and distracted behavior.

**Mitigation:**
- Collect real worker trajectory data from camera tracking (anonymized) for at least 1 week before activating near-miss detection.
- Fine-tune trajectory prediction model on real data.
- Apply conservative safety margins (+30% distance buffer) until real-data fine-tuning is complete.

**Residual risk:** Medium. Human behavior is inherently stochastic; the world model may underestimate rare but dangerous behaviors.

### 5.5 Sensor Timing and Latency

**Gap:** Simulation frames are perfectly synchronized. Real camera streams have varying capture timestamps and clock drift across the network.

**Mitigation:**
- Use PTP (IEEE 1588) hardware timestamping on all cameras.
- Apply Kalman filter interpolation for temporal alignment.
- Test with intentional timestamp jitter in simulation (up to ±5 ms).

**Residual risk:** Low. PTP achieves < 1 ms sync; detection models are robust to 1-frame delays.

### 5.6 World Model Hallucination in Occluded Regions

**Gap:** The world model may hallucinate objects or workers in occluded regions (behind shelves) based on learned priors. A misplaced hallucination could generate a false safety alert or, worse, suppress a real one.

**Mitigation:**
- World model predictions are used only for planning, never for safety decisions.
- All safety alerts must be confirmed by at least one camera with direct line of sight.
- Tag world model outputs with per-region confidence; suppress alerts from low-confidence regions.

**Residual risk:** Medium. This is an active research area. Current mitigation is conservative: default to physical sensor data for all safety-critical decisions.
