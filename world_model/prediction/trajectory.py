"""
TrajectoryPredictor — predicts multi-modal future trajectories for
all tracked agents (forklifts, workers) in the warehouse scene.

Design
------
- Extracts per-agent features from transformer hidden states using
  spatial attention over agent bounding box regions.
- Outputs a Gaussian Mixture Model (GMM) distribution over future
  2D waypoints, capturing multi-modal futures (e.g., forklift can
  turn left, right, or stop).
- Computes pairwise collision probability between predicted trajectories.

Input:   hidden_states (B, L, d_model)  from WarehouseWorldModel
         agent_ids     dict mapping agent_id → bounding boxes per frame
Output:  dict[agent_id → TrajectoryPrediction]

Architecture
------------
  Per-agent:
    Crop + pool transformer features from agent BBox region
    → agent_feature (B, d_model)
    → 2-layer MLP
    → GMM params: (pi, mu, sigma) for K mixture components
    → M future waypoints each as (x, y) 2D position

  Collision:
    Compute minimum distance between Gaussian mixture samples
    → collision probability via Monte Carlo estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AgentBoundingBox:
    """
    2D bounding box of an agent in the token grid coordinate system.

    Coordinates are in the spatial grid (Ht × Wt), not pixels.
    """
    agent_id: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    frame_idx: int


@dataclass
class GMMOutput:
    """
    Gaussian Mixture Model output for a single agent's trajectory.

    Shapes (batch dims omitted for clarity):
        pi:    (num_modes,)       mixture weights (sum to 1)
        mu:    (num_modes, horizon, 2)  mean waypoints (x, y in metres)
        sigma: (num_modes, horizon, 2)  std devs (positive)
    """
    pi: Tensor     # (B, num_modes)
    mu: Tensor     # (B, num_modes, horizon, 2)
    sigma: Tensor  # (B, num_modes, horizon, 2)

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Draw samples from the GMM.

        Returns:
            samples: (B, n_samples, horizon, 2)
        """
        B, K, H, _ = self.mu.shape
        device = self.mu.device

        # Sample mixture component indices
        component = torch.multinomial(
            self.pi.view(B, K), num_samples=n_samples, replacement=True
        )  # (B, n_samples)

        # Gather parameters for selected components
        idx = component.unsqueeze(-1).unsqueeze(-1).expand(B, n_samples, H, 2)
        mu_sel = self.mu.unsqueeze(1).expand(B, n_samples, K, H, 2).gather(2, idx.unsqueeze(2).expand_as(
            self.mu.unsqueeze(1).expand(B, n_samples, K, H, 2)
        ))
        # Simpler direct gather
        mu_sel = torch.stack([
            self.mu[b][component[b]]  # (n_samples, H, 2)
            for b in range(B)
        ])  # (B, n_samples, H, 2)
        sigma_sel = torch.stack([
            self.sigma[b][component[b]]
            for b in range(B)
        ])  # (B, n_samples, H, 2)

        eps = torch.randn_like(sigma_sel)
        return mu_sel + sigma_sel * eps  # (B, n_samples, H, 2)

    def most_likely_trajectory(self) -> Tensor:
        """
        Returns:
            traj: (B, horizon, 2) — mean of the most likely mixture component
        """
        best_mode = self.pi.argmax(dim=-1)  # (B,)
        B = self.mu.shape[0]
        return torch.stack([self.mu[b, best_mode[b]] for b in range(B)])


class TrajectoryPrediction(NamedTuple):
    """Complete trajectory prediction for one agent."""
    agent_id: str
    gmm: GMMOutput                # GMM distribution over waypoints
    most_likely: Tensor           # (B, horizon, 2) best trajectory
    samples: Tensor               # (B, n_samples, horizon, 2) diverse samples


# ---------------------------------------------------------------------------
# Agent feature extractor
# ---------------------------------------------------------------------------

class AgentROIPooling(nn.Module):
    """
    Extract per-agent features from transformer hidden states by
    pooling over the agent's spatial bounding box region.

    Uses RoI Align (average pooling over the bbox region) on the
    (Ht, Wt) spatial grid.
    """

    def __init__(self, d_model: int, pool_size: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.proj = nn.Sequential(
            nn.Linear(d_model * pool_size * pool_size, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

    def forward(
        self,
        hidden_states: Tensor,
        bbox: AgentBoundingBox,
        T: int,
    ) -> Tensor:
        """
        Args:
            hidden_states: (B, L, d_model) where L = T * Ht * Wt
            bbox:          agent bounding box in grid coords
            T:             number of time steps
        Returns:
            agent_feat: (B, d_model) pooled representation at bbox.frame_idx
        """
        B, L, D = hidden_states.shape
        Ht = Wt = int((L // T) ** 0.5)

        # Extract frame at bbox.frame_idx
        frame_start = bbox.frame_idx * Ht * Wt
        frame_end = frame_start + Ht * Wt
        frame_feats = hidden_states[:, frame_start:frame_end, :]  # (B, Ht*Wt, D)
        frame_feats = frame_feats.view(B, Ht, Wt, D).permute(0, 3, 1, 2)  # (B, D, Ht, Wt)

        # Clamp bbox to valid range
        x_min = max(0, bbox.x_min)
        y_min = max(0, bbox.y_min)
        x_max = min(Wt - 1, bbox.x_max)
        y_max = min(Ht - 1, bbox.y_max)

        # Crop the region and adaptive-pool to pool_size × pool_size
        region = frame_feats[:, :, y_min : y_max + 1, x_min : x_max + 1]
        if region.shape[2] == 0 or region.shape[3] == 0:
            # Degenerate bbox: use whole frame
            region = frame_feats

        pooled = F.adaptive_avg_pool2d(region, (self.pool_size, self.pool_size))
        # (B, D, pool, pool) → (B, D * pool^2)
        flat = pooled.reshape(B, -1)
        return self.proj(flat)  # (B, d_model)


# ---------------------------------------------------------------------------
# GMM head
# ---------------------------------------------------------------------------

class GMMHead(nn.Module):
    """
    Outputs GMM parameters (pi, mu, sigma) for trajectory prediction.

    Args:
        d_model:    input feature dimension
        num_modes:  number of Gaussian mixture components
        horizon:    prediction horizon (number of waypoints)
        hidden_dim: MLP hidden dimension
    """

    def __init__(
        self,
        d_model: int,
        num_modes: int = 6,
        horizon: int = 12,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.horizon = horizon

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # pi:    (K,)
        # mu:    (K * horizon * 2,)
        # sigma: (K * horizon * 2,)  — predict log(sigma) for positivity
        output_dim = num_modes + num_modes * horizon * 2 * 2
        self.head = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, agent_feat: Tensor) -> GMMOutput:
        """
        Args:
            agent_feat: (B, d_model)
        Returns:
            GMMOutput
        """
        B = agent_feat.shape[0]
        K = self.num_modes
        H = self.horizon

        h = self.mlp(agent_feat)      # (B, hidden_dim)
        out = self.head(h)            # (B, output_dim)

        # Split output
        pi_logits = out[:, :K]                                # (B, K)
        mu_flat = out[:, K : K + K * H * 2]                  # (B, K*H*2)
        log_sigma_flat = out[:, K + K * H * 2 :]             # (B, K*H*2)

        pi = F.softmax(pi_logits, dim=-1)                     # (B, K)
        mu = mu_flat.view(B, K, H, 2)                         # (B, K, H, 2)
        sigma = torch.exp(log_sigma_flat.view(B, K, H, 2))    # (B, K, H, 2) > 0

        return GMMOutput(pi=pi, mu=mu, sigma=sigma)


# ---------------------------------------------------------------------------
# Collision probability estimator
# ---------------------------------------------------------------------------

def pairwise_collision_probability(
    traj_a: GMMOutput,
    traj_b: GMMOutput,
    collision_radius: float = 1.5,
    n_samples: int = 200,
) -> Tensor:
    """
    Estimate collision probability between two agents via Monte Carlo.

    Samples n_samples trajectories from each agent's GMM and checks
    whether any pair of waypoints comes within collision_radius metres.

    Args:
        traj_a:           GMM for agent A
        traj_b:           GMM for agent B
        collision_radius: distance threshold for collision (metres)
        n_samples:        number of Monte Carlo samples per agent
    Returns:
        prob: (B,) collision probability per batch element
    """
    # Sample trajectories: (B, n_samples, horizon, 2)
    samples_a = traj_a.sample(n_samples)
    samples_b = traj_b.sample(n_samples)

    # Compute pairwise distances across the sample dimension
    # samples_a: (B, n_samples, H, 2)  →  (B, n_samples, 1, H, 2)
    # samples_b: (B, n_samples, H, 2)  →  (B, 1, n_samples, H, 2)
    a = samples_a.unsqueeze(2)   # (B, Na, 1, H, 2)
    b = samples_b.unsqueeze(1)   # (B, 1, Nb, H, 2)
    dist = (a - b).norm(dim=-1)  # (B, Na, Nb, H)

    # Collision if distance < threshold at any waypoint
    collision_per_pair = (dist < collision_radius).any(dim=-1)  # (B, Na, Nb)
    prob = collision_per_pair.float().mean(dim=(1, 2))           # (B,)
    return prob


# ---------------------------------------------------------------------------
# TrajectoryPredictor
# ---------------------------------------------------------------------------

class TrajectoryPredictor(nn.Module):
    """
    Predicts future waypoints for all tracked agents in the scene.

    Args:
        d_model:          transformer hidden dimension
        num_modes:        GMM mixture components (default: 6)
        horizon:          prediction horizon in frames (default: 12)
        n_samples:        number of trajectory samples for collision check
        collision_radius: collision distance threshold in metres
        pool_size:        RoI pooling output size
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_modes: int = 6,
        horizon: int = 12,
        n_samples: int = 200,
        collision_radius: float = 1.5,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes
        self.horizon = horizon
        self.n_samples = n_samples
        self.collision_radius = collision_radius

        self.roi_pool = AgentROIPooling(d_model=d_model, pool_size=pool_size)
        self.gmm_head = GMMHead(
            d_model=d_model,
            num_modes=num_modes,
            horizon=horizon,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_trajectories(
        self,
        hidden_states: Tensor,
        agent_bboxes: dict[str, list[AgentBoundingBox]],
        T: int,
    ) -> dict[str, TrajectoryPrediction]:
        """
        Predict future waypoints for each tracked agent.

        Args:
            hidden_states: (B, L, d_model)
            agent_bboxes:  dict mapping agent_id → list of AgentBoundingBox
                           (one bbox per batch element, in the last context frame)
            T:             number of time steps in hidden_states
        Returns:
            predictions:   dict[agent_id → TrajectoryPrediction]
        """
        predictions: dict[str, TrajectoryPrediction] = {}

        for agent_id, bboxes in agent_bboxes.items():
            # Use the last observed bbox for ROI pooling
            bbox = bboxes[-1]

            agent_feat = self.roi_pool(hidden_states, bbox, T)  # (B, d_model)
            gmm_out = self.gmm_head(agent_feat)

            most_likely = gmm_out.most_likely_trajectory()  # (B, horizon, 2)
            samples = gmm_out.sample(self.n_samples)         # (B, n_samples, horizon, 2)

            predictions[agent_id] = TrajectoryPrediction(
                agent_id=agent_id,
                gmm=gmm_out,
                most_likely=most_likely,
                samples=samples,
            )

        return predictions

    def collision_matrix(
        self,
        predictions: dict[str, TrajectoryPrediction],
    ) -> dict[tuple[str, str], Tensor]:
        """
        Compute pairwise collision probabilities between all agent pairs.

        Args:
            predictions: output of predict_trajectories()
        Returns:
            collision_probs: dict[(agent_a, agent_b) → (B,) probability tensor]
        """
        agent_ids = list(predictions.keys())
        collision_probs: dict[tuple[str, str], Tensor] = {}

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id_a = agent_ids[i]
                id_b = agent_ids[j]
                prob = pairwise_collision_probability(
                    predictions[id_a].gmm,
                    predictions[id_b].gmm,
                    collision_radius=self.collision_radius,
                    n_samples=self.n_samples,
                )
                collision_probs[(id_a, id_b)] = prob

        return collision_probs

    def compute_loss(
        self,
        hidden_states: Tensor,
        agent_bboxes: dict[str, list[AgentBoundingBox]],
        ground_truth_waypoints: dict[str, Tensor],
        T: int,
    ) -> Tensor:
        """
        Negative log-likelihood loss under the GMM for each agent.

        Uses the Winner-Takes-All (WTA) / best-of-K strategy:
        assigns loss only to the mixture component whose mean is
        closest to the ground truth.

        Args:
            hidden_states:          (B, L, d_model)
            agent_bboxes:           same as predict_trajectories()
            ground_truth_waypoints: dict[agent_id → (B, horizon, 2)] gt traj
            T:                      number of time steps
        Returns:
            loss: scalar NLL
        """
        total_loss = torch.tensor(0.0, device=hidden_states.device)
        count = 0

        for agent_id, bboxes in agent_bboxes.items():
            if agent_id not in ground_truth_waypoints:
                continue

            gt = ground_truth_waypoints[agent_id]  # (B, horizon, 2)
            bbox = bboxes[-1]
            agent_feat = self.roi_pool(hidden_states, bbox, T)
            gmm = self.gmm_head(agent_feat)

            loss = self._gmm_nll(gmm, gt)
            total_loss = total_loss + loss
            count += 1

        return total_loss / max(count, 1)

    def _gmm_nll(self, gmm: GMMOutput, gt: Tensor) -> Tensor:
        """
        Winner-Takes-All NLL loss.

        Args:
            gmm: GMMOutput from gmm_head
            gt:  (B, horizon, 2) ground truth waypoints
        Returns:
            loss: scalar
        """
        B, K, H, _ = gmm.mu.shape
        device = gt.device

        # Expand gt for comparison with each mode: (B, K, H, 2)
        gt_exp = gt.unsqueeze(1).expand(B, K, H, 2)

        # Euclidean distance to each mode's mean: (B, K)
        dist_to_mean = (gmm.mu - gt_exp).norm(dim=-1).mean(dim=-1)

        # Best mode
        best_mode = dist_to_mean.argmin(dim=-1)  # (B,)

        # NLL of gt under the best mode's Gaussian
        mu_best = torch.stack([gmm.mu[b, best_mode[b]] for b in range(B)])      # (B, H, 2)
        sigma_best = torch.stack([gmm.sigma[b, best_mode[b]] for b in range(B)])  # (B, H, 2)
        pi_best = torch.stack([gmm.pi[b, best_mode[b]] for b in range(B)])      # (B,)

        # Gaussian log-likelihood
        log_prob = -0.5 * (
            ((gt - mu_best) / (sigma_best + 1e-6)) ** 2
            + 2 * torch.log(sigma_best + 1e-6)
            + math.log(2 * math.pi)
        )  # (B, H, 2)
        nll = -log_prob.sum(dim=(-1, -2))   # (B,)
        nll = nll - torch.log(pi_best + 1e-6)  # weight by mixture prob

        # Diversity regulariser: encourage modes to be spread out
        pairwise_dist = (
            gmm.mu.unsqueeze(1) - gmm.mu.unsqueeze(2)
        ).norm(dim=-1).mean(dim=-1)  # (B, K, K)
        diversity_loss = -pairwise_dist.mean()

        return nll.mean() + 0.1 * diversity_loss
