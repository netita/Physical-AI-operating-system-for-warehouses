"""
warehousegpt.forklift_rl
=========================
Autonomous Forklift Reinforcement Learning subsystem — Phase 4 of WarehouseGPT.

Provides Isaac Lab-integrated Gymnasium environments and training
infrastructure for learning forklift navigation, pallet manipulation,
and multi-agent coordination.

Sub-packages
------------
environments   — Gymnasium-compatible RL environments
algorithms     — Algorithm comparison and recommendation framework
training       — RLTrainer: vectorised, curriculum-aware training loop

Quick start
-----------
::

    from warehousegpt.forklift_rl.environments import NavigationEnv, PalletPickupEnv
    from warehousegpt.forklift_rl.training import RLTrainer

    trainer = RLTrainer(
        env_id="NavigationEnv",
        algorithm="PPO",
        n_envs=4096,
    )
    trainer.train(total_timesteps=50_000_000)
"""

from __future__ import annotations

from warehousegpt.forklift_rl.algorithms.comparison import AlgorithmComparison
from warehousegpt.forklift_rl.environments.base_env import WarehouseBaseEnv
from warehousegpt.forklift_rl.environments.multi_agent import MultiAgentWarehouseEnv
from warehousegpt.forklift_rl.environments.navigation import NavigationEnv
from warehousegpt.forklift_rl.environments.pallet_pickup import PalletPickupEnv
from warehousegpt.forklift_rl.training.trainer import RLTrainer

__all__: list[str] = [
    "WarehouseBaseEnv",
    "NavigationEnv",
    "PalletPickupEnv",
    "MultiAgentWarehouseEnv",
    "AlgorithmComparison",
    "RLTrainer",
]
