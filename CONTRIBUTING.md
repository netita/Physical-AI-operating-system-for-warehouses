# Contributing to WarehouseGPT

Thank you for contributing to WarehouseGPT. This document covers the development setup, coding standards, testing requirements, and pull request process.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Code Style and Standards](#code-style-and-standards)
5. [Testing](#testing)
6. [Pull Request Process](#pull-request-process)
7. [Commit Message Convention](#commit-message-convention)
8. [Working with Isaac Sim](#working-with-isaac-sim)
9. [Working with ROS 2](#working-with-ros-2)
10. [Secrets and Environment Variables](#secrets-and-environment-variables)
11. [Reporting Issues](#reporting-issues)

---

## Prerequisites

Before starting, ensure you have the following installed and configured:

**Required:**

- Ubuntu 22.04 LTS (Jammy Jellyfish) — other Linux distributions are not officially supported.
- Python 3.10 or 3.11 (3.12+ is not supported due to dependency constraints).
- Docker 25+ with the NVIDIA Container Toolkit installed and configured.
- Docker Compose v2.24+.
- `git` 2.39+.
- `make` (GNU Make).
- An NVIDIA GPU with CUDA 12.x drivers (Ampere or newer strongly recommended). Minimum: RTX 3080 with 10 GB VRAM for local development; A100 for training.

**Optional (for specific components):**

- NVIDIA Isaac Sim 4.x — required for simulation and synthetic data generation. Download from [NVIDIA Omniverse](https://developer.nvidia.com/isaac-sim).
- ROS 2 Humble — required for robot control bridge development. Install via [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).
- `hatch` — the project build and environment manager. Install via `pip install hatch`.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/warehousegpt.git
cd warehousegpt

# Add the upstream remote
git remote add upstream https://github.com/<your-org>/warehousegpt.git
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the required values. At minimum for local development:

```bash
# Required: get your Anthropic API key from https://console.anthropic.com
ANTHROPIC_API_KEY=sk-ant-...

# Required: generate secure random values
POSTGRES_PASSWORD=$(openssl rand -hex 32)
REDIS_PASSWORD=$(openssl rand -hex 32)
NEO4J_PASSWORD=$(openssl rand -hex 16)
JWT_SECRET_KEY=$(openssl rand -hex 32)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)

# Optional: set to your GPU device indices
CUDA_VISIBLE_DEVICES=0
```

### 3. Install Python Dependencies

```bash
# Install hatch if not already present
pip install hatch

# Create the default development environment
hatch env create

# Activate the environment
hatch shell
```

For component-specific development, create the relevant hatch environment:

```bash
# For training work
hatch env create train
hatch run train:run --help

# For inference/TensorRT work
hatch env create infer
hatch run infer:build-engines --help

# For simulation work (requires Isaac Sim installation)
hatch env create sim
hatch run sim:gen-data --help
```

### 4. Start the Infrastructure Stack

```bash
make setup
```

This command runs `docker compose up -d` and waits for all services to pass their health checks. It typically takes 60–90 seconds for all services (PostgreSQL, Redis, Neo4j, ChromaDB, Triton, Prometheus, Grafana) to become healthy.

Verify the stack is healthy:

```bash
docker compose ps
# All services should show "healthy"
```

### 5. Initialize the Knowledge Graph and Vector Store

```bash
python scripts/seed_knowledge_graph.py
```

This seeds Neo4j with the default warehouse schema (zones, shelves, dock doors) and indexes the default SOP documents in ChromaDB.

### 6. Run the Test Suite

```bash
make test
```

Or run specific test levels:

```bash
hatch run test:unit         # Pure unit tests, no external services needed
hatch run test:integration  # Requires running Docker Compose stack
```

### 7. Start Development Servers

```bash
# Terminal 1: API Gateway with hot-reload
hatch run api:serve

# Terminal 2: WarehouseGPT Agent
hatch run agent:serve
```

The API will be available at `http://localhost:8000`. Interactive API docs are at `http://localhost:8000/docs`.

Grafana dashboards: `http://localhost:3000` (admin / see `GRAFANA_ADMIN_PASSWORD` in `.env`).

---

## Project Structure

```
warehousegpt/
├── apps/                    # Runnable services (FastAPI, agent, fleet manager)
│   ├── api_gateway/         # REST + WebSocket API server
│   ├── agent/               # WarehouseGPT Claude agent (LangGraph)
│   ├── fleet_manager/       # AMR task assignment
│   ├── mission_scheduler/   # Mission decomposition
│   └── digital_twin_sync/   # Isaac Sim ↔ real-world bridge
├── packages/                # Shared importable packages (no runnable entry points)
│   ├── perception/          # Triton client, TensorRT wrappers
│   ├── ros2_bridge/         # rclpy nodes and message translators
│   ├── knowledge/           # ChromaDB, Neo4j, PostgreSQL adapters
│   ├── simulation/          # Omniverse / Isaac Sim scripting helpers
│   └── common/              # Shared types, configs, utilities
├── world_model/             # VQ-VAE tokenizer + transformer world model
├── safety_ai/               # Safety detectors, labeling, training
├── forklift_rl/             # RL environments, algorithms, training
├── synthetic_data/          # Isaac Sim data pipeline and augmentation
├── isaac_sim/               # Isaac Sim scene generation scripts
├── models/                  # ONNX + TensorRT engine files (gitignored)
├── infra/                   # Kubernetes manifests, Terraform, monitoring configs
├── tests/
│   ├── unit/                # Unit tests (no external services)
│   ├── integration/         # Integration tests (requires Docker Compose)
│   └── simulation/          # Isaac Sim regression tests
├── scripts/                 # One-off utility scripts
├── docker/                  # Dockerfiles per service
├── docs/                    # Technical documentation
├── docker-compose.yml
├── pyproject.toml           # Project metadata, dependencies, tool configs
├── Makefile                 # Developer shortcuts
├── .env.example
└── README.md
```

---

## Code Style and Standards

### Python Style

All Python code must pass the Ruff linter and Mypy type checker with the project's strict configuration (defined in `pyproject.toml`).

```bash
# Check and auto-fix linting issues
hatch run default:lint

# Auto-format code
hatch run default:fmt
```

**Key requirements:**

- All public functions and methods must have type annotations.
- All modules must have a module-level docstring.
- No bare `except:` clauses; always catch specific exception types.
- Use `structlog` for logging (not `print` or `logging.basicConfig`).
- Use `pydantic` models for all request/response schemas and configuration.
- Use `async`/`await` for all I/O-bound operations (database, HTTP, Redis).
- Line length: 100 characters.

### Naming Conventions

| Category | Convention | Example |
|---|---|---|
| Python files | snake_case | `warehouse_generator.py` |
| Python classes | PascalCase | `WarehouseGenerator` |
| Python functions/methods | snake_case | `generate_scene()` |
| Python constants | UPPER_SNAKE_CASE | `MAX_FORKLIFT_SPEED_MPS` |
| Environment variables | UPPER_SNAKE_CASE | `ANTHROPIC_API_KEY` |
| Docker service names | kebab-case | `api-gateway` |
| Kubernetes resource names | kebab-case | `warehousegpt-api` |

### Docstrings

Use Google-style docstrings:

```python
def dispatch_forklift(
    forklift_id: str,
    task: ForkliftTask,
    timeout_seconds: float = 30.0,
) -> MissionResult:
    """Assign a task to a specific forklift and wait for acknowledgment.

    Args:
        forklift_id: The unique identifier of the target forklift (e.g., "fk-001").
        task: The structured task definition including source and destination.
        timeout_seconds: Maximum time to wait for forklift acknowledgment.

    Returns:
        A MissionResult containing the mission ID and initial status.

    Raises:
        ForkliftUnavailableError: If the forklift is currently in fault state or
            already executing a higher-priority mission.
        MissionTimeoutError: If the forklift does not acknowledge within timeout_seconds.
    """
```

---

## Testing

### Test Categories

| Category | Location | When to run | External deps |
|---|---|---|---|
| Unit | `tests/unit/` | Before every commit | None |
| Integration | `tests/integration/` | Before every PR | Docker Compose stack |
| Simulation | `tests/simulation/` | Before releases | Isaac Sim + GPU |

### Running Tests

```bash
# Quick unit test run (< 30 seconds)
hatch run test:unit

# Full integration test suite (requires running stack)
hatch run test:integration

# All tests with coverage report
hatch run test:all

# Run a single test file
hatch run test:unit tests/unit/test_agent.py

# Run tests matching a keyword
hatch run test:unit -k "test_near_miss"
```

### Writing Tests

- Unit tests must not start any external services, read from `.env`, or make network calls. Use `pytest-mock` and `respx` for mocking.
- Integration tests must use fixtures that reset database state between tests.
- Every new feature or bug fix must include a corresponding test.
- Safety-critical code (safety detectors, E-stop logic) requires > 95% unit test coverage.
- Use `@pytest.mark.gpu` for tests that require a CUDA GPU.

### Test Fixture Guidelines

```python
# tests/unit/test_safety.py — example
import pytest
from unittest.mock import AsyncMock, MagicMock
from safety_ai.detectors.near_miss import NearMissDetector

@pytest.fixture
def mock_triton_client():
    client = MagicMock()
    client.infer = AsyncMock(return_value={"detections": []})
    return client

@pytest.mark.unit
async def test_near_miss_alert_fires_at_threshold(mock_triton_client):
    detector = NearMissDetector(triton_client=mock_triton_client, threshold_m=2.0)
    # ... test body
```

---

## Pull Request Process

1. **Sync with upstream before starting:**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   # or for bug fixes:
   git checkout -b fix/description-of-bug
   ```

3. **Write code, tests, and documentation** for your change.

4. **Run the full local check:**
   ```bash
   make lint test
   ```
   Both must pass before opening a PR.

5. **Commit using the conventional format** (see below).

6. **Open a Pull Request** against `main` with:
   - A clear title (< 72 characters).
   - A description explaining WHY the change is needed (not just what it does).
   - Reference to any related issues (`Closes #123`).
   - The "Testing" section: what test coverage was added.
   - Any breaking changes called out explicitly.

7. **PR review requirements:**
   - Minimum 1 approval from a codeowner (see `CODEOWNERS`).
   - All CI checks pass (Ruff, Mypy, unit tests, Docker build).
   - Safety-critical changes require approval from the ML Lead or Safety Lead in addition to a codeowner.

8. **Squash merge policy:** PRs are squash-merged. The squash commit message is taken from the PR title, so make it descriptive.

---

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short summary>

[optional body: explain WHY, not just what]

[optional footer: BREAKING CHANGE, Closes #issue]
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or correcting tests |
| `docs` | Documentation changes only |
| `chore` | Build system, CI, dependencies |
| `safety` | Changes to safety-critical code paths |

**Scopes** (optional but recommended): `agent`, `world-model`, `safety-ai`, `perception`, `forklift-rl`, `digital-twin`, `fleet`, `api`, `sim`, `infra`, `docs`.

**Examples:**

```
feat(safety-ai): add PPE compliance detector with YOLO backbone

fix(agent): handle malformed tool call JSON with retry logic

safety(near-miss): tighten distance threshold from 2.0m to 1.8m after pilot data review

BREAKING CHANGE: NearMissDetector constructor now requires triton_client keyword argument
```

---

## Working with Isaac Sim

Isaac Sim requires a headless Linux environment with an NVIDIA GPU and the Omniverse launcher. For team members without local Isaac Sim access, use the shared simulation server (ask DevOps for SSH access).

```bash
# Generate synthetic data (on simulation server)
hatch run sim:gen-data \
  --scene-config isaac_sim/configs/warehouse_default.yaml \
  --output-dir /data/synthetic/run_001 \
  --num-frames 100000 \
  --randomize true

# Run headless Isaac Sim scene validation
python isaac_sim/warehouse_generator.py --validate --config isaac_sim/configs/warehouse_default.yaml
```

Do not commit Isaac Sim USD scene files (*.usd, *.usda, *.usdc) to git. Store them on the shared NFS drive and reference them by path in configuration YAML files.

---

## Working with ROS 2

ROS 2 Humble must be sourced before running any `ros2` or `rclpy` code:

```bash
source /opt/ros/humble/setup.bash

# Or add to your shell profile:
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

Build the ROS 2 bridge package:

```bash
cd packages/ros2_bridge
colcon build --symlink-install
source install/setup.bash
```

Run ROS 2 unit tests:

```bash
colcon test --packages-select ros2_bridge
colcon test-result --verbose
```

Do not mix ROS 2 `rclpy` imports with regular Python imports in the same module without proper guarding, as `rclpy.init()` must be called before any node creation.

---

## Secrets and Environment Variables

**Never commit secrets to git.** This includes:

- API keys (`ANTHROPIC_API_KEY`, etc.)
- Database passwords
- JWT secret keys
- TLS certificates

The `.env` file is in `.gitignore` and must never be committed.

If you accidentally commit a secret:

1. Immediately rotate the compromised credential.
2. Use `git filter-branch` or BFG Repo-Cleaner to remove it from history.
3. Force-push the cleaned history (coordinate with the team first).
4. Notify the security team.

For adding new environment variables:

1. Add the variable with a placeholder value to `.env.example`.
2. Add a comment explaining what the variable is, where to get the value, and whether it is required or optional.
3. Add validation in the relevant `pydantic-settings` `Settings` class.
4. Update the `docs/` if the variable affects a documented behavior.

---

## Reporting Issues

Use the GitHub Issues tracker. Before opening a new issue:

1. Search existing issues to avoid duplicates.
2. For bugs: include OS, Python version, GPU model, Docker version, the full error message and stack trace, and steps to reproduce.
3. For safety-critical bugs (missed safety alerts, incorrect E-stop behavior): open a GitHub Issue and mark it with the `safety-critical` label.
4. For feature requests: describe the use case, not just the feature, so we can evaluate the best solution.

---

## License

WarehouseGPT is proprietary software. By contributing, you agree that your contributions are licensed under the same proprietary license and that you have the right to make the contribution.
