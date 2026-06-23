# =============================================================================
# WarehouseGPT — Makefile
# =============================================================================
# Usage:
#   make setup              — start infrastructure and initialize project
#   make help               — list all targets with descriptions
#
# All targets assume the working directory is the repo root.
# Set WAREHOUSE_ID, SITE_CONFIG, etc. as needed for deployment targets.
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Python / Hatch
PYTHON := python3
HATCH := hatch

# Docker Compose
COMPOSE := docker compose
COMPOSE_FILE := docker-compose.yml

# Kubernetes
KUBECTL := kubectl
HELM := helm
K8S_NAMESPACE := warehousegpt
K8S_CHART_DIR := infra/k8s/charts/warehousegpt

# CUDA / GPU
CUDA_VISIBLE_DEVICES ?= 0

# Training defaults (override on CLI)
WORLD_MODEL_CONFIG ?= world_model/training/config.yaml
SAFETY_TRAINING_CONFIG ?= safety_ai/training/config.yaml
RL_TRAINING_CONFIG ?= forklift_rl/training/config.yaml

# Deployment defaults
SITE_CONFIG ?= isaac_sim/configs/warehouse_default.yaml
WAREHOUSE_ID ?= wh-001

# Colors for terminal output
BOLD := \033[1m
RESET := \033[0m
GREEN := \033[32m
YELLOW := \033[33m
CYAN := \033[36m

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "$(BOLD)WarehouseGPT — Developer Makefile$(RESET)"
	@echo ""
	@echo "$(CYAN)Infrastructure:$(RESET)"
	@grep -E '^(setup|infra-up|infra-down|infra-restart|infra-status|infra-logs|infra-reset):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Training:$(RESET)"
	@grep -E '^(train-world-model|train-safety|train-forklift-rl|train-all):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Running Services:$(RESET)"
	@grep -E '^(run-digital-twin|run-agent|run-api|run-fleet|run-all-services):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Inference:$(RESET)"
	@grep -E '^(build-trt|build-trt-all|triton-status|triton-reload):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Simulation:$(RESET)"
	@grep -E '^(gen-data|gen-safety-data|run-isaac-sim|validate-scene):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Deployment:$(RESET)"
	@grep -E '^(deploy-k8s|undeploy-k8s|deploy-status|rollback-k8s):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Quality:$(RESET)"
	@grep -E '^(test|test-unit|test-integration|test-sim|lint|fmt|typecheck|coverage):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Database:$(RESET)"
	@grep -E '^(db-migrate|db-seed|db-reset|db-backup):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(CYAN)Utilities:$(RESET)"
	@grep -E '^(clean|prune|env-check|version):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
.PHONY: setup
setup: env-check infra-up db-seed ## Full first-time setup: start infra, init DBs, seed data
	@echo "$(GREEN)Setup complete. API: http://localhost:8000 | Grafana: http://localhost:3000$(RESET)"

.PHONY: infra-up
infra-up: ## Start all Docker Compose services and wait for health checks
	@echo "$(CYAN)Starting WarehouseGPT infrastructure...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) up -d
	@echo "$(CYAN)Waiting for services to become healthy (up to 120s)...$(RESET)"
	@timeout 120 bash -c '\
		until $$($(COMPOSE) ps --format json | python3 -c \
			"import sys,json; data=sys.stdin.read(); \
			 services=[json.loads(l) for l in data.strip().split(\"\\n\") if l]; \
			 print(all(s.get(\"Health\",\"healthy\")==\"healthy\" for s in services if \"Health\" in s))" \
			2>/dev/null | grep -q True); do \
			echo "  Still waiting..."; sleep 5; \
		done' || (echo "$(YELLOW)Warning: some services may not be healthy. Check: make infra-status$(RESET)")
	@echo "$(GREEN)Infrastructure is up.$(RESET)"

.PHONY: infra-down
infra-down: ## Stop all Docker Compose services (preserves volumes)
	$(COMPOSE) -f $(COMPOSE_FILE) down

.PHONY: infra-restart
infra-restart: ## Restart all Docker Compose services
	$(COMPOSE) -f $(COMPOSE_FILE) restart

.PHONY: infra-status
infra-status: ## Show status of all services
	$(COMPOSE) -f $(COMPOSE_FILE) ps

.PHONY: infra-logs
infra-logs: ## Tail logs from all services (Ctrl+C to stop)
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f

.PHONY: infra-reset
infra-reset: ## DESTRUCTIVE: stop services and delete all data volumes
	@echo "$(YELLOW)WARNING: This will delete all data. Press Ctrl+C to cancel, Enter to continue.$(RESET)"
	@read _confirm
	$(COMPOSE) -f $(COMPOSE_FILE) down -v
	@echo "$(GREEN)All volumes deleted.$(RESET)"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
.PHONY: train-world-model
train-world-model: ## Train the VQ-VAE + Transformer world model
	@echo "$(CYAN)Starting world model training (config: $(WORLD_MODEL_CONFIG))...$(RESET)"
	@echo "$(YELLOW)Estimated duration: 7 days on 4x A100 80GB. Use CTRL+C to stop; training checkpoints automatically.$(RESET)"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(HATCH) run train:run \
		--config $(WORLD_MODEL_CONFIG) \
		--stage world_model \
		$(EXTRA_ARGS)

.PHONY: train-vqvae
train-vqvae: ## Train only the VQ-VAE tokenizer stage (prerequisite for train-world-model)
	@echo "$(CYAN)Starting VQ-VAE tokenizer training...$(RESET)"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(HATCH) run train:run \
		--config $(WORLD_MODEL_CONFIG) \
		--stage vqvae \
		$(EXTRA_ARGS)

.PHONY: train-safety
train-safety: ## Train all safety AI detectors (fire, near-miss, PPE, collision, zone)
	@echo "$(CYAN)Starting safety AI training pipeline...$(RESET)"
	@echo "$(YELLOW)Estimated duration: 24 hours per detector on 1x A100.$(RESET)"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(PYTHON) -m safety_ai.training.pipeline \
		--config $(SAFETY_TRAINING_CONFIG) \
		--detectors fire near_miss worker_safety collision zone_violation \
		$(EXTRA_ARGS)

.PHONY: train-safety-fire
train-safety-fire: ## Train only the fire/smoke detector
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(PYTHON) -m safety_ai.training.pipeline \
		--config $(SAFETY_TRAINING_CONFIG) \
		--detectors fire \
		$(EXTRA_ARGS)

.PHONY: train-safety-nearmiss
train-safety-nearmiss: ## Train only the near-miss detector
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(PYTHON) -m safety_ai.training.pipeline \
		--config $(SAFETY_TRAINING_CONFIG) \
		--detectors near_miss \
		$(EXTRA_ARGS)

.PHONY: train-forklift-rl
train-forklift-rl: ## Train forklift RL navigation policy (PPO, requires Isaac Sim)
	@echo "$(CYAN)Starting forklift RL training (Isaac Sim required)...$(RESET)"
	@echo "$(YELLOW)Estimated duration: 48 hours for single-agent navigation on 1x A100.$(RESET)"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(PYTHON) -m forklift_rl.training.trainer \
		--config $(RL_TRAINING_CONFIG) \
		--algorithm ppo \
		--env navigation \
		$(EXTRA_ARGS)

.PHONY: train-forklift-marl
train-forklift-marl: ## Train multi-agent forklift coordination policy (MAPPO)
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(PYTHON) -m forklift_rl.training.trainer \
		--config $(RL_TRAINING_CONFIG) \
		--algorithm mappo \
		--env multi_agent \
		--num-agents 3 \
		$(EXTRA_ARGS)

.PHONY: train-all
train-all: train-vqvae train-world-model train-safety ## Run full training pipeline: VQ-VAE → world model → safety detectors
	@echo "$(GREEN)All training complete.$(RESET)"

# ---------------------------------------------------------------------------
# Running Services (development)
# ---------------------------------------------------------------------------
.PHONY: run-api
run-api: ## Start the FastAPI gateway with hot-reload
	$(HATCH) run api:serve

.PHONY: run-agent
run-agent: ## Start the WarehouseGPT Claude agent service
	$(HATCH) run agent:serve

.PHONY: run-digital-twin
run-digital-twin: ## Start the digital twin sync service (requires Isaac Sim)
	@echo "$(CYAN)Starting Digital Twin sync for warehouse: $(WAREHOUSE_ID)$(RESET)"
	WAREHOUSE_ID=$(WAREHOUSE_ID) \
		$(PYTHON) -m warehousegpt.apps.digital_twin_sync.main \
		--site-config $(SITE_CONFIG)

.PHONY: run-fleet
run-fleet: ## Start the fleet manager service
	$(PYTHON) -m warehousegpt.apps.fleet_manager.main \
		--warehouse-id $(WAREHOUSE_ID)

.PHONY: run-all-services
run-all-services: ## Start all application services in background (for local full-stack dev)
	@echo "$(CYAN)Starting all application services...$(RESET)"
	$(HATCH) run api:serve &
	$(HATCH) run agent:serve &
	$(PYTHON) -m warehousegpt.apps.fleet_manager.main &
	@echo "$(GREEN)All services started. Use 'pkill -f warehousegpt' to stop.$(RESET)"

# ---------------------------------------------------------------------------
# Inference / TensorRT
# ---------------------------------------------------------------------------
.PHONY: build-trt
build-trt: ## Build TensorRT engine for a specific model (MODEL=detection|pose|grasp|seg)
	@[ -n "$(MODEL)" ] || (echo "$(YELLOW)Usage: make build-trt MODEL=detection$(RESET)" && exit 1)
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(HATCH) run infer:build-engines --models $(MODEL) --fp16

.PHONY: build-trt-all
build-trt-all: ## Build TensorRT engines for all models (detection, pose, grasp, segmentation)
	@echo "$(CYAN)Building TensorRT engines for all models (requires A100 or A10G)...$(RESET)"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(HATCH) run infer:build-engines \
		--models detection pose_estimation grasp_prediction segmentation safety_fire safety_nearmiss \
		--fp16 --int8 --calibration-data data/calibration/

.PHONY: triton-status
triton-status: ## Check Triton Inference Server model repository status
	@curl -sf http://localhost:8000/v2/models | python3 -m json.tool || \
		echo "$(YELLOW)Triton not reachable. Check: docker compose ps triton-server$(RESET)"

.PHONY: triton-reload
triton-reload: ## Reload Triton model repository without restarting the server
	@curl -sf -X POST http://localhost:8000/v2/repository/index | python3 -m json.tool
	@echo "$(GREEN)Triton model repository reloaded.$(RESET)"

# ---------------------------------------------------------------------------
# Simulation / Synthetic Data
# ---------------------------------------------------------------------------
.PHONY: gen-data
gen-data: ## Generate synthetic warehouse dataset (requires Isaac Sim)
	@echo "$(CYAN)Generating synthetic dataset (config: $(SITE_CONFIG))...$(RESET)"
	$(HATCH) run sim:gen-data \
		--scene-config $(SITE_CONFIG) \
		--output-dir data/synthetic/$(shell date +%Y%m%d_%H%M%S) \
		--num-frames ${NUM_FRAMES:-100000} \
		--randomize true \
		$(EXTRA_ARGS)

.PHONY: gen-safety-data
gen-safety-data: ## Generate safety-specific dataset (fire, near-miss, PPE violation events)
	@echo "$(CYAN)Generating safety scenario dataset...$(RESET)"
	$(HATCH) run sim:gen-data \
		--scene-config $(SITE_CONFIG) \
		--output-dir data/synthetic/safety_$(shell date +%Y%m%d_%H%M%S) \
		--num-frames ${NUM_FRAMES:-50000} \
		--scenario-types fire near_miss zone_violation ppe_violation collision \
		--event-rate 0.05 \
		$(EXTRA_ARGS)

.PHONY: run-isaac-sim
run-isaac-sim: ## Launch Isaac Sim in headless mode with the default warehouse scene
	@echo "$(CYAN)Launching Isaac Sim (headless)...$(RESET)"
	$(PYTHON) isaac_sim/warehouse_generator.py \
		--config $(SITE_CONFIG) \
		--headless \
		$(EXTRA_ARGS)

.PHONY: validate-scene
validate-scene: ## Validate an Isaac Sim scene config (checks assets, camera placements, physics)
	$(PYTHON) isaac_sim/warehouse_generator.py \
		--config $(SITE_CONFIG) \
		--validate-only

# ---------------------------------------------------------------------------
# Kubernetes Deployment
# ---------------------------------------------------------------------------
.PHONY: deploy-k8s
deploy-k8s: ## Deploy WarehouseGPT to Kubernetes using Helm
	@echo "$(CYAN)Deploying to Kubernetes namespace: $(K8S_NAMESPACE)...$(RESET)"
	$(KUBECTL) create namespace $(K8S_NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(HELM) upgrade --install warehousegpt $(K8S_CHART_DIR) \
		--namespace $(K8S_NAMESPACE) \
		--values $(K8S_CHART_DIR)/values.yaml \
		--values $(K8S_CHART_DIR)/values.production.yaml \
		--set image.tag=$(shell git rev-parse --short HEAD) \
		--set warehouseId=$(WAREHOUSE_ID) \
		--wait --timeout 300s
	@echo "$(GREEN)Deployment complete.$(RESET)"
	$(MAKE) deploy-status

.PHONY: deploy-k8s-staging
deploy-k8s-staging: ## Deploy to Kubernetes staging environment
	@echo "$(CYAN)Deploying to Kubernetes staging...$(RESET)"
	$(HELM) upgrade --install warehousegpt-staging $(K8S_CHART_DIR) \
		--namespace $(K8S_NAMESPACE)-staging \
		--create-namespace \
		--values $(K8S_CHART_DIR)/values.yaml \
		--values $(K8S_CHART_DIR)/values.staging.yaml \
		--set image.tag=$(shell git rev-parse --short HEAD) \
		--wait --timeout 300s
	@echo "$(GREEN)Staging deployment complete.$(RESET)"

.PHONY: undeploy-k8s
undeploy-k8s: ## Remove WarehouseGPT from Kubernetes (preserves PVCs)
	@echo "$(YELLOW)WARNING: This will remove the application but preserve data volumes.$(RESET)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(HELM) uninstall warehousegpt --namespace $(K8S_NAMESPACE)

.PHONY: deploy-status
deploy-status: ## Show status of Kubernetes deployment
	@echo "$(CYAN)Kubernetes deployment status:$(RESET)"
	$(KUBECTL) get pods,svc,hpa -n $(K8S_NAMESPACE)

.PHONY: rollback-k8s
rollback-k8s: ## Rollback Kubernetes deployment to previous release
	@echo "$(YELLOW)Rolling back to previous Helm release...$(RESET)"
	$(HELM) rollback warehousegpt --namespace $(K8S_NAMESPACE)
	@echo "$(GREEN)Rollback complete.$(RESET)"
	$(MAKE) deploy-status

.PHONY: k8s-logs
k8s-logs: ## Stream logs from Kubernetes pods (APP=api|agent|fleet|twin)
	@[ -n "$(APP)" ] || (echo "$(YELLOW)Usage: make k8s-logs APP=api$(RESET)" && exit 1)
	$(KUBECTL) logs -f -l app=warehousegpt-$(APP) -n $(K8S_NAMESPACE) --max-log-requests 5

# ---------------------------------------------------------------------------
# Quality: Tests, Lint, Type Check
# ---------------------------------------------------------------------------
.PHONY: test
test: test-unit test-integration ## Run unit and integration tests
	@echo "$(GREEN)All tests passed.$(RESET)"

.PHONY: test-unit
test-unit: ## Run unit tests (no external services required)
	$(HATCH) run test:unit tests/

.PHONY: test-integration
test-integration: ## Run integration tests (requires running Docker Compose stack)
	@echo "$(CYAN)Checking Docker Compose stack health before integration tests...$(RESET)"
	@$(COMPOSE) ps | grep -q "healthy" || (echo "$(YELLOW)Stack not healthy. Run: make infra-up$(RESET)" && exit 1)
	$(HATCH) run test:integration

.PHONY: test-sim
test-sim: ## Run simulation regression tests (requires Isaac Sim + GPU)
	$(HATCH) run test:simulation

.PHONY: test-safety
test-safety: ## Run safety detector unit tests with recall assertions
	$(HATCH) run test:unit tests/unit/test_safety/ -v

.PHONY: test-all
test-all: ## Run all tests including simulation tests
	$(HATCH) run test:all

.PHONY: coverage
coverage: ## Run unit tests with HTML coverage report
	$(HATCH) run test:unit --cov=warehousegpt --cov-report=html:htmlcov --cov-report=term
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

.PHONY: lint
lint: ## Run Ruff linter and Mypy type checker
	@echo "$(CYAN)Running Ruff linter...$(RESET)"
	$(HATCH) run default:lint
	@echo "$(GREEN)Lint passed.$(RESET)"

.PHONY: fmt
fmt: ## Auto-format code with Ruff formatter
	$(HATCH) run default:fmt
	@echo "$(GREEN)Code formatted.$(RESET)"

.PHONY: typecheck
typecheck: ## Run Mypy type checker only
	$(HATCH) run mypy warehousegpt/

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
.PHONY: db-migrate
db-migrate: ## Run Alembic database migrations
	$(HATCH) run alembic upgrade head

.PHONY: db-migrate-create
db-migrate-create: ## Create a new Alembic migration (MSG="description of change")
	@[ -n "$(MSG)" ] || (echo "$(YELLOW)Usage: make db-migrate-create MSG='add forklift telemetry table'$(RESET)" && exit 1)
	$(HATCH) run alembic revision --autogenerate -m "$(MSG)"

.PHONY: db-seed
db-seed: ## Seed knowledge graph (Neo4j) and vector store (ChromaDB) with default data
	@echo "$(CYAN)Seeding knowledge graph...$(RESET)"
	$(PYTHON) scripts/seed_knowledge_graph.py --warehouse-id $(WAREHOUSE_ID)
	@echo "$(GREEN)Knowledge graph seeded.$(RESET)"

.PHONY: db-reset
db-reset: ## DESTRUCTIVE: drop and recreate all database tables, re-run migrations
	@echo "$(YELLOW)WARNING: This will destroy all database data.$(RESET)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(HATCH) run alembic downgrade base
	$(HATCH) run alembic upgrade head
	$(MAKE) db-seed
	@echo "$(GREEN)Database reset complete.$(RESET)"

.PHONY: db-backup
db-backup: ## Backup PostgreSQL database to ./backups/
	@mkdir -p backups
	@BACKUP_FILE="backups/warehousegpt_$(shell date +%Y%m%d_%H%M%S).sql.gz"; \
		$(COMPOSE) exec -T postgres \
		pg_dump -U $${POSTGRES_USER:-warehousegpt} $${POSTGRES_DB:-warehousegpt} | \
		gzip > "$$BACKUP_FILE"; \
		echo "$(GREEN)Backup saved to $$BACKUP_FILE$(RESET)"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
.PHONY: env-check
env-check: ## Verify required environment variables and tools are present
	@echo "$(CYAN)Checking prerequisites...$(RESET)"
	@command -v docker >/dev/null 2>&1 || (echo "$(YELLOW)ERROR: docker not found$(RESET)" && exit 1)
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "$(YELLOW)ERROR: python3 not found$(RESET)" && exit 1)
	@command -v $(HATCH) >/dev/null 2>&1 || (echo "$(YELLOW)ERROR: hatch not found. Install: pip install hatch$(RESET)" && exit 1)
	@[ -f .env ] || (echo "$(YELLOW)WARNING: .env file not found. Copy .env.example and fill in values.$(RESET)")
	@if [ -f .env ]; then \
		grep -q "ANTHROPIC_API_KEY" .env && \
		! grep -q "ANTHROPIC_API_KEY=$$" .env || \
		echo "$(YELLOW)WARNING: ANTHROPIC_API_KEY is empty in .env$(RESET)"; \
	fi
	@nvidia-smi >/dev/null 2>&1 && echo "$(GREEN)GPU: $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)$(RESET)" || \
		echo "$(YELLOW)WARNING: No NVIDIA GPU detected. Training and Triton will not work.$(RESET)"
	@$(PYTHON) --version
	@docker --version
	@echo "$(GREEN)Prerequisites check complete.$(RESET)"

.PHONY: version
version: ## Show current version and git commit
	@echo "WarehouseGPT version: $$($(PYTHON) -c 'import tomllib; print(tomllib.load(open(\"pyproject.toml\",\"rb\"))[\"project\"][\"version\"])')"
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
	@echo "Git branch: $$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
	@$(PYTHON) --version
	@docker --version
	@$(HATCH) --version

.PHONY: clean
clean: ## Remove Python cache files, build artifacts, and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	@echo "$(GREEN)Cache and build artifacts cleaned.$(RESET)"

.PHONY: prune
prune: ## Remove unused Docker images and volumes (frees disk space)
	@echo "$(YELLOW)Pruning unused Docker resources...$(RESET)"
	docker image prune -f
	docker volume prune -f
	docker builder prune -f
	@echo "$(GREEN)Docker pruned.$(RESET)"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally with MkDocs
	@command -v mkdocs >/dev/null 2>&1 || pip install mkdocs mkdocs-material
	mkdocs serve

# ---------------------------------------------------------------------------
# CI helpers (used by GitHub Actions)
# ---------------------------------------------------------------------------
.PHONY: ci-lint
ci-lint: ## Lint check for CI (no auto-fix)
	$(HATCH) run ruff check .
	$(HATCH) run mypy warehousegpt/

.PHONY: ci-test
ci-test: ## Run unit tests for CI
	$(HATCH) run test:unit --cov=warehousegpt --cov-report=xml

.PHONY: ci-build-images
ci-build-images: ## Build all Docker images for CI validation
	docker build -f docker/Dockerfile.api -t warehousegpt/api-gateway:ci .
	docker build -f docker/Dockerfile.agent -t warehousegpt/agent:ci .
	docker build -f docker/Dockerfile.perception -t warehousegpt/perception:ci .
