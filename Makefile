# ParcelVision — common workflows.
# On Windows, run from Git Bash or WSL (`make` required).

COMPOSE = docker compose

.PHONY: up down build logs demo seed test test-docker lint fmt clean

up: .env            ## Start the full stack
	$(COMPOSE) up -d --build

down:               ## Stop the stack
	$(COMPOSE) down

build:              ## Build all images
	$(COMPOSE) build

logs:               ## Tail logs from all services
	$(COMPOSE) logs -f

.env:
	cp .env.example .env

# Demo mode: start db/api/frontend (no worker/ML needed) and load precomputed
# results for a St. Louis AOI so the app is demoable without live inference.
demo: .env
	$(COMPOSE) up -d --build db redis api frontend
	$(COMPOSE) run --rm --no-deps --entrypoint python api scripts/load_seed.py /seed/demo_stl.geojson
	@echo "Demo ready: open http://localhost:3000 and click 'Load demo AOI'"

seed:               ## Reload seed data into a running stack
	$(COMPOSE) run --rm --no-deps --entrypoint python api scripts/load_seed.py /seed/demo_stl.geojson

test:               ## Run tests inside the images (works without local Python)
	$(COMPOSE) build api
	$(COMPOSE) run --rm --no-deps --entrypoint pytest api /app/tests -q
	docker build ./worker --target slim -t parcelvision-worker-slim
	docker run --rm parcelvision-worker-slim pytest /app/tests -q

lint:               ## Ruff + black + mypy inside the api image
	$(COMPOSE) build api
	$(COMPOSE) run --rm --no-deps --entrypoint sh api -c "ruff check /app && black --check /app && mypy /app/app || true"

fmt:                ## Auto-format
	$(COMPOSE) run --rm --no-deps --entrypoint sh api -c "ruff check --fix /app; black /app"

clean: down         ## Stop stack and remove volumes (destroys DB data)
	$(COMPOSE) down -v
