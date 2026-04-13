set shell := ["bash", "-cu"]
set windows-shell := ["powershell", "-NoProfile", "-Command"]

ai-refs:
    #!/usr/bin/env bash
    mkdir -p .ai/refs
    if [ -d .ai/refs/akshare/.git ]; then
        git -C .ai/refs/akshare pull --ff-only
    else
        git clone --depth=1 https://github.com/akfamily/akshare .ai/refs/akshare
    fi

# ------------------------
# Test tasks
# ------------------------

test:
    uv run pytest

test-coverage:
    uv run pytest --cov=crates --cov=src --cov-report=html

test-ci:
    uv run pytest --cov=crates --cov=src --cov-report=term-missing

# ------------------------
# Lint / format / typing
# ------------------------

lint:
    uv run ruff check .

lint-fix:
    uv run ruff check --fix .

format:
    uv run ruff format .

type:
	uv run ty .
