all: test

lint: FORCE
	uv run ruff check .
	uv run ruff format --check .
	uv run python scripts/update_headers.py --check
	uv run ty check

license: FORCE
	uv run python scripts/update_headers.py

format: license FORCE
	uv run ruff format .
	uv run ruff check --fix .

install: FORCE
	uv sync

test: lint FORCE
	uv run pytest -v

docs: FORCE
	uv run properdocs build

serve-docs: FORCE
	uv run properdocs serve

clean: FORCE
	git clean -dfx -e .venv

FORCE:
