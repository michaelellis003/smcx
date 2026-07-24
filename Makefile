CLEAN_DIRS := \
	.cache \
	.hypothesis \
	.mypy_cache \
	.nox \
	.pyre \
	.pytest_cache \
	.pytype \
	.ruff_cache \
	.tox \
	build \
	dist \
	htmlcov \
	site \
	wheels
CLEAN_FILES := .coverage coverage.xml
PYTHON_TREES := benchmarks docs scripts src tests

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
	uv run properdocs build --strict

serve-docs: FORCE
	uv run properdocs serve

clean: FORCE
	$(RM) -r $(CLEAN_DIRS)
	$(RM) $(CLEAN_FILES)
	find . -maxdepth 1 -type f -name '.coverage.*' -delete
	find $(PYTHON_TREES) -type d \
		\( -name '__pycache__' -o -name '*.egg-info' \) -prune \
		-exec $(RM) -r {} +
	find $(PYTHON_TREES) -type f \
		\( -name '*.py[codz]' -o -name '*.so' \) -delete

FORCE:
