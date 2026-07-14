# smcx

Sequential Monte Carlo for Apple silicon, built on MLX

## Quick start

```bash
git clone https://github.com/michaelellis003/smcx.git
cd smcx
uv sync
```

Install the pre-commit hooks (this only needs to be done once):

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

## Makefile

A `Makefile` collects the common development commands:

| Target | What it does |
|---|---|
| `make test` | Lint, then run pytest |
| `make lint` | Ruff check, format check, license header check, ty |
| `make format` | Add license headers, ruff format, ruff fix |
| `make license` | Add missing SPDX license headers |
| `make docs` | Build documentation |
| `make serve-docs` | Serve documentation locally |
| `make install` | `uv sync` |
| `make clean` | `git clean` (preserves `.venv`) |

## License

Apache 2.0. See [LICENSE](https://github.com/michaelellis003/smcx/blob/main/LICENSE)
for the full text.
