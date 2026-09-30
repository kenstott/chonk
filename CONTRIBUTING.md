# Contributing to Chonk

Bug reports, fixes, new transports/extractors/backends, and documentation
improvements are welcome.

## Reporting bugs and requesting features

Open an issue at <https://github.com/kenstott/chonk/issues>. For a bug, include:

- `chonk-rag` version (`pip show chonk-rag`), Python version, and OS
- the extras installed (for example `chonk-rag[storage,ner]`)
- a minimal script that reproduces the problem, and the full traceback

Security issues: do not open a public issue. Email the maintainer at
kennethstott@gmail.com.

## Development setup

```bash
git clone https://github.com/kenstott/chonk.git
cd chonk
uv sync --all-extras --group dev
uv run python -m spacy download en_core_web_sm
```

## Checks

Every pull request must pass these gates. The pre-commit hooks run the lint,
format, security, and type checks; install them with `uv run pre-commit install`.
CI runs the unit tests and pyright on every push and pull request.

```bash
uv run pytest tests/unit          # unit tests
uv run pytest tests/integration   # integration tests (some need Docker)
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The backend parity suite runs its PostgreSQL lane only when
`CHONK_TEST_PG_DSN` points at a PostgreSQL instance with pgvector.

## Pull requests

1. Branch from `main`.
2. Write a failing test first, then the fix or feature.
3. Keep one logical change per pull request.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
   messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
5. Add an entry under `[Unreleased]` in `CHANGELOG.md`.

### Code rules

- The library is pure Python. No servers, frontends, or web frameworks in `chonk/`.
- No silent fallbacks. A missing value or failed operation raises; it does not
  return a default that masks the failure.
- New public functions are fully type-annotated; `pyright` must pass.
- A new optional dependency goes in an extra in `pyproject.toml`, never in the
  core `dependencies`, and must be MIT-compatible.
- New `VectorBackend` implementations must pass the backend parity suite in
  `tests/unit/test_backend_registry_contract.py`.

## Licensing of contributions

By submitting a pull request you agree that your contribution is licensed under
the project's MIT License.

## Code of conduct

Be respectful and constructive. The maintainer may remove comments, issues, or
pull requests that are abusive or off-topic.
