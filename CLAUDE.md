# RewardBench Development Guide

## Package Manager

This project uses **uv** for dependency management. Always use `uv` commands:

```bash
# Install dependencies
uv sync

# Install with generative extra (for LLM-as-judge features)
uv sync --extra generative

# Run commands
uv run python scripts/run_rm.py
uv run rewardbench --help
```

## Version Pinning Policy

**Always pin `transformers` and `vllm` versions** to avoid dependency headaches. These packages have frequent breaking changes and complex dependency trees.

Current pinned versions:
- `transformers==4.57.6`
- `vllm==0.13.0` (in `[generative]` extra)

When updating these versions:
1. Update the pin in `pyproject.toml`
2. Run `uv sync --extra generative` to verify resolution
3. Test the entry points: `uv run rewardbench --help` and `uv run rewardbench-gen --help`
4. Run tests: `uv run pytest`

## Entry Points

- `rewardbench` - Main evaluation CLI (works with base install)
- `rewardbench-gen` - Generative RM evaluation (requires `[generative]` extra)

## Code Quality Checks

Run these before committing:

```bash
# Format code (automatically fixes formatting)
uv run black .
uv run isort .

# Check formatting only (for CI)
uv run black --check .
uv run isort --check-only .

# Lint
uv run flake8 --max-line-length 120 rewardbench/ scripts/
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_data.py
```


## Common Issues

### Import errors for generative modules

If you see `ModuleNotFoundError: No module named 'anthropic'` or similar, you need to install with the generative extra:

```bash
uv sync --extra generative
```

### vLLM platform issues

vLLM has specific platform requirements. On unsupported platforms (like aarch64/ARM), you may need custom wheels. See the main CLAUDE.md in `~/dev/` for DGX Spark-specific instructions.
