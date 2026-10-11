# RewardBench Development Guide

## Package Manager

This project uses **uv** for dependency management. Always use `uv` commands:

```bash
# Install base dependencies
uv sync

# Install with API clients (OpenAI, Anthropic, etc.) for LLM-as-judge
uv sync --extra api

# Install with vLLM for local LLM inference (Linux + CUDA only)
uv sync --extra vllm

# Install everything (api + vllm)
uv sync --extra generative

# Run commands
uv run python scripts/run_rm.py
uv run rewardbench --help
```

## Optional Extras

- `api` - API-based LLM clients (openai, anthropic, google-genai, together) - works on any platform
- `vllm` - Local LLM inference via vLLM - Linux (x86_64/aarch64) + CUDA only, pins torch to 2.13 (CUDA 13.0 wheels, NVIDIA driver >= 580)
- `generative` - Both api + vllm (backwards compatible alias)
- `dev` - Development tools (black, flake8, isort, pytest)

### Legacy v1 scripts (fschat)

`fschat` is unmaintained and has unpatched security advisories, so it is deliberately kept out of `pyproject.toml` and `uv.lock`. It is needed for `rewardbench-gen`, `scripts/run_rm.py`, `run_dpo.py`, `run_bon.py`, and the `--chat_template` flag. Add it to your env by hand:

```bash
uv sync --extra api        # or --extra generative
uv pip install fschat
```

`uv run` keeps it installed, but a plain `uv sync` removes packages that aren't in the lockfile; use `uv sync --inexact` (or reinstall fschat) afterwards. The Docker images install it after their final `uv sync`.

## Version Pinning Policy

**Always pin `transformers` and `vllm` versions** to avoid dependency headaches. These packages have frequent breaking changes and complex dependency trees.

Current pinned versions:
- `transformers==5.17.0` (vllm 0.31 requires `>=5.10.4,<5.18`)
- `vllm==0.31.0` (in `[vllm]` extra)

When updating these versions:
1. Update the pin in `pyproject.toml`
2. Run `uv lock` to update the lock file
3. Test the entry points: `uv run rewardbench --help` and `uv run rewardbench-gen --help`
4. Run tests: `uv run pytest`

## Docker Images

Two Docker images are available:

| Image | Dockerfile | Use Case | Build Time |
|-------|------------|----------|------------|
| `rewardbench` | `Dockerfile` | Reward models, API-based judges | ~5-10 min |
| `rewardbench-vllm` | `Dockerfile.vllm` | Local LLM inference via vLLM | ~45 min |

The base image uses prebuilt flash-attn wheels (torch ≤2.8). The vllm image uses vllm's bundled attention kernels (torch 2.13 / CUDA 13.0 required by vllm, so the host needs NVIDIA driver >= 580).

## Entry Points

- `rewardbench` - Main evaluation CLI (works with base install)
- `rewardbench-gen` - v1 generative RM evaluation (requires `[api]` plus fschat, see below)
- `scripts/run_generative_v2.py` - RewardBench 2 generative RMs (`[api]` for API judges, `[vllm]` for local models)

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

### Running outside Beaker

The run scripts write `/output/metrics.json` for AI2 Beaker by default. Elsewhere this is skipped with a warning; pass `--disable_beaker_save` to silence it.

### DGX Spark (GB10, aarch64 + Blackwell)

On linux aarch64, torch 2.13 and `vllm==0.31.0` both install from their PyPI aarch64 wheels, which are built against CUDA 13.0 and support sm_121 directly (no custom torch index, no `TRITON_PTXAS_PATH` workaround).

- Use `--attn_implementation sdpa` for reward models (no flash-attn wheels on aarch64).
