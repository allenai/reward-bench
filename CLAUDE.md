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
- `vllm` - Local LLM inference via vLLM - Linux + CUDA only, pins torch to 2.9
- `generative` - Both api + vllm (backwards compatible alias)
- `v1` - Legacy dependencies (fschat, trl) for v1 scripts
- `dev` - Development tools (black, flake8, isort, pytest)

## Version Pinning Policy

**Always pin `transformers` and `vllm` versions** to avoid dependency headaches. These packages have frequent breaking changes and complex dependency trees.

Current pinned versions:
- `transformers==4.57.6`
- `vllm==0.13.0` (in `[vllm]` extra)

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

The base image uses prebuilt flash-attn wheels (torch ≤2.8). The vllm image builds flash-attn from source (torch 2.9 required by vllm).

## Entry Points

- `rewardbench` - Main evaluation CLI (works with base install)
- `rewardbench-gen` - v1 generative RM evaluation (requires `[api]` + `[v1]` extras; it uses fschat conversation templates)
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

On linux aarch64, torch comes from the PyTorch cu129 index (see `[tool.uv.sources]`), and `vllm==0.13.0` installs from its PyPI aarch64 wheel. Both are built against CUDA 12.x. Do not switch torch to cu130: vllm's kernels link `libcudart.so.12`, so `import vllm` then fails.

- `torch` warns that sm_121 is above its supported max (sm_120); this is harmless.
- Use `--attn_implementation sdpa` for reward models (no flash-attn wheels on aarch64).
- Triton's bundled `ptxas` does not know `sm_121a`, so vLLM CUDA graph capture / torch.compile fails. Either point Triton at the system CUDA 13 ptxas or run vLLM in eager mode:
  ```bash
  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  ```
- If an existing venv previously had cu130 torch, force the torch wheels to swap (the aarch64 torchvision/torchaudio filenames carry no CUDA tag, so uv keeps the stale build):
  ```bash
  uv sync --extra generative --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
  ```
