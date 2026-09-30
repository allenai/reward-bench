# Dependency security

The September 2026 refresh updates the locked ML and API stacks, pins Transformers
5.10.4 and vLLM 0.30.0, and sets security floors for affected transitive packages.
Keep `uv.lock` committed and regenerate it with `uv lock` after dependency changes;
CI checks that the lock agrees with the project manifest. CUDA images use the
locked PyTorch versions instead of installing an older system copy.

Legacy conversation templates now live in `rewardbench.conversation`, adapted
from FastChat 0.2.36 under its Apache-2.0 license. This module retains text prompt
formatting and API-message conversion, with regression snapshots for all 86
original and RewardBench templates. It excludes FastChat's image, network, Gradio
and server helpers. FastChat is no longer an installed dependency. Legacy Gemini
judging uses the same `google-genai` SDK as v2, removing the old SDK's protobuf cap.

## Unresolved upstream issues

Do not interpret an advisory version-range change as proof of remediation.

- **Accelerate — [GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79),
  repository alert #178.** There is no published patched version in the advisory.
  Although 1.15.0 falls outside the advisory's stated `<=1.14.0` range, its
  `load_checkpoint_in_model` implementation still joins shard filenames from the
  checkpoint index without validating containment. Treat untrusted sharded
  checkpoints as unsafe. RewardBench has no direct call to that loader; this is
  not a guarantee about all transitive model-loading paths.
  [Release source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/modeling.py).
- **Setuptools — [GHSA-h35f-9h28-mq5c](https://github.com/advisories/GHSA-h35f-9h28-mq5c),
  repository alert #144.** The fix requires setuptools 83+, but vLLM 0.30.0
  requires `<81` on Python 3.12+. The lock therefore retains 80.9.0. This advisory
  concerns Unicode-normalization collisions in `MANIFEST.in` exclusions when
  building source distributions on affected filesystems. Upgrade once vLLM lifts
  its bound, or validate an upstream-compatible replacement before overriding it.
  [vLLM requirements](https://github.com/vllm-project/vllm/blob/v0.30.0/requirements/common.txt).

Neither issue is dismissed or suppressed by this repository.
