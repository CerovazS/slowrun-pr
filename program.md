# SlowRun Dynamic QKV Convolution Program

## Active Request

Build on PR #93 (`tiny-fp8-mtp-xsa`, commit `7b9a633`) and test the dynamic short convolution idea from arXiv:2606.03825 on the tiny-track model. The first implementation should add head-wise dynamic causal short convolutions on `q`, `k`, and `v` after projection and before RoPE/QK norm.

Initial variant:
- head-wise dynamic convolution
- kernel width `W=4`
- head size `32` where compatible
- fused residual enabled
- applied to `q`, `k`, and `v`
- controlled by CLI flags so the PR #93 baseline remains runnable from the same file

## Success Criteria

- [x] Preserve the current PR #93 baseline path when dynamic convolution is disabled. Implemented behind `--dynamic-conv-qkv`; default is off.
- [x] Vendor only the required official dynamic-conv head-wise Triton component, with copyright/license header intact. Added `tiny/dynamic_conv1d_headwise.py`.
- [x] Add Q/K/V dynamic convolution before RoPE and before QK norm in `tiny/train.py`. Applied after Q/K/V projection and value residual, before RoPE/QK norm.
- [x] Use head size `32` for `n_embd=1024`; fail clearly if a requested head size does not divide the Q/K/V channel count. Wrapper validates `conv_dim % head_size == 0`.
- [x] Initialize the dynamic path to a static causal short convolution plus residual, matching the paper's head-wise initialization strategy. Dynamic projection is zero-init; static filter uses `U(-1/sqrt(W), 1/sqrt(W))`; residual is enabled.
- [x] Keep dynamic-conv parameters in AdamW/scalar-style optimizer groups with zero weight decay. Dynamic params are excluded from Muon and appended to AdamW at `SCALAR_LR`.
- [x] Run local syntax/import checks: `python -m py_compile tiny/train.py tiny/dynamic_conv1d_headwise.py`. `python3 -m py_compile ...` passed; local import/runtime is blocked by missing `tiktoken`/`triton`.
- [x] Run a small CUDA smoke test if a GPU is available locally; otherwise note that full kernel testing must happen on RunPod. Local CUDA is unavailable; RunPod kernel smoke passed on H100 with `[B=2, T=128, D=1024, H=32, W=4]`.
- [x] Run an 8-GPU trainer smoke with `--dynamic-conv-qkv`. First attempt `dynamic_qkv_smoke_20260608a` completed initial validation but failed at optimizer step because dynamic-conv `static_w` has first dimension `4`, not divisible by `world_size=8`; patched AdamW reduction to all-reduce non-shardable tensors. Retry `dynamic_qkv_smoke_20260608b` completed one training step on 8xH100: final train loss `18.944839`, val loss `10.824631`, peak memory `19188.69 MiB`, result saved to `runs/dynamic_qkv_smoke_20260608b/result.json`.
- [x] Do not run the PR #93 baseline before the dynamic-conv test. No baseline run was launched.
- [ ] Prepare unique run IDs for full QKV dynamic-conv experiments before remote execution.

## Implementation Plan

1. Add `tiny/dynamic_conv1d_headwise.py` from the paper repo.
   - Verify: import compiles locally.

2. Add a small `HeadwiseDynamicShortConvolution` wrapper in `tiny/train.py`.
   - Generate weights from the pre-conv attention input `x`.
   - Apply the official head-wise kernel to projected Q/K/V tensors as `[B, T, D]`.
   - Reshape back to `[B, T, n_head, head_dim]` for attention.
   - Verify: dynamic path disabled is behaviorally unchanged.

3. Add CLI/config controls.
   - `--dynamic-conv-qkv`
   - `--dynamic-conv-width`, default `4`
   - `--dynamic-conv-head-size`, default `32`
   - Verify: run metadata prints the chosen settings.

4. Optimizer integration.
   - Put dynamic weight-generator and static convolution parameters into AdamW with scalar LR and zero weight decay.
   - Keep existing Muon matrix group behavior unchanged for normal projection/MLP matrices.
   - Verify: all dynamic params are excluded from Muon and no trainable param is dropped.

5. Smoke tests.
   - Local `py_compile`.
   - If CUDA is present, instantiate a tiny model and run forward/backward with `--dynamic-conv-qkv`.
   - On RunPod later, run one short precompile/minute-scale test before full tiny experiments.
   - Caveat: the RunPod smoke used PyTorch SDPA fallback because the PR #93 FA3 loader needs `trust_remote_code=True` for `kernels-community/flash-attn3`; fix before timing-sensitive experiments.

## Backlog

- Test `v`-only dynamic conv if QKV is too slow.
- Compare head-wise dynamic conv against static short conv QKV.
- Test low-rank dynamic conv (`R=16`) only if head-wise is promising or too memory-heavy.
- If tiny improves, port the smallest winning variant to the limited track.
