# Slowrun Affine Reparametrization Program

## Active Request

Explore affine reparametrization for looped/recursive Slowrun tiny-track models, starting from the current tiny SOTA on `main` and the RLM idea from PR #92. Test both tokenwise input-dependent affine re-injection and global input-dependent affine re-injection. A scalar baseline rerun was added after explicit user request for direct comparison. The latest affine-only rerun uses the hyperparameters from the PR #92 attached submission log, without rerunning baseline.

Execution target: use the active RunPod pod `runpod-slowrun` with 8xH100 80GB at `/workspace/slowrun`. The environment and data are expected to already be installed there. Run the tiny-track experiments even if they exceed the official 15-minute cap; first objective is to determine whether the ideas improve validation loss, then optimize efficiency.

## Working Answer

Affine reparametrization does not have to be input-dependent in the abstract. It can be a static learned affine transform, for example per-layer learned scale/shift/gate. However, if the goal is to replace or improve `x0` re-injection in a looped transformer, the scientifically relevant version should be input-dependent: the affine terms should be generated from `x0` or from a summary of `x0`.

Do not represent input-dependent affine terms as direct learned parameters of shape `[T, D]`. That would be position-dependent but not content-dependent. Use functions of `x0`:

- Tokenwise affine: `x0: [B, T, D] -> scale, shift, gate: [B, T, D]`.
- Global affine: pool or summarize `x0: [B, T, D] -> c: [B, D] -> scale, shift, gate: [B, 1, D]`.

The affine terms should be computed once from the original embedding anchor `x0` and reused across recurrent loop iterations unless an experiment explicitly tests iteration-dependent modulation.

## Success Criteria

- [x] Confirm whether RLM/PR #92 code is present on `origin/main`; PR #92 is not on `main`, so affine work used the PR branch as the experiment base.
- [x] Run scalar recurrent baseline after explicit user request; baseline completed on RunPod.
- [x] Implement tokenwise input-dependent affine re-injection with zero-init/gate-zero identity behavior.
- [x] Implement global input-dependent affine re-injection with zero-init/gate-zero identity behavior.
- [x] Run smoke tests for compile/import compatibility; `py_compile`, `git diff --check`, and remote precompile completed.
- [x] Run the two affine experiments under unique run IDs; tokenwise and global completed on RunPod with both the initial settings and the PR #92 log settings.
- [x] Compare best/EMA/checkpoint-avg validation loss, wall time, step time, peak memory, and early stability.
- [x] Decide whether tokenwise or global affine is worth promoting to tuning or combining with RLM schedule changes; global affine with PR #92 settings narrowly beats the earlier scalar rerun on best validation loss, while tokenwise affine is clearly worse.

## Implementation Plan

1. Establish the branch base.
   - Verify `main` contains XSA tiny SOTA.
   - If RLM is not in `main`, create an experiment branch from `main` and add the minimal looped forward: `_run_network_once`, scheduled loop count, and BPTT=1.
   - Verify: `python -m py_compile tiny/train.py` and a small single-process forward/backward smoke test.

2. Add an affine mode flag.
   - Suggested CLI: `--reinject-mode scalar|tokenwise_affine|global_affine|static_affine`.
   - Keep `scalar` equivalent to current `resid_lambdas/x0_lambdas`.
   - Put affine parameters in AdamW/scalar optimizer groups with zero weight decay.
   - Verify: parameter grouping prints and optimizer step with no `grad=None` failure.

3. Tokenwise affine re-injection.
   - Compute `terms_i = MLP_i(norm(x0))`, chunked into `scale, shift, gate`, each `[B, T, D]`.
   - Bound scale with a small multiplier, e.g. `scale = 0.1 * tanh(scale)`.
   - Bound gate with `gate = tanh(gate)`.
   - Zero-init the last projection so step 0 is identity.
   - Apply before each block, replacing the raw `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` path for this mode.
   - Verify: no change at init beyond numerical noise if gate is zero.

4. Global affine re-injection.
   - Build a causal-safe or whole-sequence summary from `x0`. For first experiment, use mean pooling over token dimension because training already sees full context; note this is not autoregressive-safe for generation, but validation loss uses teacher-forced full sequences.
   - Produce `scale, shift, gate` as `[B, 1, D]` and broadcast across tokens.
   - Reuse the same bounded/zero-init structure as tokenwise.
   - Verify: lower memory and step-time overhead than tokenwise.

5. Experiment queue.
   - `affine_tiny_scalar_baseline_20260608j`: scalar baseline. Completed; best val loss 3.5268.
   - `affine_tiny_tokenwise_20260608i`: tokenwise affine, same schedule. Completed; best val loss 3.6211.
   - `affine_tiny_global_20260608i`: global affine, same schedule. Completed; best val loss 3.5543.
   - `affine_pr92_tokenwise_20260608l`: tokenwise affine with PR #92 log hyperparameters. Completed; best val loss 3.8999, EMA/final val loss 3.9387, 2858 steps.
   - `affine_pr92_global_20260608l`: global affine with PR #92 log hyperparameters. Completed; best/EMA val loss 3.5113, checkpoint-average val loss 3.6073, 2850 steps, 19.62m wall time.
   - Optional control: `affine_tiny_static_20260608a`, static learned per-layer affine not generated from input.

6. Remote execution.
   - Sync the experiment branch to `/workspace/slowrun` on `runpod-slowrun`.
   - Launch runs in `tmux` with dedicated logs under unique output directories.
   - Babysit runs until completion, checking GPU utilization, logs, and result JSON after every run.
   - Ignore the 15-minute cap for this phase; do not stop a scientifically useful run solely because it exceeds the leaderboard limit.
   - For `20260608l`, `/workspace` filled during the first tokenwise attempt, so final run directories were symlinked from `/workspace/slowrun/runs/<run_id>` to `/root/slowrun_runs/<run_id>` and logs were written under `/root/slowrun_logs/`.

## Backlog

- Add a scalar multiplier sweep for affine gate max: `0.05`, `0.1`, `0.2`.
- Test whether affine terms should be shared across layers, per-layer, or generated once then projected per-layer.
- Test whether affine terms should be reused across loops or depend on iteration index with a small learned iteration embedding.
- If affine improves RLM, test with Anderson/fixed-point forward from PAN-65 as a separate follow-up.
- If tokenwise affine helps but is too slow, test low-rank or bottleneck affine generators.

## Run Hygiene

Every run must write to a fresh directory. Do not reuse output directories. Preserve stdout logs, config, metrics JSON, checkpoint-average result, final summary, and timing stats for each run.
