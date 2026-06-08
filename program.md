# SlowRun MLP Vector Gate Program

## Active Request

Build on PR #93 (`tiny-fp8-mtp-xsa`, commit `7b9a633`) and test a baseline-plus-MLP-gate variant.

Clarification:
- The PR #93 MLP already has an internal SwiGLU-style activation gate: `silu(c_gate(x)) * c_fc(x)`.
- This experiment adds a separate learned vector gate on the MLP residual branch output.

Initial variant:
- learned gate shape `[n_layer, n_embd]`
- applied as `x = x + mlp(x_norm) * gate[layer]`
- initialized to `1.0`, so the model starts exactly on the PR #93 baseline function
- optimized with AdamW scalar-style settings and zero weight decay
- controlled by `--mlp-vector-gate`; default off preserves the PR #93 baseline command

## Success Criteria

- [x] Start from clean PR #93 baseline, not the dynamic-conv branch.
- [x] Confirm whether an MLP gate already exists. Internal SwiGLU `c_gate` exists; external residual vector gate does not.
- [x] Add the MLP vector gate behind a CLI flag with baseline default unchanged.
- [x] Initialize the gate to identity (`1.0`) and exclude it from Muon/FLOP matmul counting.
- [x] Run local syntax checks. `python3 -m py_compile tiny/train.py` and `git diff --check` passed.
- [x] Run an 8-GPU smoke test with `--mlp-vector-gate`. `mlp_vector_gate_smoke_20260608c` completed with FA3 active.
- [x] If smoke passes, run full PR #93 default training plus `--mlp-vector-gate`. `mlp_vector_gate_full_20260608a` completed; ckpt avg val loss `3.317193`, total train time `13.86m`.

## Backlog

- Baseline-plus-vector-gate is effectively neutral/slightly worse than PR #93 baseline (`3.316223`), so do not prioritize this exact variant.
- If still exploring gates, try a cheaper per-layer scalar MLP gate before data-dependent gating.
