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
- [ ] Run local syntax checks.
- [ ] Run an 8-GPU smoke test with `--mlp-vector-gate`.
- [ ] If smoke passes, run full PR #93 default training plus `--mlp-vector-gate`.

## Backlog

- If vector gate helps but is slow/unstable, try per-layer scalar MLP gate.
- If identity vector gate is too weak, test a data-dependent MLP output gate from a small prefix of the residual stream.
