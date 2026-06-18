# Variable-Width X-Profile on Current Tiny Record

## Active Request

Test `> <former`-style variable-width profiles on top of the current `qlabs-eng/slowrun` tiny-track record, not the old PR93 branch.

Base commit target:

```text
origin/main @ 98557a1
tiny record commit d1fbdb7: Merge recurrence and fp8 kernel, 3.307 val loss, 14.1m (#94)
```

Keep the record architecture and training defaults unless explicitly changed:

- `n_layer=16`, `n_embd=1024`, `n_head=8`.
- 2x recurrence with `iteration_schedule=late-transition`.
- final `30%` of training at `num_iterations=2` via `iteration_transition_ratio=0.3`.
- fp8 MTP, full-layer XSA, document shuffling, EMA/SWA/checkpoint averaging unchanged.

The question is whether an X-shaped active width schedule helps the current recurrence/fp8/XSA record, or whether the bottleneck disrupts the recurrent U-Net-style computation.

## Branch

```bash
git fetch origin --prune
git switch codex/pr94-variable-width-xprofile
```

## Implementation

`tiny/train.py` adds:

```text
--variable-width-profile {off,x}
--vw-base-width 1024
--vw-max-width 1280
--vw-bottleneck-width <512|768>
--vw-bottleneck-layer <1-indexed layer>
--vw-residual-mode carry_forward
```

Design constraints:

- embeddings and `lm_head` stay at base width `1024`.
- transformer blocks use per-layer active widths.
- residual stream stays at max width with carry-forward inactive dimensions.
- each layer width is divisible by `n_head * 8 = 64`, so every FA3 head dimension is a multiple of 8.
- Dynamo recompile limits are raised because variable-width intentionally creates more Muon matrix shapes.
- result JSON records `vw_layer_widths` and `vw_residual_width`.

## Experiment Matrix

The previous PR93 plan used 24 layers. The current record uses `L=16`, so the paper sweep ratios map to:

- `r_l=0.50 -> layer 8/16`
- `r_l=0.75 -> layer 12/16`
- `r_l=0.875 -> layer 14/16`

| Run | Run name | Bottleneck ratio | Bottleneck layer | Middle width | Purpose |
| --- | --- | --- | --- | --- | --- |
| 1 | `pr94_vw_x_r075_mid512_recur30_s42` | `r_l=0.75` | `12/16` | `512` | Aggressive paper-default bottleneck. |
| 2 | `pr94_vw_x_r075_mid768_recur30_s42` | `r_l=0.75` | `12/16` | `768` | Softer paper-default bottleneck. |
| 3 | `pr94_vw_x_r050_mid768_recur30_s42` | `r_l=0.50` | `8/16` | `768` | Bottleneck at the U-Net midpoint. |
| 4 | `pr94_vw_x_r0875_mid768_recur30_s42` | `r_l=0.875` | `14/16` | `768` | Late bottleneck after most recurrent mixing. |
| 5 | `pr94_vw_x_r075_mid512_recur30_s42_e16` | `r_l=0.75` | `12/16` | `512` | Diagnostic rerun of run 1 with one extra epoch to test undertraining. |

## Smoke Tests

```bash
uv run python -m py_compile tiny/train.py
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name smoke_pr94_vw_x_r075_mid512_recur30_s42_v3 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 512 \
  --vw-bottleneck-layer 12 \
  --vw-residual-mode carry_forward \
  --max-train-steps 5
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name smoke_pr94_vw_x_r0875_mid768_recur30_s42_v3 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 14 \
  --vw-residual-mode carry_forward \
  --max-train-steps 5
```

Smoke checks:

- no shape mismatch in U-Net skips, recurrence re-injection, VE projections, XSA, MTP, or checkpoint averaging.
- `runs/<run-name>/result.json` includes `vw_layer_widths`.
- peak memory fits at default `device_batch_size=32`.

## Full Runs

Run each command into a unique run directory and preserve the full terminal logs under `runs/<run-name>/terminal.log`.

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr94_vw_x_r075_mid512_recur30_s42 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 512 \
  --vw-bottleneck-layer 12 \
  --vw-residual-mode carry_forward
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr94_vw_x_r075_mid768_recur30_s42 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 12 \
  --vw-residual-mode carry_forward
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr94_vw_x_r050_mid768_recur30_s42 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 8 \
  --vw-residual-mode carry_forward
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr94_vw_x_r0875_mid768_recur30_s42 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 14 \
  --vw-residual-mode carry_forward
```

Diagnostic extra-epoch run. This is not primarily a tiny-record candidate if wall time exceeds 15 minutes; use it to test whether the aggressive `mid512` bottleneck is undertrained rather than intrinsically weak.

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr94_vw_x_r075_mid512_recur30_s42_e16 \
  --num-epochs 16 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 512 \
  --vw-bottleneck-layer 12 \
  --vw-residual-mode carry_forward
```

## Success Criteria

- `win`: best/checkpoint-avg val loss `< 3.307` and training time `<= 15.00m`.
- `promising`: best/checkpoint-avg val loss within `0.005` of `3.307` with a clear width-location trend.
- `reject_time`: training time `> 15.50m`.
- `reject_quality`: best/checkpoint-avg val loss worse than `3.327`.
- `diagnostic_extra_epoch`: compare quality to run 1, but do not treat as tiny-eligible if wall time exceeds 15 minutes.
- `failed`: OOM, NaN, shape mismatch, or nonzero exit.

## Next Experiment: Prefix-Wide Baseline

The X-profile sweep did not produce a tiny-eligible win: the best quality came from
`r050_mid768`, but it effectively bought quality with extra compute because the residual
stream widened to `1280`.

Next test a dedicated `prefix_wide` profile instead of another X-profile:

- keep the base/residual stream at `1024` throughout the model.
- expand only the first two transformer blocks internally to `1280`.
- project each widened block output back to `1024` before returning to the residual stream.
- keep layers 3-16 at the record baseline width `1024`.
- keep PR94 recurrence: final `30%` at `2x`, fp8 MTP, XSA, document shuffling, and all training defaults unchanged.

Target run:

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr94_prefix_wide_l02_1280_recur30_s42 \
  --variable-width-profile prefix_wide \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-prefix-layers 2 \
  --vw-residual-mode project_back
```

Hypothesis:

- The first two widened blocks may improve early feature extraction without imposing a global `1280` residual stream.
- Compute should stay close to the PR94 baseline, plausibly within the unused roughly one-minute wall-time margin.
- A win requires best/checkpoint-avg val loss `< 3.307` while staying near or below `15.00m` training time.

Outcome:

- `pr94_prefix_wide_l02_1280_recur30_s42_fa3_cached` completed with `best_val_loss=3.320786`, `train_time=15.15m`, `wall_time=17.49m`.
- This is a `reject_quality` and borderline `reject_time`: widening two full blocks internally to `1280` was materially more expensive than expected (`337.7M` parameters) and did not recover enough quality.

## Checklist

- [x] Create branch from current `origin/main`.
- [x] Port variable-width implementation onto PR94/current-record code.
- [x] Update this program for `L=16` and PR94 recurrence.
- [x] Verify `uv run python -m py_compile tiny/train.py`.
- [x] Run both 8xH100 smoke tests.
- [x] Run the four full experiments.
- [x] Build summary JSON/report from `runs/*/result.json`.
- [x] Interpret results relative to the current 2x recurrence and fp8 MTP record.
- [x] Implement dedicated `prefix_wide` profile with base/residual stream fixed at `1024`.
- [x] Smoke test `pr94_prefix_wide_l02_1280_recur30_s42`.
- [x] Run full `pr94_prefix_wide_l02_1280_recur30_s42_fa3_cached` and compare against PR94 record: `reject_quality`, borderline `reject_time`.
- [ ] Run diagnostic `pr94_vw_x_r075_mid512_recur30_s42_e16` only if still useful after `prefix_wide`.
