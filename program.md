# Variable-Width X-Profile Looped Transformer Sweep

## Active Request

Prepare the next 8xH100 pod run for variable-width `> <former`-style profiles on top of the latest SlowRun looped-transformer PR.

Keep the current PR93 looped-transformer behavior:

- U-NET decoder loop enabled only in the last `30%` of training.
- `loop_mode=unet_decoder`
- `loop_start=8`
- `loop_end=13`
- `loop_repeat_count=2`
- `loop_schedule=late-transition`
- `loop_transition_ratio=0.3`
- `loop_bptt_mode=bptt1`
- scalar x0 re-injection active.
- no x0 MLP conditioning.

The scientific question is whether an X-shaped layer-width profile complements or conflicts with the current looped-transformer signal. The paper's initial sweep parameterizes bottleneck location as `l* = r_l L` and bottleneck width as `d* = r_d d`; its default recipe uses `r_l = 0.75` and `r_d = 0.3`. For SlowRun tiny we keep the user-requested practical widths and test the paper's location ratios around the loop window.

## Branch And Target

Local implementation branch:

```bash
codex/pr93-variable-width-xprofile
```

Remote execution target, after the pod is active:

```bash
ssh -tt -i ~/.ssh/id_ed25519 <pod-user>@<pod-host>
cd /workspace/slowrun
```

The implementation branch should start from the latest PR93 loop-sweep code, not from an older loop implementation:

```bash
git fetch --all --prune
git switch codex/pr93-mid-layer-loop-sweep
git pull --ff-only || true
```

Then create or switch to the experiment branch on the pod:

```bash
git switch -c codex/pr93-variable-width-xprofile || git switch codex/pr93-variable-width-xprofile
```

## Required Implementation Before Runs

Add a minimal variable-width mode to `tiny/train.py` without changing the PR93 loop path when the mode is disabled.

Expected CLI surface:

```text
--variable-width-profile {off,x}
--vw-base-width 1024
--vw-max-width 1280
--vw-bottleneck-width <512|768>
--vw-bottleneck-layer <1-indexed layer>
--vw-residual-mode carry_forward
```

Implementation constraints:

- Keep embeddings and `lm_head` at base width `1024` unless a smoke test proves the wider head is necessary.
- Use internal per-layer widths for attention and MLP blocks.
- Use a fixed residual stream with carry-forward dimensions, following the paper's parameter-free expansion idea.
- Keep `n_head=8`; each layer width must remain divisible by `8`.
- Do not alter document shuffling, SWA, EMA, XSA, or the PR93 loop schedule.
- Write the resolved per-layer widths into each run JSON.

## Experiment Matrix

All runs use:

```text
base d = 1024
early/final d = 1280
n_layer = 24
n_head = 8
seed = 42
loop active in final 30% of training
```

Layer numbering below is 1-indexed. If the implementation uses 0-indexed internals, subtract 1.

| Run | Run name | Bottleneck ratio | Bottleneck layer | Middle width | Purpose |
| --- | --- | --- | --- | --- | --- |
| 1 | `pr93_vw_x_r075_mid512_loop30_s42` | `r_l=0.75` | `18/24` | `512` | Aggressive bottleneck at the paper-default location. |
| 2 | `pr93_vw_x_r075_mid768_loop30_s42` | `r_l=0.75` | `18/24` | `768` | Softer bottleneck at the paper-default location. |
| 3 | `pr93_vw_x_r050_mid768_loop30_s42` | `r_l=0.50` | `12/24` | `768` | Move the bottleneck into the middle/loop-adjacent region from the paper sweep. |
| 4 | `pr93_vw_x_r0875_mid768_loop30_s42` | `r_l=0.875` | `21/24` | `768` | Move the bottleneck late, after the main looped-decoder range. |

## Smoke Tests

Run these before any full training:

```bash
python -m py_compile tiny/train.py

torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name smoke_pr93_vw_x_r075_mid512_loop30_s42 \
  --n_layer 24 \
  --n_embd 1024 \
  --n_head 8 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 512 \
  --vw-bottleneck-layer 18 \
  --vw-residual-mode carry_forward \
  --loop-mode unet_decoder \
  --loop-start 8 \
  --loop-end 13 \
  --loop-repeat-count 2 \
  --loop-schedule late-transition \
  --loop-transition-ratio 0.3 \
  --loop-bptt-mode bptt1 \
  --max-train-steps 5

torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name smoke_pr93_vw_x_r0875_mid768_loop30_s42 \
  --n_layer 24 \
  --n_embd 1024 \
  --n_head 8 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 21 \
  --vw-residual-mode carry_forward \
  --loop-mode unet_decoder \
  --loop-start 8 \
  --loop-end 13 \
  --loop-repeat-count 2 \
  --loop-schedule late-transition \
  --loop-transition-ratio 0.3 \
  --loop-bptt-mode bptt1 \
  --max-train-steps 5
```

Smoke-test checks:

- no shape mismatch in U-NET skips, x0 re-injection, XSA, VE projections, MTP, or checkpoint averaging.
- no compile explosion from heterogeneous layer widths.
- peak memory fits at `device_batch_size=32`.
- result JSON includes the final width schedule.

## Full Run Commands

Run each command in its own log file. Do not reuse run directories.

```bash
mkdir -p outputs/pr93_variable_width_xprofile/logs \
  outputs/pr93_variable_width_xprofile/metrics \
  outputs/pr93_variable_width_xprofile/reports
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr93_vw_x_r075_mid512_loop30_s42 \
  --n_layer 24 --n_embd 1024 --n_head 8 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 512 \
  --vw-bottleneck-layer 18 \
  --vw-residual-mode carry_forward \
  --loop-mode unet_decoder \
  --loop-start 8 --loop-end 13 \
  --loop-repeat-count 2 \
  --loop-schedule late-transition \
  --loop-transition-ratio 0.3 \
  --loop-bptt-mode bptt1 \
  2>&1 | tee outputs/pr93_variable_width_xprofile/logs/pr93_vw_x_r075_mid512_loop30_s42.log
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr93_vw_x_r075_mid768_loop30_s42 \
  --n_layer 24 --n_embd 1024 --n_head 8 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 18 \
  --vw-residual-mode carry_forward \
  --loop-mode unet_decoder \
  --loop-start 8 --loop-end 13 \
  --loop-repeat-count 2 \
  --loop-schedule late-transition \
  --loop-transition-ratio 0.3 \
  --loop-bptt-mode bptt1 \
  2>&1 | tee outputs/pr93_variable_width_xprofile/logs/pr93_vw_x_r075_mid768_loop30_s42.log
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr93_vw_x_r050_mid768_loop30_s42 \
  --n_layer 24 --n_embd 1024 --n_head 8 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 12 \
  --vw-residual-mode carry_forward \
  --loop-mode unet_decoder \
  --loop-start 8 --loop-end 13 \
  --loop-repeat-count 2 \
  --loop-schedule late-transition \
  --loop-transition-ratio 0.3 \
  --loop-bptt-mode bptt1 \
  2>&1 | tee outputs/pr93_variable_width_xprofile/logs/pr93_vw_x_r050_mid768_loop30_s42.log
```

```bash
torchrun --standalone --nproc_per_node=8 tiny/train.py \
  --run-name pr93_vw_x_r0875_mid768_loop30_s42 \
  --n_layer 24 --n_embd 1024 --n_head 8 \
  --variable-width-profile x \
  --vw-base-width 1024 \
  --vw-max-width 1280 \
  --vw-bottleneck-width 768 \
  --vw-bottleneck-layer 21 \
  --vw-residual-mode carry_forward \
  --loop-mode unet_decoder \
  --loop-start 8 --loop-end 13 \
  --loop-repeat-count 2 \
  --loop-schedule late-transition \
  --loop-transition-ratio 0.3 \
  --loop-bptt-mode bptt1 \
  2>&1 | tee outputs/pr93_variable_width_xprofile/logs/pr93_vw_x_r0875_mid768_loop30_s42.log
```

## Success Criteria

Primary comparisons:

- PR93 control selected/checkpoint-avg val loss: `3.3158069409822164`.
- Best completed prior U-NET loop candidate: `pr93_unet_decoder_8_13_bptt1_x0_s42`, selected/checkpoint-avg val loss `3.3145322297748767`, training time `14.55m`.

Outcome labels:

- `win`: selected/checkpoint-avg val loss `< 3.3145322297748767` and training time `<= 15.00m`.
- `promising`: selected/checkpoint-avg val loss `< 3.3158069409822164` and training time `<= 15.00m`.
- `architecture_signal`: loss is worse than PR93 control by `<= 0.01`, but width-location trend is monotonic or interpretable.
- `reject_time`: training time `> 15.50m`.
- `reject_quality`: selected/checkpoint-avg val loss worse than PR93 control by `> 0.02`.
- `failed`: OOM, NaN, shape mismatch, or nonzero exit at `device_batch_size=32`.

## Interpretation Plan

After the four runs, compare:

- `mid512` vs `mid768` at `r_l=0.75`: does stronger bottleneck regularization help or starve the looped decoder?
- `r_l=0.50` vs `0.75` vs `0.875` at `mid768`: does the bottleneck need to align with, precede, or follow the looped region?
- width schedule vs runtime: does the theoretical average-width saving survive actual H100 kernels and PR93 loop overhead?
- training loss vs validation loss: does the bottleneck reduce overfitting in the fixed-data regime?

Expected qualitative outcomes:

- If `r_l=0.50` wins, the bottleneck likely helps by compressing before or during the repeated mid-decoder computation.
- If `r_l=0.75` wins, the paper's late bottleneck transfers despite the PR93 loop.
- If `r_l=0.875` wins, late narrowing may act as post-loop regularization.
- If all variable-width runs lose but are stable, the next test should isolate variable width without the loop.

## Artifacts

For every run, preserve:

- full terminal log under `outputs/pr93_variable_width_xprofile/logs/`.
- run JSON under `outputs/pr93_variable_width_xprofile/metrics/`.
- resolved width schedule in the JSON and copied into the final report.
- compact summary JSON: `outputs/pr93_variable_width_xprofile/metrics/variable_width_xprofile_summary.json`.
- Markdown report: `outputs/pr93_variable_width_xprofile/reports/variable_width_xprofile_report.md`.

## Checklist

- [ ] Activate 8xH100 pod and clone/sync the repo.
- [ ] Start from latest PR93 loop-sweep code.
- [ ] Create remote branch `codex/pr93-variable-width-xprofile`.
- [ ] Implement minimal variable-width X profile in `tiny/train.py`.
- [ ] Verify `python -m py_compile tiny/train.py`.
- [ ] Run both smoke tests successfully.
- [ ] Run `pr93_vw_x_r075_mid512_loop30_s42`.
- [ ] Run `pr93_vw_x_r075_mid768_loop30_s42`.
- [ ] Run `pr93_vw_x_r050_mid768_loop30_s42`.
- [ ] Run `pr93_vw_x_r0875_mid768_loop30_s42`.
- [ ] Build summary JSON/report.
- [ ] Interpret results relative to the looped-transformer mechanism.
