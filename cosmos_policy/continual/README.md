# Online EWC for the Cosmos Policy

`OnlineEWC` (in [`ewc.py`](./ewc.py)) implements the *online* variant of
Elastic Weight Consolidation (Schwarz et al., 2018), wired into the existing
Cosmos Policy training loop. References:

- continual-openpi (JAX) — <https://github.com/Continual-VLAs/continual-openpi/blob/main/src/openpi/continual/ewc.py>
- Lotus (canonical PyTorch EWC) — <https://github.com/UT-Austin-RPL/Lotus/blob/master/lotus/lifelong/algos/ewc.py>

The penalty added to each training step is

```
L_ewc(theta) = (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2
```

After every CL stage we update

```
F      <-  gamma * F + F_new
theta* <-  theta_new
```

so the helper carries `O(|theta|)` extra state across tasks (one Fisher
diagonal + one parameter snapshot, both stored in `bfloat16` by default).

The implementation is FSDP/DTensor aware: per-rank local shards are used
throughout, including for the penalty contribution. No code changes are
required to switch between the existing `policy_ddp` and `policy_fsdp` model
configs.

## Config fields (added to `CosmosPolicyModelConfig`)

| Field | Default | Description |
| --- | --- | --- |
| `ewc_enabled` | `False` | Master switch. When `False`, the entire EWC code path is bypassed. |
| `ewc_lambda` | `50000.0` | Regularisation strength `lambda`. Default mirrors continual-openpi's `TrainConfig.ewc_lambda` for LIBERO-sequential. |
| `ewc_gamma` | `0.9` | Decay applied to the running Fisher. Default mirrors continual-openpi's `TrainConfig.ewc_gamma`. Use `1.0` for canonical EWC (no decay). |
| `ewc_prev_state_path` | `None` | DCP directory (`meta.pt` + `ewc_state/`) or legacy `.pt`. Loaded after the model checkpoint. |
| `ewc_save_state_path` | `None` | If set, the trainer estimates Fisher over `ewc_num_fisher_batches` and writes the new `(F, theta*)` here at the very end of training. |
| `ewc_num_fisher_batches` | `50` | Batches consumed for the Fisher estimate (mirrors continual-openpi's `ewc_max_batches`). Set to `0` to use the full loader. |
| `ewc_fisher_dtype` | `"auto"` | Storage dtype for `F` and `theta*`. `"auto"` matches each parameter's own dtype (so the EWC state inherits the base-stage checkpoint precision exactly). Set explicitly to `"bfloat16"`, `"float16"`, or `"float32"` to force a different storage dtype. |
| `ewc_log_every` | `50` | Log local penalty every N optimiser steps (`0` to silence). |

## Wiring across CL stages

You do **not** need a new experiment config per stage — pass the EWC fields
as Hydra overrides on top of any existing `cl_stage` config. Save paths follow
your own convention; suggested layout:

```
ewc_states/
└── libero_goal/
    ├── base_stage_iter7000/   # DCP: meta.pt + ewc_state/ (after base stage)
    ├── cl_stage1/             # DCP after CL stage 1
    └── cl_stage2/
    └── ...
```

All commands wrap `torchrun` with `uv run` so the right Python environment is
used; this matches the pattern in `LIBERO.md` / `data_generation.sh`.

### Stage 0 — base stage (no penalty, only emit Fisher)

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_goal_base_stage_k_1 \
  model.config.ewc_enabled=true \
  model.config.ewc_lambda=0.0 \
  model.config.ewc_save_state_path=/workspace/ewc_states/libero_goal/base_stage_iter7000
```

`ewc_lambda=0.0` is harmless on the very first stage (there is no
`theta*` yet), but enables the code path so the trainer runs `compute_fisher`
+ `save` at the end of the run.

### Stage 1 — load prior Fisher, train on the new task with the penalty

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage \
  model.config.ewc_enabled=true \
  model.config.ewc_prev_state_path=/workspace/ewc_states/libero_goal/base_stage_iter7000 \
  model.config.ewc_save_state_path=/workspace/ewc_states/libero_goal/cl_stage1
  # ewc_lambda=50000, ewc_gamma=0.9 come from the default; override here if tuning.
```

### Stage 2+ — chain the same way

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage \
  dataloader_train.dataset=<your_stage2_dataset_lazy> \
  model.config.ewc_enabled=true \
  model.config.ewc_prev_state_path=/workspace/ewc_states/libero_goal/cl_stage1 \
  model.config.ewc_save_state_path=/workspace/ewc_states/libero_goal/cl_stage2
```

## Standalone Fisher computation

If you'd rather not couple Fisher computation to the end of the training run
(e.g. you want to recompute `F` on a different data mixture), use
[`scripts/compute_ewc_fisher.py`](../scripts/compute_ewc_fisher.py):

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 \
  -m cosmos_policy.scripts.compute_ewc_fisher \
  --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_k_1 \
  --ckpt /workspace/checkpoints/.../iter_000040000 \
  --output_path /workspace/ewc_states/libero_goal/base_stage_iter7000 \
  --num_batches 50 \
  --fisher_dtype auto
```

**Step 1 should use DCP** (`--format dcp`, default): pass a directory path with no
`.pt` suffix. Legacy `.pt` still works but requires re-sharding across GPU counts.
To convert DCP → `.pt`, see [`export_ewc_state_to_pt.py`](../scripts/export_ewc_state_to_pt.py).

The script internally loads `cosmos_policy/config/config.py` (the same config
router used by `cosmos_policy.scripts.train`) and applies the EWC overrides
itself; you only need to specify the experiment **name** via `--experiment`.

`--ewc_lambda` and `--ewc_gamma` default to `50000` and `0.9` respectively
(matching continual-openpi's `TrainConfig`); they are written to the file as
metadata only and have no effect on the Fisher computation itself. The values
that actually drive the penalty during the next training run are taken from
`model.config.ewc_lambda` / `model.config.ewc_gamma`.

## Calibrating `ewc_lambda` for *your* model

`50000` is the value continual-openpi tuned for π₀ on LIBERO-sequential. The
right value for the Cosmos Policy depends on the magnitude of the Kendall/EDM
loss and the magnitude of the Fisher diagonal at the base-stage checkpoint —
both of which can differ by orders of magnitude from π₀. Rather than guessing,
calibrate λ empirically with
[`scripts/calibrate_ewc_lambda.py`](../scripts/calibrate_ewc_lambda.py):

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 \
  -m cosmos_policy.scripts.calibrate_ewc_lambda \
  --experiment cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage \
  --ckpt /workspace/checkpoints/.../iter_000007000 \
  --ewc_state_path /workspace/ewc_states/libero_goal/base_stage.pt \
  --n_calibration_steps 100
```

The script:

1. Loads the base-stage weights and the EWC state (Fisher + θ*).
2. Sets `ewc_lambda = 1.0` so that the model's `ewc_penalty` term equals the
   raw quadratic `Q = (1/2) Σᵢ Fᵢ (θᵢ − θ*ᵢ)²` per rank.
3. Takes ~100 optimiser steps on the *next* CL stage's training data (this is
   needed because at step 0 `θ = θ*` so `Q = 0` trivially — we need a small
   amount of drift to see a representative `Q`). Updates are discarded.
4. All-reduces `Q` (sum across DP ranks, since the parameters are sharded) and
   `L_task` (mean across DP ranks) per step, averages over the second half
   of the run, and prints recommended `ewc_lambda` values for several target
   ratios `α = L_ewc / L_task`:

```
alpha =  0.01   =>  ewc_lambda = ...
alpha =  0.10   =>  ewc_lambda = ...
alpha =  1.00   =>  ewc_lambda = ...   <-- balanced; usually a good default
alpha = 10.00   =>  ewc_lambda = ...
```

Pick the row whose `α` matches your forgetting/plasticity tradeoff and pass it
as `model.config.ewc_lambda=<value>` to the real CL stage 1 training run.

The math: by construction the script measures the empirical ratio
`R = L_task / Q` during early training. Setting `λ = α · R` gives `λ · Q ≈ α · L_task`,
i.e. the EWC penalty is `α` times the magnitude of the task loss. `α = 1.0`
is the natural balanced choice; smaller values prioritise the new task,
larger values prioritise retention.

## Tuning notes

- `ewc_lambda` is the dominant knob. The default `50000` is taken from
  continual-openpi's LIBERO-sequential setup; sweep `{1e3, 1e4, 5e4, 1e5}` if
  you see either too much forgetting (raise it) or the new task failing to
  learn (lower it).
- `ewc_gamma = 0.9` (default, also from continual-openpi) decays the running
  Fisher per stage. Use `1.0` for canonical EWC (no decay) on short chains
  (≤ 3 stages); `0.9–0.95` is appropriate for longer chains where you want
  very old tasks to gradually fade.
- `ewc_num_fisher_batches` of 50 (default) is usually enough — the diagonal
  Fisher is a coarse Monte-Carlo approximation, and 50 batches × 8 ranks ×
  `batch_size` already covers ~10k samples.
- File size: roughly `2 * |theta| * sizeof(param_dtype)` bytes
  (~4 GB for a 2B model in `bfloat16`, ~8 GB in `float32`). The default
  matches the base-stage parameter dtype.
- Memory at runtime: an extra `2 * |theta|` of GPU memory for `F` and
  `theta*`, sharded by FSDP.
