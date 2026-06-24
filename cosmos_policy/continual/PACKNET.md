# PackNet for the Cosmos Policy

`PackNet` (in [`packnet.py`](./packnet.py)) implements the dynamic-architecture
continual-learning method from Mallya & Lazebnik (2018), following the LIBERO
reference implementation. After each CL stage:

1. **Train** on the new task (only mask==0 capacity is assigned to the task at
   `start_task`; prior tasks stay frozen).
2. **Prune** the lowest `packnet_prune_perc` (default 25%) of current-task weights
   by magnitude.
3. **Post-prune fine-tune** (optional, `packnet_post_prune_iters`) on the pruned
   subnet without re-running `start_task` (zeros in the mask stay 0).
4. **Save** masks for the next stage.

### Starting from a multi-task base checkpoint

When `packnet_init_from_base=True` (and no `packnet_prev_state_path`):

1. All weights are marked as **base task 0** (mask label 1) — already trained.
2. **Prune** ~75% of base weights immediately (lowest magnitude).
3. **Base post-prune retrain** (optional, `packnet_base_post_prune_iters`) on prior-task data.
4. Freed slots (mask 0) are assigned to `packnet_task_id=1` for the first CL task.
5. Training updates **only** those freed slots; base weights (label 1) stay frozen.

Biases and normalization layers are **not** trained (same as LIBERO).

Evaluation metrics should be computed on checkpoints **before** the post-prune
fine-tuning stage (save / eval the main training checkpoint, not the mask file
written at shutdown).

## Config fields (`CosmosPolicyModelConfig`)

| Field | Default | Description |
| --- | --- | --- |
| `packnet_enabled` | `False` | Master switch. Mutually exclusive with `ewc_enabled`. |
| `packnet_task_id` | `0` | PackNet task index for **this** stage (base = 0, first CL after base = 1, …). |
| `packnet_init_from_base` | `False` | If True: loaded ckpt = base task 0; prune at startup; train only freed 25%. |
| `packnet_base_task_id` | `0` | Task index for the pretrained base (usually 0). |
| `packnet_prune_perc` | `0.25` | Fraction of **current-task** weights pruned after training (and at base init). LIBERO uses `0.75`. |
| `packnet_prev_state_path` | `None` | DCP directory (`meta.pt` + `packnet_state/`) or legacy `.pt`. |
| `packnet_save_state_path` | `None` | Post-prune masks (after end-of-stage prune + optional post-prune FT). Use as `packnet_prev_state_path` for the **next** CL stage. |
| `packnet_save_pre_prune_state_path` | `None` | Pre-end-prune masks (matches the last training checkpoint, e.g. `iter_2000`). Defaults to `{packnet_save_state_path}_pre_prune`. **Use for LIBERO eval.** |
| `packnet_base_post_prune_iters` | `0` | After `init_from_pretrained_base`: retrain kept base subnet on prior-task data (`dataloader_packnet_base_retrain`). |
| `packnet_post_prune_iters` | `0` | Extra training iterations after end-of-stage pruning (`0` = skip). |
| `packnet_save_post_prune_ckpt` | `True` | Save model ckpt after base post-prune retrain and after end post-prune FT. |
| `packnet_base_post_prune_ckpt_iter` | `1` | Checkpoint folder for base post-prune retrain (`iter_000000001/`). Default avoids overwriting when `save_iter=1000`. End post-prune saves at `main_iter + packnet_post_prune_iters`. |

## Example: LIBERO CL stage 1

Use the registered experiment (loads 6-task base at iter 7000, prunes 75% at startup,
base post-prune retrain 1000 iters on tasks 0–5, trains task 6 only in freed slots):

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_goal_packnet_cl_stage1
```

Key config fields: `packnet_task_id=1`, `packnet_init_from_base=True`.

Or override on top of any CL stage config:

```bash
  experiment=cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage \
  model.config.packnet_enabled=true \
  model.config.ewc_enabled=false \
  model.config.packnet_task_id=0 \
  model.config.packnet_save_state_path=/workspace/packnet_states/libero_goal/cl_stage0 \
  trainer.max_iter=2000 \
  model.config.packnet_post_prune_iters=2000
```

## Example: CL stage 2 (chain masks)

```bash
  experiment=cosmos_predict2_2b_480p_libero_goal_packnet_cl_stage1 \
  model.config.packnet_task_id=2 \
  model.config.packnet_init_from_base=false \
  model.config.packnet_prev_state_path=/workspace/packnet_states/libero_goal/cl_stage1 \
  model.config.packnet_save_state_path=/workspace/packnet_states/libero_goal/cl_stage2 \
  checkpoint.load_path=<packnet_cl_stage1_iter_2000_ckpt>
```

## Checkpoint layout (DCP)

```
packnet_states/libero_goal/cl_stage0/
    meta.pt           # prune_perc, task_id, param names
    packnet_state/    # per-rank sharded uint8 masks
```

## Inference

For evaluation on task `k`, zero weights where `mask == 0` or `mask > k + 1`:

```python
model.packnet.apply_eval_mask(model.net, eval_task_id=k)
```

Call this before running the policy server or LIBERO eval on a stage-`k` checkpoint.

## Eval after CL stage 1 (tasks 0–5 + task 6)

Use the **iter_2000** checkpoint (before post-prune fine-tuning) with **pre-end-prune**
masks (`packnet_save_pre_prune_state_path`, e.g. `cl_stage1_pre_prune/`). Do **not** use
post-prune masks (`cl_stage1/`) with `iter_2000` — end-of-stage prune zeros ~75% of the
current-task subnet in the mask file, which destroys task-6 performance when applied to
pre-prune weights.

`run_libero_eval` applies the correct subnet per task when `--packnet_mask_path` points
at the pre-prune directory:

- LIBERO tasks **0–5** → `eval_task_id=0` (base subnet)
- LIBERO task **6** → `eval_task_id=1` (base + full CL stage 1 allocation)

```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \
    --ckpt_path /path/to/packnet_cl_stage1/checkpoints/iter_000002000/model.pt \
    --packnet_mask_path /workspace/packnet_states/libero_goal/cl_stage1_pre_prune \
    --packnet_first_cl_libero_task 6 \
    --config_file cosmos_policy/config/config.py \
    --task_suite_name libero_goal \
    --task_ids_to_run "0,1,2,3,4,5,6" \
    --dataset_stats_path /path/to/dataset_statistics.json \
    --t5_text_embeddings_path /workspace/LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl \
    --use_wrist_image True --use_proprio True --normalize_proprio True \
    --unnormalize_actions True --trained_with_image_aug True \
    --chunk_size 16 --num_open_loop_steps 16 --num_trials_per_task 50 \
    --num_denoising_steps_action 5 --deterministic True --seed 195 \
    --use_jpeg_compression True --flip_images True \
    --local_log_dir cosmos_policy/libero-object/logs/ \
    --run_id_note packnet-cl-stage1-eval-all-tasks
```

**Already finished a run without pre-prune masks?** Reconstruct them from the
post-prune state + ``iter_2000`` checkpoint (no retraining):

```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.scripts.export_packnet_pre_prune_masks \
    --post_prune_masks /workspace/packnet_states/libero_goal/cl_stage1 \
    --output /workspace/packnet_states/libero_goal/cl_stage1_pre_prune \
    --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \
    --ckpt /path/to/packnet_cl_stage1/checkpoints/iter_000002000/model.pt
```

Or split eval without reconstruction: task 6 with no masks (~70%); tasks 0–5
with post-prune masks ``cl_stage1/`` (base label 1 unchanged by end prune).

## Inspect saved masks

```bash
uv run --extra cu128 --group libero --python 3.10 \\
  python -m cosmos_policy.scripts.inspect_packnet_state \\
  --packnet_path /workspace/packnet_states/libero_goal/cl_stage1 \\
  --meta_only

# Per-layer label counts (DCP needs model ckpt for shard layout):
uv run --extra cu128 --group libero --python 3.10 \\
  python -m cosmos_policy.scripts.inspect_packnet_state \\
  --packnet_path /workspace/packnet_states/libero_goal/cl_stage1 \\
  --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \\
  --ckpt /path/to/iter_000002000/model.pt
```
