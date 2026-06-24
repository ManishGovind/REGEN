# World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays




## 1. Environment Setup

### Option A: Docker (recommended)

From the project root:

```bash
# Build the image
docker build -t cosmos-policy docker

# Start an interactive container
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v $(pwd):/workspace \
  --gpus all \
  --ipc=host \
  -it --rm \
  -w /workspace \
  --entrypoint bash \
  cosmos-policy
```

### Option B: Local install

```bash
cd cosmos_policy
./install.sh cu128
```

### Install LIBERO dependencies

Inside the container (or after local install):

```bash
uv sync --extra cu128 --group libero --python 3.10
```

### Download LIBERO dataset

```bash
hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir LIBERO-Cosmos-Policy
export BASE_DATASETS_DIR=$(pwd)
```

---

## 2. Continual Learning Pipeline

Tasks are learned **one at a time** across LIBERO suites (e.g. `libero_goal`, `libero_object`, `libero_spatial`). Each suite has 10 tasks; we typically train a **base model on tasks 0–5**, then add tasks 6–9 sequentially.

```
Base (tasks 0–5)  →  CL stage 1 (task 6)  →  CL stage 2 (task 7)  →  ...
```

All training uses `cosmos_policy/scripts/train.py`. Set `--nproc_per_node` to your GPU count.

### Stage 0 — Base training

Train on the first 6 tasks of a suite:

```bash
export BASE_DATASETS_DIR=/path/to/parent/of/LIBERO-Cosmos-Policy

uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_goal_base_stage
```

Checkpoints are saved under `checkpoints/imaginaire4-output/cosmos_policy/cosmos_v2_finetune/<experiment_name>/`.

### Stage k — Continual fine-tuning

Load the previous stage checkpoint and train on the next task. Each CL stage runs for **2000 iterations** by default.

#### Seq-FT (naive sequential fine-tuning)

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage \
  checkpoint.load_path=/path/to/previous_stage/checkpoints/iter_000002000 \
  dataloader_train.dataset=libero_goal_suites_single_task_dataset \
  job.name=my_seqft_cl_stage1
```

#### REGEN (Recurrent Generative Replay)

1. **Generate synthetic rollouts** from the current world action model (see [Section 3](#3-data-generation-for-regen)).
2. **Train** with the generated data in `er_data_dir`:

```bash
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage \
  checkpoint.load_path=/path/to/previous_stage/checkpoints/iter_000002000 \
  dataloader_train.dataset=libero_spatial_suites_cl_stage_task_dataset \
  job.name=my_regen_cl_stage4
```

---

## 3. Data Generation for REGEN

Generate synthetic demonstration rollouts using the world action model's predictions (recurrent generative replay):

```bash
bash data_generation.sh \
  <task_suite_name> \
  <checkpoint_experiment_name> \
  <task_ids_to_run> \
  <dataset_stats_path>
```

**Example** — generate rollouts from the base model on tasks 0–5:

```bash
bash data_generation.sh \
  libero_goal \
  cosmos_predict2_2b_480p_libero_goal_base_stage \
  eval-on-tasks0-5 \
  cl_base_stage_libero_goal_success_only
```

Generated HDF5 files are saved under `LIBERO-Cosmos-Policy/`. Point `er_data_dir` in the dataset config to this directory for REGEN training.

---

## 4. Evaluation

### Standard evaluation (Seq-FT / REGEN)

```bash
bash inference.sh \
  <task_suite_name> \
  <checkpoint_experiment_name> \
  <run_id_note> \
  <dataset_stats_path>
```

**Example:**

```bash
bash inference.sh \
  libero_object \
  cosmos_predict2_2b_480p_libero_object_seqft_cl_stage1 \
  tasks-0-6 \
  LIBERO-Cosmos-Policy/cl_base_stage_libero_object_success_only/dataset_statistics.json
```

Or run the eval script directly:

```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero_single_task_inference_only \
    --ckpt_path /path/to/checkpoints/iter_000002000/model.pt \
    --config_file cosmos_policy/config/config.py \
    --task_suite_name libero_object \
    --task_ids_to_run "0,1,2,3,4,5,6" \
    --dataset_stats_path /path/to/dataset_statistics.json \
    --t5_text_embeddings_path LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --num_trials_per_task 50 \
    --seed 195 \
    --deterministic True \
    --num_denoising_steps_action 5 \
    --use_jpeg_compression True \
    --flip_images True \
    --save_rollout_video True \
    --local_log_dir cosmos_policy/experiments/robot/libero/logs/
```

**Eval notes:**
- Default: 50 trials per task. Use `--num_trials_per_task` to change.
- Seeds `{195, 196, 197}` with `--deterministic True` for reproducibility.
- Logs are written to `cosmos_policy/experiments/robot/libero/logs/`.
- Rollout videos are saved when `--save_rollout_video True`.

---

## 5. Project Structure

```
REGEN/
├── cosmos_policy/
│   ├── config/experiment/cosmos_policy_experiment_configs.py  # Training configs
│   ├── datasets/                   # LIBERO dataset loaders (REGEN support)
│   ├── experiments/robot/libero/   # Evaluation scripts & logs
│   └── scripts/train.py            # Main training entry point
├── data_generation.sh              # REGEN rollout generation
├── inference.sh                    # Standard LIBERO evaluation
└── LIBERO.md                       # Upstream Cosmos Policy LIBERO docs
```

---

## Acknowledgments

Our base **world action model (WAM)** policy is built on [Cosmos Policy](https://github.com/NVlabs/cosmos-policy). Continual learning experiments are conducted on the [LIBERO](https://libero-project.github.io/) simulation benchmark.

---

## Citation

If you find our research useful, please consider citing us:

```bibtex
@article{govind2026world,
  title={World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays},
  author={Govind, Manish},
  year={2026},
  note={Code available at \url{https://github.com/ManishGovind/REGEN}}
}
```
