# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.

Adapted from: https://github.com/user/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py

Parallel Inference:
    To enable parallel inference across multiple GPUs, use:
        --use_parallel_inference True
        --available_gpus "0,1,2,3"
        --num_queries_best_of_n 4

    This will run model queries in parallel across the specified GPUs using torch.multiprocessing, which can
    significantly speed up evaluation when using value functions that require multiple queries per action.

    Requirements:
    - Multiple GPUs must be available
    - CUDA must be properly configured
    - Sufficient GPU memory for multiple model copies

    Note: Uses torch.multiprocessing with 'spawn' start method for CUDA compatibility.

Task-Level Parallelism Across GPUs:
    To run different LIBERO tasks in parallel, one task per GPU (task 0 -> GPU 0, task 1 -> GPU 1, ...), use:
        --parallel_tasks_across_gpus True
        --available_gpus "0,1,2,3,4,5,6,7"

    Tasks are round-robin assigned to the listed GPUs. Each GPU gets its own worker process that loads its
    own model copy and runs its assigned task IDs sequentially. ``num_trials_per_task`` runs within each
    task still execute sequentially inside the worker.

    Mutually exclusive with ``--use_parallel_inference``.

    Requirements:
    - One full model copy fits in each listed GPU's memory.
    - LIBERO MuJoCo envs can be created per process (each worker constructs its own task suite + env).

Usage examples:
    # *** Main checkpoint: 98.5% success rate ***
    #   Replace `task_suite_name` with one of {libero_spatial, libero_object, libero_goal, libero_10}
    #   Replace `seed` with one of {195, 196, 197}
    #   Replace `run_id_note` with a unique identifier for the run
    uv run -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name libero_10 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0,1,2,3,4,5,6,7" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note chkpt45000--5stepAct--seed195--deterministic \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1
"""

import base64
import glob
import io
import json
import logging
import os
import math
import re
from socket import PACKET_FASTROUTE
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import draccus
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import wandb
from libero.libero import benchmark
from PIL import Image, ImageDraw
from cosmos_policy.config.experiment.libero_task_constants import LIBERO_SUITE_TASK_ID_TO_DESCRIPTION

from cosmos_policy.experiments.robot.cosmos_utils import (
    WorkerPoolManager,
    apply_packnet_eval_for_libero_task,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_policy.datasets.dataset_utils import (
    decode_jpeg_bytes_dataset,
    resize_images,
)
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: action, 5: future proprio, 6: future wrist img, 7: future primary img, 8: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 3
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 5, 7


BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")
# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "libero"                                                # Evaluation suite name

    #################################################################################################################
    # Cosmos Policy-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = ""                                                     # Inference config name
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_policy/config/config.py"  # Cosmos default config file path

    # PackNet continual-learning eval (apply per-task subnet before rollouts)
    packnet_mask_path: str = ""                                          # Pre-end-prune masks (packnet_save_pre_prune_state_path) for iter_N ckpt eval
    packnet_first_cl_libero_task: int = 6                                # First CL task index (libero_goal task 6); tasks below use base subnet only

    use_third_person_image: bool = True                                  # Whether to include primary (third-person) image in input
    num_third_person_images: int = 1                                     # Number of third-person images to include in input (LIBERO: 1 agentview image)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 1                                            # Number of wrist images to include in input (LIBERO: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = True                                             # Whether to flip images vertically across x-axis
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = True                                    # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 5                                  # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    dataset_stats_path: str = ""                                         # Path to dataset statistics file for action unnormalization and proprio normalization
    t5_text_embeddings_path: str = ""                                    # Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 16                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 16                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode
    deterministic_reset: bool = False                                    # Whether to run in deterministic reset mode (sets global random seed right before env reset)
    deterministic_reset_seed: int = None                                 # (Only applicable if deterministic_reset==True) The seed to set before deterministic reset; if not provided, defaults to the base seed

    #################################################################################################################
    # Planning model and best-of-N search parameters
    #################################################################################################################
    use_ensemble_future_state_predictions: bool = False                  # Whether to use ensemble of future state predictions
    num_future_state_predictions_in_ensemble: int = 3                    # Number of future state predictions in ensemble
    future_state_ensemble_aggregation_scheme: str = "average"            # How to aggregate future state predictions in an ensemble of future state predictions (options: "average", "first")
    use_ensemble_value_predictions: bool = False                         # Whether to use ensemble of value predictions
    num_value_predictions_in_ensemble: int = 5                           # Number of value predictions in ensemble
    value_ensemble_aggregation_scheme: str = "average"                   # How to aggregate values in an ensemble of value predictions (options: "average", "gamma_weighted_average", "lcb", "success_vote", "majority_mean")
    search_depth: int = 1                                                # Number of levels to search through in the best-of-N search tree
    mask_current_state_action_for_value_prediction: bool = False         # Whether to use input masking to mask out certain inputs (current state and action) during value prediction
    mask_future_state_for_qvalue_prediction: bool = False                # Whether to use input masking to mask out certain inputs (future state) during Q(s, a) value prediction

    num_queries_best_of_n: int = 1                                       # Number of queries to make to the model (this is the N in best-of-N search)
    use_parallel_inference: bool = False                                 # Whether to use parallel inference across multiple GPUs
    available_gpus: str = "0,1,2,3,4,5,6,7"                              # Comma-separated list of GPU IDs available for use for parallel inference (defaults to all 8 GPUs on a node)
    parallel_timeout: int = 15                                           # Timeout in seconds for each parallel query

    # Fan out tasks across GPUs (task 0 -> GPU 0, task 1 -> GPU 1, round-robin if more tasks than GPUs).
    # Each worker process loads its own model on its assigned GPU and runs its assigned task_ids sequentially.
    # Mutually exclusive with `use_parallel_inference` (which parallelizes best-of-N queries instead).
    parallel_tasks_across_gpus: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL                      # Task suite (must be one of: LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_10, LIBERO_90)
    num_trials_per_task: int = 50                                 # Number of rollouts per task
    task_ids_to_run: str = ""                                             # Optional comma-separated task IDs to run (e.g. "0,1,2,5"); empty means run all tasks
    initial_states_path: str = "DEFAULT"                                 # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256       
                                            # Resolution for rendering environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"                            # Local directory for eval logs
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging

    use_wandb: bool = False                                              # Whether to also log results in Weights & Biases
    wandb_entity: str = "YOUR_ENTITY"                                    # Name of WandB entity
    wandb_project: str = "YOUR_PROJECT"                                  # Name of WandB project

    seed: int = 7                                                        # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    data_collection: bool = False                                        # If True, save episodic data for later offline use
    jpeg_compress: bool = True                                           # If True, apply JPEG compression to images before saving
    reward_stop_threshold: Optional[float] = None                        # If set, reward-window early-stop is enabled in live env rollouts
    reward_stop_consecutive_steps: int = 3                               # Reward window length for early-stop rule
    use_gpt_reward: bool = False                                         # If True, query GPT vision reward using current observation image + task text
    gpt_reward_model: str = "gpt-5.4-mini"                                # OpenAI model id for vision reward
    gpt_reward_stop_threshold: Optional[float] = 4                  # If set, stop early in HDF5-observation rollout when GPT score >= threshold
    gpt_reward_timeout_s: float = 30.0  
                                     # Timeout per GPT reward HTTP request
    save_rollout_video: bool = False                                      # Whether to save rollout video
    #################################################################################################################
    # Cross-task / HDF5 observation policy inputs (optional; no simulator stepping in that mode)
    #################################################################################################################
    #: If set, :func:`run_task` uses :func:`run_episode_with_hdf5_observations`: policy inputs from this LIBERO HDF5 only (offline roll; ``env`` is not stepped).
    eval_hdf5_path: str = ""
    #: Demo key under ``data/`` in ``eval_hdf5_path`` (e.g. ``demo_0``). Ignored when ``eval_hdf5_path`` is empty.
    eval_hdf5_demo_key: str = "demo_0"
    replay_actions: bool = False
    #: HDF5 file or directory of ``.hdf5`` / ``.h5`` for :func:`run_episode_with_predicted_actions` (sorted; cycled by ``episode_idx``). Empty → use ``eval_hdf5_path``.
    replay_action_path_dir: str = ""
    replay_hdf5_path: str = ""
    data_generation: bool = False                                      # If True, data generation is enabled


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_config(cfg: PolicyEvalConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.ckpt_path is not None, "ckpt_path must not be None!"

    assert not (cfg.parallel_tasks_across_gpus and cfg.use_parallel_inference), (
        "`parallel_tasks_across_gpus` and `use_parallel_inference` are mutually exclusive: "
        "the former pins one task per GPU process, the latter parallelizes best-of-N queries."
    )

    if "image_aug" in str(cfg.ckpt_path):
        assert cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==True` because model was trained with image augmentations!"
        )

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    if cfg.eval_hdf5_path:
        p = os.path.abspath(os.path.expanduser(cfg.eval_hdf5_path))
        assert os.path.isfile(p), f"eval_hdf5_path does not exist or is not a file: {p}"
        assert cfg.eval_hdf5_demo_key, "eval_hdf5_demo_key must be non-empty when eval_hdf5_path is set"

    if cfg.replay_actions and cfg.replay_action_path_dir:
        rp = os.path.abspath(os.path.expanduser(cfg.replay_action_path_dir))
        assert os.path.isfile(rp) or os.path.isdir(rp), f"replay_action_path_dir must be a file or directory: {rp}"
        if os.path.isdir(rp):
            assert any(
                x.endswith(".hdf5") or x.endswith(".h5") for x in os.listdir(rp)
            ), f"replay_action_path_dir has no .hdf5/.h5 files: {rp}"

    if cfg.task_ids_to_run:
        parsed = _parse_task_ids_to_run(cfg.task_ids_to_run)
        assert len(parsed) > 0, "task_ids_to_run parsed to empty list"
    assert cfg.reward_stop_consecutive_steps >= 1, "reward_stop_consecutive_steps must be >= 1"
    if cfg.use_gpt_reward:
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set when use_gpt_reward=True"
    if cfg.gpt_reward_stop_threshold is not None:
        assert 1.0 <= cfg.gpt_reward_stop_threshold <= 5.0, "gpt_reward_stop_threshold must be in [1, 5]"


def _parse_task_ids_to_run(task_ids_to_run: str) -> list[int]:
    """Parse a comma-separated task-id list like '0,1,2,5'."""
    tokens = [x.strip() for x in task_ids_to_run.split(",") if x.strip()]
    if not tokens:
        return []
    return sorted({int(x) for x in tokens})


def _extract_score_from_model_content(content: str) -> float:
    """Extract a numeric score from model output text/json."""
    txt = content.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").replace("json\n", "", 1).strip()
    try:
        obj = json.loads(txt)
        print(obj)
        if isinstance(obj, dict) and "score" in obj:
            return int(obj["score"])
    except Exception:
        pass
    # Fallback: parse as plain float text
    return int(txt)



def _get_gpt_reward_score(cfg: PolicyEvalConfig, observation: dict, task_description: str) -> float:
    """Query GPT vision model with current observation and task instruction, returning score in [0, 1]."""
    # LIBERO / HDF5 ``agentview_rgb`` is RGB, HxWx3 uint8 — same path as ``gpt_video_reward_score`` after BGR→RGB.
    image = np.asarray(observation["primary_image"], dtype=np.uint8)
    pil_img = Image.fromarray(image, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    # system = (
    #     "You are a strict vision-based reward model for robot manipulation. "
    #     "Given one RGB image and a natural-language task, estimate task completion. "
    #     'Reply with a single JSON object only, no markdown, exactly one key: "score" '
    #     "(float from 0.0 to 1.0; use 1.0 only if the goal appears fully satisfied)."
    # )

    system = (
    "You are a strict vision-based reward model for robot manipulation. "
    "Given one RGB image and a task description, judge the final progress towards the task.\n"
    "Score on this rubric:\n"
    "1 - No goal-relevant change\n"
    "2 - Minimal progress\n"
    "4 - Near completion: missing one minor requirement\n"
    "5 - Perfect: Task is completed successfully\n"
    'Reply with JSON only: {"score": N} where N is an integer 1-5.'
    )

    body = {
        "model": cfg.gpt_reward_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task instruction: {task_description}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}", "detail": "low"
                    }},
                ],
            },
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.gpt_reward_timeout_s)) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GPT reward HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e

    content = payload["choices"][0]["message"]["content"]
    score = _extract_score_from_model_content(content)
    return max(1.0, min(5.0, float(score)))


def _list_replay_action_hdf5_paths(cfg: PolicyEvalConfig) -> list[str]:
    """Return sorted replay-action HDF5 paths from a file or directory (recursive)."""
    if not cfg.replay_action_path_dir:
        return []
    p = os.path.abspath(os.path.expanduser(cfg.replay_action_path_dir))
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        raise FileNotFoundError(f"replay_action_path_dir is not a file or directory: {p}")

    paths = []
    for root, _dirs, files in os.walk(p):
        for name in files:
            if name.endswith(".hdf5") or name.endswith(".h5"):
                paths.append(os.path.join(root, name))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files (.hdf5/.h5) in replay_action_path_dir: {p}")
    return paths


def _task_description_from_hdf5_filename(hdf5_path: str) -> str:
    """Infer task description from '<task>_demo*.hdf5' filename."""
    base = os.path.splitext(os.path.basename(hdf5_path))[0]
    task_slug = re.sub(r"_demo\d+$", "", base, flags=re.IGNORECASE)
    return task_slug.replace("_", " ").strip()





def check_unnorm_key(cfg: PolicyEvalConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in Cosmos Policy `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def load_initial_states(cfg: PolicyEvalConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size, flip_images: bool = False):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs, flip_images)
    wrist_img = get_libero_wrist_image(obs, flip_images)

    # Prepare observations dict
    observation = {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }

    return observation  # Return processed observation


def _observation_from_future_prediction_pack(
    future_pack: dict,
    future_proprio,
    *,
    resize_size: int,
    flip_images: bool,
) -> dict:
    """Build policy ``observation`` from one model future head (physical proprio, uint8 images)."""
    prim = np.asarray(future_pack["future_image"], dtype=np.uint8)
    wrist = future_pack.get("future_wrist_image")
    if wrist is not None:
        wrist = np.asarray(wrist, dtype=np.uint8)
    if flip_images:
        prim = np.flipud(prim)
        if wrist is not None:
            wrist = np.flipud(wrist)
    prim = np.asarray(Image.fromarray(prim).resize((resize_size, resize_size)), dtype=np.uint8)
    if wrist is not None:
        wrist = np.asarray(Image.fromarray(wrist).resize((resize_size, resize_size)), dtype=np.uint8)
    if future_proprio is None:
        raise ValueError("future_proprio is required")
    prop = np.asarray(future_proprio, dtype=np.float64).reshape(-1)
    return {
        "primary_image": np.copy(prim),
        "wrist_image": None if wrist is None else np.copy(wrist),
        "proprio": prop.copy(),
    }


def _libero_env_flat_state(env) -> np.ndarray:
    """Flattened MuJoCo sim state (matches LIBERO ``states`` in ``regenerate_libero_dataset``)."""
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64)


def _expand_future_images_to_timesteps(
    fut: np.ndarray,
    T: int,
    num_open_loop_steps: int,
) -> np.ndarray:
    """Expand per-chunk future images ``(n_chunks, H, W, C)`` to dense ``(T, H, W, C)`` (same as eval video indexing)."""
    fut = np.asarray(fut, dtype=np.uint8)
    if fut.shape[0] == T:
        return fut
    n_chunks = int(fut.shape[0])
    if n_chunks == 0:
        raise ValueError("future image array has zero chunks")
    if fut.shape[0] > T:
        raise ValueError(f"future length {fut.shape[0]} exceeds T={T}")
    h, w, c = int(fut.shape[1]), int(fut.shape[2]), int(fut.shape[3])
    out = np.zeros((T, h, w, c), dtype=np.uint8)
    n_ols = max(1, int(num_open_loop_steps))
    for t in range(T):
        ci = min(t // n_ols, n_chunks - 1)
        out[t] = fut[ci]
    return out


def _resize_single_frame_to_hw(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize one HxWxC uint8 frame to ``(h, w)`` (no-op if already that size). Uses LANCZOS for downscaling."""
    frame = np.asarray(frame, dtype=np.uint8)
    if int(frame.shape[0]) == h and int(frame.shape[1]) == w:
        return frame


    return np.asarray(Image.fromarray(frame).resize((w, h), Image.LANCZOS), dtype=np.uint8)


def _draw_timestep_label_on_pil_image(img: Image.Image, t: int) -> None:
    """Draw ``t=<index>`` at the top-left (for side-by-side query debug PNGs)."""
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), f"t={t}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))


def _resize_all_frames_to_hw(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize every frame in ``(T,H,W,C)`` to ``(h, w)`` (no-op per frame when already that size)."""
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.shape[1] == h and frames.shape[2] == w:
        return frames
    out = np.zeros((frames.shape[0], h, w, frames.shape[3]), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = _resize_single_frame_to_hw(frames[i], h, w)
    return out




def save_episode_collected_data_libero_hdf5(
    ep_filepath: str,
    cfg: PolicyEvalConfig,
    collected_data: dict,
    task_description: str,
    episode_success: bool,
    *,
    demo_key: str = "demo_0",
) -> None:
    """Write ``collected_data`` in LIBERO training HDF5 layout under ``data/<demo_key>/``.

    Expects ``primary_images``, ``wrist_images``, ``proprio``, ``actions``; optional ``states`` (sim flatten).
    Optional ``future_primary_images`` / ``future_wrist_images`` (per open-loop chunk or already length ``T``)
    are written under ``obs/`` as ``policy_future_agentview_rgb*`` / ``policy_future_eye_in_hand_rgb*``.
    Ignores non-array keys such as ``success``.
    """
    primary = collected_data["primary_images"]
    wrist = collected_data["wrist_images"]
    proprio = collected_data["proprio"]
    if "future_proprio" in collected_data:
        future_proprio = collected_data["future_proprio"]
    else:
        future_proprio = None
    actions = np.asarray(collected_data["actions"], dtype=np.float32)
    fut_prim = collected_data.get("future_primary_images")
    fut_wrist = collected_data.get("future_wrist_images")
    t = int(actions.shape[0])
    if primary.shape[0] != t or wrist.shape[0] != t or proprio.shape[0] != t:
        raise ValueError(
            f"Length mismatch: actions T={t}, primary={primary.shape[0]}, wrist={wrist.shape[0]}, proprio={proprio.shape[0]}"
        )

    dones = np.zeros(t, dtype=np.uint8)
    dones[-1] = 1
    rewards = np.zeros(t, dtype=np.uint8)
    if episode_success:
        rewards[-1] = 1

    ep_dir = os.path.dirname(ep_filepath)
    if ep_dir:
        os.makedirs(ep_dir, exist_ok=True)
    with h5py.File(ep_filepath, "w") as f:
        f.attrs["task_description"] = task_description
        f.attrs["cosmos_policy_rollout"] = np.bool_(True)
        data_grp = f.create_group("data")
        demo_grp = data_grp.create_group(demo_key)
        obs_grp = demo_grp.create_group("obs")
        if cfg.jpeg_compress:
            dt = h5py.vlen_dtype(np.dtype("uint8"))
            obs_grp.create_dataset(
                "agentview_rgb_jpeg",
                data=[jpeg_encode_image(primary[i], quality=95) for i in range(t)],
                dtype=dt,
            )
            obs_grp.create_dataset(
                "eye_in_hand_rgb_jpeg",
                data=[jpeg_encode_image(wrist[i], quality=95) for i in range(t)],
                dtype=dt,
            )
        else:
            obs_grp.create_dataset("agentview_rgb", data=primary.astype(np.uint8))
            obs_grp.create_dataset("eye_in_hand_rgb", data=wrist.astype(np.uint8))

        if fut_prim is not None:
            fut_prim = _expand_future_images_to_timesteps(fut_prim, t, cfg.num_open_loop_steps)
            if fut_prim.shape[0] != t:
                raise ValueError(
                    f"policy_future primary length {fut_prim.shape[0]} != T={t} after expansion"
                )
            if cfg.jpeg_compress:
                obs_grp.create_dataset(
                    "policy_future_agentview_rgb_jpeg",
                    data=[jpeg_encode_image(fut_prim[i], quality=95) for i in range(t)],
                    dtype=dt,
                )
            else:
                obs_grp.create_dataset("policy_future_agentview_rgb", data=fut_prim, dtype=np.uint8)
        if fut_wrist is not None:
            fut_wrist = _expand_future_images_to_timesteps(fut_wrist, t, cfg.num_open_loop_steps)
            if fut_wrist.shape[0] != t:
                raise ValueError(
                    f"policy_future wrist length {fut_wrist.shape[0]} != T={t} after expansion"
                )
            if cfg.jpeg_compress:
                obs_grp.create_dataset(
                    "policy_future_eye_in_hand_rgb_jpeg",
                    data=[jpeg_encode_image(fut_wrist[i], quality=95) for i in range(t)],
                    dtype=dt,
                )
            else:
                obs_grp.create_dataset(
                    "policy_future_eye_in_hand_rgb",
                    data=fut_wrist,
                    dtype=np.uint8,
                )

        demo_grp.create_dataset("actions", data=actions)
        demo_grp.create_dataset("robot_states", data=np.asarray(proprio, dtype=np.float32))
        demo_grp.create_dataset("future_robot_states", data=np.asarray(future_proprio, dtype=np.float32))
        demo_grp.create_dataset("dones", data=dones)
        demo_grp.create_dataset("rewards", data=rewards)
        demo_grp.attrs["task_description"] = task_description
        demo_grp.attrs["success"] = np.bool_(episode_success)


def _empty_action_queue(cfg: PolicyEvalConfig) -> deque:
    """Return a new deque for the open-loop action buffer (max length ``num_open_loop_steps``)."""
    return deque(maxlen=cfg.num_open_loop_steps)


def _compute_psnr_db(img_a: np.ndarray, img_b: np.ndarray, max_value: float = 255.0) -> float:
    """PSNR (dB) between two same-shaped float RGB images in [0, max_value]."""
    if img_a.shape != img_b.shape:
        raise ValueError(f"Image shape mismatch: {img_a.shape} vs {img_b.shape}")
    mse = float(np.mean((img_a - img_b) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


def _psnr_stats_hdf5_vs_replay(
    ref_frames: np.ndarray,
    replay_frames: list[np.ndarray],
    *,
    flip_ref: bool,
    log_file=None,
    label: str = "primary",
) -> dict[str, float | int]:
    """Frame-aligned PSNR between HDF5 reference RGB and sim replay frames."""
    n = min(int(ref_frames.shape[0]), len(replay_frames))
    if n == 0:
        stats = {
            "n_pairs": 0,
            "mean_psnr": float("nan"),
            "std_psnr": float("nan"),
            "min_psnr": float("nan"),
            "max_psnr": float("nan"),
            "n_identical": 0,
        }
        log_message(f"Replay PSNR ({label}): no frame pairs to compare.", log_file)
        return stats

    if int(ref_frames.shape[0]) != len(replay_frames):
        log_message(
            f"Replay PSNR ({label}): length mismatch ref={int(ref_frames.shape[0])} "
            f"replay={len(replay_frames)}; using first {n} pairs.",
            log_file,
        )

    psnrs: list[float] = []
    for i in range(n):
        ref = np.asarray(ref_frames[i], dtype=np.uint8)
        cmp = np.asarray(replay_frames[i], dtype=np.uint8)
        if cmp.shape[:2] != ref.shape[:2]:
            cmp = np.asarray(
                Image.fromarray(cmp).resize((ref.shape[1], ref.shape[0])),
                dtype=np.uint8,
            )
        psnr = _compute_psnr_db(ref.astype(np.float32), cmp.astype(np.float32))
        psnrs.append(float("inf") if math.isinf(psnr) else float(psnr))

    finite = [x for x in psnrs if not math.isinf(x)]
    stats: dict[str, float | int] = {
        "n_pairs": n,
        "n_identical": sum(1 for x in psnrs if math.isinf(x)),
        "mean_psnr": float(np.mean(finite)) if finite else float("nan"),
        "std_psnr": float(np.std(finite)) if finite else float("nan"),
        "min_psnr": min(finite) if finite else float("nan"),
        "max_psnr": max(finite) if finite else float("nan"),
    }
    log_message(
        f"Replay PSNR ({label}): mean={stats['mean_psnr']:.4f} dB "
        f"(std={stats['std_psnr']:.4f}, min={stats['min_psnr']:.4f}, max={stats['max_psnr']:.4f}, "
        f"n={stats['n_pairs']}, identical={stats['n_identical']})",
        log_file,
    )
    return psnrs , stats


def run_episode_with_predicted_actions(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    hdf5_path: str | None = None,
    *,
    hdf5_demo_key: str = "demo_0",
    initial_state=None,
    log_file=None,
):
    """Replay HDF5 actions in sim and return rendered rollout frames."""
    del model, planning_model, dataset_stats, worker_pool, task_description

    resolved_demo_key = _resolve_libero_hdf5_demo_key(hdf5_path, hdf5_demo_key)
    if resolved_demo_key != hdf5_demo_key:
        log_message(
            f"HDF5 demo {hdf5_demo_key!r} not in {hdf5_path}; using {resolved_demo_key!r} instead.",
            log_file,
        )
    hdf5_demo_key = resolved_demo_key

    _primary, _wrist, _proprio, actions_gt = _load_libero_hdf5_demo_primary_wrist_proprio(
        hdf5_path, hdf5_demo_key
    )

    # Reset environment
    if cfg.deterministic_reset:
        reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(reset_seed)
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    replay_images = []
    replay_wrist_images = [] if cfg.use_wrist_image else None
    future_image_predictions_list = []
    success = False


    if cfg.data_collection:
        primary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []
   
    
    for t, action in enumerate(actions_gt):
        observation = prepare_observation(obs, resize_size, cfg.flip_images)
        replay_images.append(observation["primary_image"])
        if replay_wrist_images is not None:
            replay_wrist_images.append(observation["wrist_image"])

        action_np = np.asarray(action, dtype=np.float32)
        obs, reward, done, info = env.step(action_np.tolist())
        
        if done:
            success = True
            # break
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),
            wrist_images=np.stack(wrist_images_list, axis=0),
            proprio=np.stack(proprio_list, axis=0),
            actions=np.stack(actions_list, axis=0),
            success=success,
        )
    else:
        collected_data = None

    return (
        success,
        replay_images,
        replay_wrist_images,
        future_image_predictions_list,
        collected_data,
        _primary,
        _wrist,
    )


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    if cfg.deterministic_reset:
        reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(reset_seed)
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != cfg.chunk_size:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match cfg.chunk_size "
            f"{cfg.chunk_size}! For best performance (in terms of both speed and success rate), we "
            "recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    replay_wrist_images = [] if cfg.use_wrist_image else None
    future_image_predictions_list = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Best-of-N search variables
    base_seed = cfg.seed  # Used for seed switching (if applicable)

    # Data collection buffers
    if cfg.data_collection:
        primary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []

    # Run episode
    success = False
    reward_window = deque(maxlen=int(cfg.reward_stop_consecutive_steps))
    try:
        NUM_STEPS_WAIT = 10
        while t < max_steps + NUM_STEPS_WAIT:
            # If the deterministic flag is set, reset the random state with the same seed in every step
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                seed = 0
                set_seed_everywhere(seed)

            # Do nothing for the first few timesteps to let objects stabilize
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation = prepare_observation(obs, resize_size, cfg.flip_images)
            replay_images.append(observation["primary_image"])
            if replay_wrist_images is not None:
                replay_wrist_images.append(observation["wrist_image"])

            if cfg.data_collection:
                primary_images_list.append(observation["primary_image"])
                wrist_images_list.append(observation["wrist_image"])
                proprio_list.append(observation["proprio"])

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                best_actions = None
                best_future_predictions = None

                # Query model multiple times if value functions are available
                num_queries = cfg.num_queries_best_of_n

                # Use parallel inference if enabled and multiple queries are needed
                if cfg.use_parallel_inference and num_queries > 1 and worker_pool and worker_pool.initialized:
                    # Query model in parallel
                    start_time = time.time()
                    query_results = query_model_parallel(
                        cfg, observation, task_description, worker_pool, cfg.parallel_timeout
                    )
                    total_query_time = time.time() - start_time
                    log_message(
                        f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                    )
                else:
                    # Serial execution (original behavior)
                    query_results = []
                    for query_idx in range(num_queries):
                        actions_by_depth = []  # Action chunks across all depths of the search
                        future_image_predictions_by_depth = []  # Future image predictions across all depths of the search
                        value_predictions_by_depth = []  # Value predictions across all depths of the search
                        return_dict = {}
                        # Query model to get action
                        start_time = time.time()
                        action_return_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            task_description,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=not (
                                cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                            ),
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries}: Action query time = {query_time:.3f} sec", log_file
                        )
                        return_dict["actions"] = action_return_dict["actions"]
                        actions_by_depth.append(return_dict["actions"])

                        if cfg.ar_future_prediction:
                            # Autoregressively query model to get future state prediction
                            start_time = time.time()
                            future_state_return_dict = get_future_state_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                generated_latent_with_action=action_return_dict["generated_latent"],
                                orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                                future_proprio_latent_idx=action_return_dict["latent_indices"][
                                    "future_proprio_latent_idx"
                                ],
                                future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image_latent_idx"
                                ],
                                future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image2_latent_idx"
                                ],
                                future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                                future_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_image2_latent_idx"
                                ],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Future state prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["future_image_predictions"] = future_state_return_dict[
                                "future_image_predictions"
                            ]
                            future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                        else:
                            return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                        if cfg.ar_value_prediction:
                            # Autoregressively query model to get value prediction
                            start_time = time.time()
                            value_return_dict = get_value_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                log_file,
                            )
                        elif cfg.ar_qvalue_prediction:
                            # Autoregressively query model to get Q-value prediction
                            start_time = time.time()
                            value_return_dict = get_qvalue_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                action_sample=action_return_dict["generated_latent"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                log_file,
                            )
                        else:
                            return_dict["value_prediction"] = action_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])

                        return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                        return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                        return_dict["actions_by_depth"] = actions_by_depth
                        query_results.append(return_dict)

                # Print all value predictions
                log_message(f"t={t}: Current base seed: {base_seed}", log_file)
                for query_idx, return_dict in enumerate(query_results):
                    predicted_value = return_dict["value_prediction"]
                    log_message(
                        f"Query {query_idx + 1}/{num_queries} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                        log_file,
                    )
                # Get dict: seed number -> (action chunk, future state, value)
                seed_to_return_dict = {
                    cfg.seed + query_idx: (
                        return_dict["actions"],
                        return_dict["future_image_predictions"],
                        return_dict["value_prediction"],
                    )
                    for query_idx, return_dict in enumerate(query_results)
                }
                # Get seed with highest value
                best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
                best_actions = best_return_dict[0]
                best_future_predictions = best_return_dict[1]
                best_value_predictions = best_return_dict[2]
                # Use the best actions, future predictions, and value predictions found
                action_queue.extend(best_actions)
                future_image_predictions_list.append(best_future_predictions)
                log_message(f"t={t}: Selected seed {best_seed} with value = {best_value_predictions:.4f}", log_file)

            # Get action from queue
            action = action_queue.popleft()
            # Drop remaining predicted actions so the next step re-queries the model (requery every step).
            # if cfg.data_collection:
            #     action_queue = _empty_action_queue(cfg)
            

            # Process action
            print(f"t: {t}\t action: {action}")


            if cfg.data_collection:
                actions_list.append(action.copy())

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        error_msg = f"Episode error: {e}"
        traceback_str = traceback.format_exc()
        log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    # Fill data collection buffers
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C)
            wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
            proprio=np.stack(proprio_list, axis=0),  # (T, D)
            actions=np.stack(actions_list, axis=0),  # (T, action_dim)
            success=success,
        )
        # Add future image predictions if available
        if len(future_image_predictions_list) > 0:
            if cfg.use_third_person_image:
                future_primary_images = [
                    x["future_image"] for x in future_image_predictions_list if x["future_image"] is not None
                ]
                if len(future_primary_images) > 0:
                    collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            # Wrist image predictions (may be None depending on config)
            if (
                cfg.use_wrist_image
                and "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None

    
        
   
    return success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data


def _split_libero_robot_states_hdf5(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split LIBERO HDF5 ``robot_states`` row into gripper / eef_pos / eef_quat (same layout as training pipeline)."""
    v = np.asarray(rs, dtype=np.float32).ravel()
    d = int(v.shape[0])
    if d == 7:
        return v[0:3].copy(), v[3:6].copy(), v[6:7].copy()
    if d == 9:
        return v[0:2].copy(), v[2:5].copy(), v[5:9].copy()
    if d == 8:
        return v[0:1].copy(), v[1:4].copy(), v[4:8].copy()
    raise ValueError(f"robot_states must be 8 or 9 floats (gripper + eef_pos + eef_quat), got dim {d}")


def _load_libero_hdf5_demo_primary_wrist_proprio(hdf5_path: str, demo_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load agentview + wrist RGB and ``robot_states`` from a LIBERO-style ``data/<demo>/`` group."""
    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo_key}"]
        obs = g["obs"]
        if "agentview_rgb" in obs:
            primary = np.asarray(obs["agentview_rgb"][:], dtype=np.uint8)
        elif "agentview_rgb_jpeg" in obs:
            primary = decode_jpeg_bytes_dataset(obs["agentview_rgb_jpeg"])
        else:
            raise KeyError("HDF5 demo missing obs/agentview_rgb or obs/agentview_rgb_jpeg")
        if "eye_in_hand_rgb" in obs:
            wrist = np.asarray(obs["eye_in_hand_rgb"][:], dtype=np.uint8)
        elif "eye_in_hand_rgb_jpeg" in obs:
            wrist = decode_jpeg_bytes_dataset(obs["eye_in_hand_rgb_jpeg"])
        else:
            raise KeyError("HDF5 demo missing obs/eye_in_hand_rgb or obs/eye_in_hand_rgb_jpeg")
        proprio = np.asarray(g["robot_states"][:], dtype=np.float32)
        actions = np.asarray(g["actions"][:], dtype=np.float32)
    if primary.shape[0] != wrist.shape[0] or primary.shape[0] != proprio.shape[0]:
        raise ValueError(
            f"Length mismatch: primary {primary.shape[0]}, wrist {wrist.shape[0]}, proprio {proprio.shape[0]}"
        )
    return primary, wrist, proprio, actions


def _resolve_libero_hdf5_demo_key(hdf5_path: str, requested: str) -> str:
    """Resolve ``data/<demo>/`` for eval when ``requested`` is ``demo_{episode_idx}``.

    If ``requested`` exists, return it. Otherwise prefer the smallest existing ``demo_*`` index
    strictly greater than ``n``; if none exist, use the largest existing index ``<= n`` (so
    high ``episode_idx`` clips to the last demo in the file instead of jumping to ``demo_0``).
    """
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"No 'data' group in {hdf5_path}")
        data_grp = f["data"]
        demo_keys = sorted(
            [k for k in data_grp.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if not demo_keys:
            raise KeyError(f"No demo_* groups under data/ in {hdf5_path}")
        if requested in data_grp:
            return requested
        try:
            n = int(requested.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            n = 0
        indices = [int(k.split("_")[1]) for k in demo_keys]
        for idx in indices:
            if idx > n:
                return f"demo_{idx}"
        for idx in reversed(indices):
            if idx <= n:
                return f"demo_{idx}"
        return demo_keys[0]


def _raw_observation_from_hdf5_frame(
    primary: np.ndarray,
    wrist: np.ndarray,
    proprio: np.ndarray,
    frame_idx: int,
) -> dict:
    g, p, q = _split_libero_robot_states_hdf5(proprio[frame_idx])
    return {
        "agentview_image": np.asarray(primary[frame_idx], dtype=np.uint8),
        "robot0_eye_in_hand_image": np.asarray(wrist[frame_idx], dtype=np.uint8),
        "robot0_gripper_qpos": g,
        "robot0_eef_pos": p,
        "robot0_eef_quat": q,
    }
    



#This is for REGEN replay trajectory from current task data
def run_episode_with_hdf5_observations(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    hdf5_path: str,
    *,
    hdf5_demo_key: str = "demo_0",
    initial_state=None,
    log_file=None,
):
    """Run one offline rollout using HDF5 frames as policy observations.

    This mirrors ``run_episode`` output structure but does not execute ``env.step``.
    ``hdf5_demo_key`` is usually ``demo_{episode_idx}`` from the eval loop; see
    :func:`_resolve_libero_hdf5_demo_key` if that group is missing.
    """
    del env, worker_pool, initial_state  # Kept for API parity with run_episode.

    resolved_demo_key = _resolve_libero_hdf5_demo_key(hdf5_path, hdf5_demo_key)
    if resolved_demo_key != hdf5_demo_key:
        log_message(
            f"HDF5 demo {hdf5_demo_key!r} not in {hdf5_path}; using {resolved_demo_key!r} instead.",
            log_file,
        )
    hdf5_demo_key = resolved_demo_key

    primary, wrist, proprio, _actions_gt = _load_libero_hdf5_demo_primary_wrist_proprio(hdf5_path, hdf5_demo_key)
    t_max = 200
    # t_max = 150
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images = []
    replay_wrist_images = [] if cfg.use_wrist_image else None
    future_image_predictions_list = []
    future_proprio_list = []
    # Per-query snapshots (one entry per get_action call) for side-by-side future visualization.
    query_primary_images = []
    query_wrist_images = [] if cfg.use_wrist_image else None
    query_timesteps = []

    if cfg.data_collection:
        primary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []

    # End early when recent best values satisfy in the 3-step window:
    # - at least one value very close to 1.0, and
    # - all three values are > 0.99.
    consecutive_val_one_steps = 3
    value_one_eps = 1e-3
    best_val_window = deque(maxlen=consecutive_val_one_steps)

    for t in range(t_max):
        seed_obs = cfg.chunk_size
        if t < seed_obs:
            raw_obs = _raw_observation_from_hdf5_frame(primary, wrist, proprio, t)
            observation = prepare_observation(raw_obs, resize_size, cfg.flip_images)
        else:
            # use future image predictions and proprio predictions for the next chunk to get the action 
            observation = {"primary_image": future_image_predictions_list[t-seed_obs]['future_image'], "wrist_image": future_image_predictions_list[t-seed_obs]['future_wrist_image'], "proprio": future_proprio_list[t-seed_obs]}  
            observation["primary_image"] = np.array(Image.fromarray(observation["primary_image"]).resize((256, 256)))
            if observation["wrist_image"] is not None:
                observation["wrist_image"] = np.array(Image.fromarray(observation["wrist_image"]).resize((256, 256)))
           
        
        replay_images.append(observation["primary_image"])  
        if replay_wrist_images is not None:
            replay_wrist_images.append(observation["wrist_image"])

        if cfg.data_collection:
            primary_images_list.append(observation["primary_image"])
            wrist_images_list.append(observation["wrist_image"])
            proprio_list.append(observation["proprio"])
            
        if len(action_queue) == 0:
            query_primary_images.append(observation["primary_image"])
            query_timesteps.append(t)
            if query_wrist_images is not None:
                query_wrist_images.append(observation["wrist_image"])
            seed_to_pack = {}
            for query_idx in range(cfg.num_queries_best_of_n):
                action_return_dict = get_action(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    task_description,
                    seed=cfg.seed + query_idx,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=not (
                        cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                    ),
                )

                # print(f"action_return_dict: {action_return_dict.keys()}")
                vp = action_return_dict.get("value_prediction", 0.0)
                print(f"vp from policy: {vp}")
                vp = float(vp[0]) if isinstance(vp, (list, tuple)) else float(vp)
                fut = action_return_dict.get("future_image_predictions")
                if cfg.ar_future_prediction:
                    # Autoregressively query model to get future state prediction
                    start_time = time.time()
                    future_state_return_dict = get_future_state_prediction(
                        cfg,
                        model=planning_model if planning_model is not None else model,
                        data_batch=action_return_dict["data_batch"],
                        generated_latent_with_action=action_return_dict["generated_latent"],
                        orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                        future_proprio_latent_idx=action_return_dict["latent_indices"][
                            "future_proprio_latent_idx"
                        ],
                        future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                            "future_wrist_image_latent_idx"
                        ],
                        future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                            "future_wrist_image2_latent_idx"
                        ],
                        future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                        future_image2_latent_idx=action_return_dict["latent_indices"][
                            "future_image2_latent_idx"
                        ],
                        seed=cfg.seed + query_idx,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                        use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                        num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                        future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                        dataset_stats=dataset_stats,
                    )
                
                    fut = future_state_return_dict["future_image_predictions"]
                    f_proprio = future_state_return_dict["future_proprio_prediction"]
                    print(f"f_proprio on action conditioned: {f_proprio}")
                else:
                    fut = action_return_dict["future_image_predictions"]
                    f_proprio = action_return_dict["future_proprio_prediction"]
                
                if cfg.ar_value_prediction:
                    value_return_dict = get_value_prediction(
                        cfg,
                        model=planning_model if planning_model is not None else model,
                        data_batch=action_return_dict["data_batch"],
                        future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                        seed=cfg.seed + query_idx,
                    )
                    vp = value_return_dict.get("value_prediction", 0.0)
                    vp = float(vp[0]) if isinstance(vp, (list, tuple)) else float(vp)
                    print(f"vp from value prediction: {vp}")
                else:
                    vp = action_return_dict.get("value_prediction", 0.0)
                    vp = float(vp[0]) if isinstance(vp, (list, tuple)) else float(vp)
               
                # f_proprio_no_action_conditioned = action_return_dict.get("future_proprio_prediction")
                # print(f"f_proprio on without action conditioned: {f_proprio_no_action_conditioned}")
                # proprio_error = np.linalg.norm(f_proprio - proprio[t+16])
                # print(f"proprio_error: {proprio_error}")
                # proprio_error_no_action_conditioned = np.linalg.norm(f_proprio_no_action_conditioned - proprio[t+16])
                # print(f"proprio_error_no_action_conditioned: {proprio_error_no_action_conditioned}")
                # breakpoint()
                seed_to_pack[cfg.seed + query_idx] = (action_return_dict["actions"], fut, vp, f_proprio)

            
            best_seed, (best_actions, best_future, best_val, best_f_proprio) = max(seed_to_pack.items(), key=lambda x: x[1][2])
            log_message(f"{hdf5_demo_key} t={t}: selected query seed {best_seed} value={best_val:.4f}", log_file)
            action_queue.extend(best_actions)
            future_proprio_list.append(best_f_proprio)
            if best_future is not None:
               future_image_predictions_list.append(best_future)

        action = action_queue.popleft()
        action_queue = _empty_action_queue(cfg)
        if cfg.data_collection:
            actions_list.append(np.asarray(action, dtype=np.float32).copy())
        best_val_window.append(float(best_val))
        if len(best_val_window) >= consecutive_val_one_steps:
            window_vals = list(best_val_window)[-consecutive_val_one_steps:]
            has_one = any(v >= 1.0  for v in window_vals)
            near_one_count = sum(v > 0.99 for v in window_vals)
            if has_one and near_one_count >= 3 :
                log_message(
                    f"{hdf5_demo_key} t={t}: stopping on value criterion window={window_vals} "
                    f"(eps={value_one_eps})",
                    log_file,
                )
                break
           

        if cfg.use_gpt_reward:
            gpt_reward = _get_gpt_reward_score(cfg, observation, task_description)
            log_message(f"{hdf5_demo_key} t={t}: GPT reward={gpt_reward:.4f}", log_file)
            if cfg.gpt_reward_stop_threshold is not None and gpt_reward >= cfg.gpt_reward_stop_threshold:
                log_message(
                    f"{hdf5_demo_key} t={t}: stopping on GPT reward >= {cfg.gpt_reward_stop_threshold:.4f}",
                    log_file,
                )
                break

    success = True  # Offline HDF5 replay has no simulator done/success signal.

    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),
            wrist_images=np.stack(wrist_images_list, axis=0),
            proprio=np.stack(proprio_list, axis=0),
            actions=np.stack(actions_list, axis=0),
            future_proprio=np.stack(future_proprio_list, axis=0),
            success=success,
        )
        if len(future_image_predictions_list) > 0:
            if cfg.use_third_person_image:
                future_primary_images = [
                    x["future_image"] for x in future_image_predictions_list if x.get("future_image") is not None
                ]
                if len(future_primary_images) > 0:
                    collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            if (
                cfg.use_wrist_image
                and future_image_predictions_list
                and "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None

    # Save per-query side-by-side PNGs: [current replay image | predicted future image].
    # if len(future_image_predictions_list) > 0 and len(query_primary_images) > 0 :

    #     task_tag = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    #     out_dir = os.path.join(
    #         cfg.local_log_dir,
    #         "query_side_by_side_frames",
    #         f"{DATE_TIME}--task={task_tag}--demo={hdf5_demo_key}",
    #     )
    #     os.makedirs(out_dir, exist_ok=True)

    #     n = min(len(query_primary_images), len(future_image_predictions_list))
    #     for i in range(n):
    #         curr = np.asarray(query_primary_images[i], dtype=np.uint8)
    #         fut = np.asarray(future_image_predictions_list[i]["future_image"], dtype=np.uint8)
    #         # Match sizes before horizontal concat.
    #         if curr.shape[:2] != fut.shape[:2]:
    #             fut = _resize_single_frame_to_hw(fut, int(curr.shape[0]), int(curr.shape[1]))
    #         sbs = np.concatenate([curr, fut], axis=1)
    #         t_step = query_timesteps[i] if i < len(query_timesteps) else i
    #         img = Image.fromarray(sbs)
    #         _draw_timestep_label_on_pil_image(img, t_step)
    #         img.save(os.path.join(out_dir, f"query_{i:04d}_t{t_step:04d}.png"))
    #     log_message(f"Saved {n} query side-by-side frames to {out_dir}", log_file)
    # # print(f"primary_image shape: {replay_images[0].shape}" , f"wrist_image shape: {replay_wrist_images[0].shape}", f"future_image shape: {future_image_predictions_list[0]['future_image'].shape}", f"future_wrist_image shape: {future_image_predictions_list[0]['future_wrist_image'].shape}")
    return success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data

def get_task_id(task_des: str, task_suite: str) -> int:
     
    task_suite_map = LIBERO_SUITE_TASK_ID_TO_DESCRIPTION[task_suite]
    for task_id, task_description in task_suite_map.items():
        task_description = task_description.replace("_", " ")
        if task_des == task_description:
            return task_id
    return None

def run_task(
    cfg: PolicyEvalConfig,
    task_suite,
    task_id: int,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    if getattr(model, "_packnet_eval_state", None) is not None:
        eval_task_id = apply_packnet_eval_for_libero_task(
            model,
            model._packnet_eval_state,
            task_id,
            first_cl_libero_task=cfg.packnet_first_cl_libero_task,
        )
        if log_file is not None:
            log_message(
                f"[PackNet] libero_task={task_id} → eval_task_id={eval_task_id} "
                f"(first_cl_task={cfg.packnet_first_cl_libero_task})",
                log_file,
            )
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    actual_task_description = task_description
    psnr_task_wise = defaultdict(list)
    if cfg.replay_actions:
        replay_dir = os.path.abspath(os.path.expanduser(cfg.replay_action_path_dir))
        task_replay_paths = sorted(glob.glob(os.path.join(replay_dir, "*_demo*.hdf5")))
        for episode_idx, replay_hdf5_path in enumerate(tqdm.tqdm(task_replay_paths)):
            replay_task_description = _task_description_from_hdf5_filename(replay_hdf5_path)
            log_message(f"Replaying file={replay_hdf5_path} task={replay_task_description}", log_file)
            # Get task
            task_id = get_task_id(replay_task_description, task_suite.name)        
            task = task_suite.get_task(task_id)
            # Get initial states
            initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
            # Initialize environment and get task description
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

            # Handle initial state
            if cfg.initial_states_path == "DEFAULT":
                # Use default initial state
                initial_state = initial_states[episode_idx % 10]
            else:
                # Get keys for fetching initial episode state from JSON
                task_description = actual_task_description
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_0"
                print(initial_states_task_key)
                # Skip episode if expert demonstration failed to complete the task
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                    continue

                # Get initial state
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
            (
                success,
                replay_images,
                replay_wrist_images,
                future_image_predictions_list,
                collected_data,
                hdf5_primary,
                hdf5_wrist,
            ) = run_episode_with_predicted_actions(
                cfg,
                env,
                replay_task_description,
                model,
                planning_model,
                dataset_stats,
                worker_pool,
                resize_size,
                replay_hdf5_path,
                hdf5_demo_key="demo_0",
                initial_state=initial_state,
                log_file=log_file,
            )

            task_episodes += 1
            total_episodes += 1

            if success:
                task_successes += 1
                total_successes += 1

            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=replay_task_description,
                log_file=log_file,
                run_name=cfg.run_id_note,
            )

            psnrs, stats = _psnr_stats_hdf5_vs_replay(
                hdf5_primary,
                replay_images,
                flip_ref=cfg.flip_images,
                log_file=log_file,
                label="primary",
            )
            psnr_task_wise[replay_task_description].extend(psnrs)


        mean_psnr_task_wise = {task_description: np.mean(psnr_task_wise[task_description]) for task_description in psnr_task_wise}
        log_message(f"Mean PSNR task wise: {mean_psnr_task_wise}", log_file)

        overall_psnr = np.mean(list(mean_psnr_task_wise.values()))
        std_psnr = np.std(list(mean_psnr_task_wise.values()))
        log_message(f"Std PSNR: {std_psnr}", log_file)
        log_message(f"Overall PSNR per stage: {overall_psnr}", log_file)
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
        log_message(
            f"Current task success rate: {task_success_rate:.4f} ({task_success_rate * 100:.1f}%)",
            log_file,
        )
        return total_episodes, total_successes
    else :
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            log_message(f"\nTask: {task_description}", log_file)

            # Handle initial state
            if cfg.initial_states_path == "DEFAULT":
                # Use default initial state
                initial_state = initial_states[episode_idx]
            else:
                # Get keys for fetching initial episode state from JSON
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"

                # Skip episode if expert demonstration failed to complete the task
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                    continue

                # Get initial state
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

            log_message(f"Starting episode {task_episodes + 1}...", log_file)

            # Run episode (sim observations, or HDF5 frames for policy inputs when eval_hdf5_path is set)
            if cfg.data_generation  :
                demo_key = f"demo_{episode_idx}"
                log_message(
                    f"hdf5_file: file={cfg.eval_hdf5_path} demo={cfg.eval_hdf5_demo_key} , task={task_description}",
                    log_file,
                )

                success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data = (
                    run_episode_with_hdf5_observations(
                        cfg,
                        env,
                        task_description,
                        model,
                        planning_model,
                        dataset_stats,
                        worker_pool,
                        resize_size,
                        cfg.eval_hdf5_path,
                        hdf5_demo_key=demo_key,
                        initial_state=initial_state,
                        log_file=log_file,
                    )
                )
            else:
                success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data = run_episode(
                    cfg,
                    env,
                    task_description,
                    model,
                    planning_model,
                    dataset_stats,
                    worker_pool,
                    resize_size,
                    initial_state,
                    log_file,
                )

            # Update counters
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Save replay video
            if cfg.save_rollout_video:
                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=success,
                    task_description=task_description+f"",
                    log_file=log_file,
                    run_name=cfg.run_id_note,
                )

            # Save replay video with future image predictions included
            future_primary_image_predictions = None
            if cfg.use_third_person_image:
                future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
            future_wrist_image_predictions = None
            if cfg.use_wrist_image:
                future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]

            print(len(replay_images), len(replay_wrist_images), len(future_primary_image_predictions), len(future_wrist_image_predictions))  
        

            save_rollout_video_with_future_image_predictions(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                chunk_size=cfg.chunk_size,
                num_open_loop_steps=cfg.num_open_loop_steps,
                rollout_wrist_images=replay_wrist_images,
                future_primary_image_predictions=future_primary_image_predictions,
                future_wrist_image_predictions=future_wrist_image_predictions,
                log_file=log_file,
                show_diff=False,
                run_name=cfg.run_id_note,
            )

            # Save episodic data (in data collection mode)
            if cfg.data_collection and collected_data is not None:

                def _save_episode_data():
                    """Save collected episode data as LIBERO-style HDF5 (``data/demo_0/...``)."""
                    ep_filename = f"{task_description.replace(' ', '_')}_demo{total_episodes}.hdf5"
                    rollout_data_dir = os.path.join( BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", cfg.run_id_note)
                    os.makedirs(rollout_data_dir, exist_ok=True)
                    ep_filepath = os.path.join(rollout_data_dir, ep_filename)

                    save_episode_collected_data_libero_hdf5(
                        ep_filepath,
                        cfg,
                        collected_data,
                        task_description,
                        success,
                        demo_key=f"demo_{total_episodes-1}",
                    )
                    # save the collected data to a new hdf5 file
                    data_dir = os.path.join( BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "wv_" + cfg.run_id_note)
                    os.makedirs(data_dir, exist_ok=True)
                    ep_filepath = os.path.join(data_dir, ep_filename)
                    with h5py.File(ep_filepath, "w") as f:
                        for k, v in collected_data.items():
                            if isinstance(v, np.ndarray):
                                is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                                if is_image and cfg.jpeg_compress:
                                    jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                                    dt = h5py.vlen_dtype(np.dtype("uint8"))
                                    f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                                else:
                                    f.create_dataset(k, data=v)
                            else:
                                f.attrs[k] = v
                        f.attrs["task_description"] = task_description


                _save_episode_data()

            # Log results
            log_message(f"Success: {success}", log_file)
            log_message(f"# episodes completed so far: {total_episodes}", log_file)
            log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

        # Log task results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
        log_message(f"Current task success rate: {task_success_rate}", log_file)
        log_message(f"Current total success rate: {total_success_rate}", log_file)

        # Log to wandb if enabled
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{cfg.task_suite_name}/{task_description}": task_success_rate,
                    f"num_episodes/{cfg.task_suite_name}/{task_description}": task_episodes,
                    f"num_successes/{cfg.task_suite_name}/{task_description}": task_successes,
                },
            )

        return (
            total_episodes,
            total_successes,
        )


def _run_task_worker(
    gpu_id: int,
    assigned_task_ids: list,
    cfg: PolicyEvalConfig,
    result_queue,
) -> None:
    """Worker process: pin to one GPU, build model, run a subset of task IDs sequentially.

    Each worker is launched via ``mp.Process`` with the ``spawn`` start method, so this
    function executes in a fresh Python interpreter. We restrict the worker to a single
    GPU by setting ``CUDA_VISIBLE_DEVICES`` BEFORE CUDA is initialized; the model then
    sits on local device 0 (which maps to the physical ``gpu_id``).

    Args:
        gpu_id: Physical GPU index this worker is pinned to.
        assigned_task_ids: Task IDs in the LIBERO suite this worker is responsible for.
        cfg: Full eval config (replicated to the child by spawn pickling).
        result_queue: ``mp.Queue`` for returning per-worker (episodes, successes) totals.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    try:
        if cfg.deterministic:
            os.environ["DETERMINISTIC"] = "True"

        set_seed_everywhere(cfg.seed)
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
        dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! "
            f"Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        planning_model = None
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)

        resize_size = get_image_resize_size(cfg.model_family)

        worker_run_id_note = (cfg.run_id_note or "") + f"--gpu{gpu_id}"
        log_file, _local_log_filepath, _run_id = setup_logging(
            cfg=cfg,
            task_identifier=f"{cfg.task_suite_name}_gpu{gpu_id}",
            log_dir=cfg.local_log_dir,
            run_id_note=worker_run_id_note,
            use_wandb=False,
            wandb_entity=cfg.wandb_entity,
            wandb_project=cfg.wandb_project,
        )
        log_message(
            f"[worker gpu={gpu_id}] tasks={assigned_task_ids} CUDA_VISIBLE_DEVICES={gpu_id}",
            log_file,
        )

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()

        total_episodes, total_successes = 0, 0
        for task_id in assigned_task_ids:
            log_message(f"[worker gpu={gpu_id}] starting task_id={task_id}", log_file)
            total_episodes, total_successes = run_task(
                cfg,
                task_suite,
                task_id,
                model,
                planning_model,
                dataset_stats,
                None,
                resize_size,
                total_episodes,
                total_successes,
                log_file,
            )
            log_message(
                f"[worker gpu={gpu_id}] finished task_id={task_id} cum_ep={total_episodes} cum_succ={total_successes}",
                log_file,
            )

        if log_file:
            log_file.close()

        result_queue.put({
            "gpu_id": gpu_id,
            "assigned_task_ids": assigned_task_ids,
            "total_episodes": int(total_episodes),
            "total_successes": int(total_successes),
            "error": None,
        })
    except Exception as e:
        result_queue.put({
            "gpu_id": gpu_id,
            "assigned_task_ids": assigned_task_ids,
            "total_episodes": 0,
            "total_successes": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        })


def _assign_tasks_round_robin(task_ids: list, gpu_ids: list) -> dict:
    """Round-robin: task_ids[i] -> gpu_ids[i % len(gpu_ids)]."""
    assignment = {gpu: [] for gpu in gpu_ids}
    for i, tid in enumerate(task_ids):
        assignment[gpu_ids[i % len(gpu_ids)]].append(tid)
    return assignment


def _eval_libero_parallel_tasks(cfg: PolicyEvalConfig) -> float:
    """Fan out task IDs across GPUs, one worker process per GPU."""
    mp.set_start_method("spawn", force=True)
    validate_config(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    num_tasks = benchmark_dict[cfg.task_suite_name]().n_tasks
    selected_task_ids = (
        sorted(set(_parse_task_ids_to_run(cfg.task_ids_to_run)))
        if cfg.task_ids_to_run
        else list(range(num_tasks))
    )
    available_gpus = [int(g.strip()) for g in cfg.available_gpus.split(",") if g.strip()]
    assert len(available_gpus) > 0, "available_gpus must list at least one GPU id"
    assert torch.cuda.device_count() >= max(available_gpus) + 1, (
        f"Requested GPU id {max(available_gpus)} but only {torch.cuda.device_count()} CUDA devices visible"
    )

    assignment = _assign_tasks_round_robin(selected_task_ids, available_gpus)
    print(f"[parallel_tasks_across_gpus] task assignment: {assignment}")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for gpu_id, tids in assignment.items():
        if not tids:
            continue
        p = ctx.Process(target=_run_task_worker, args=(gpu_id, tids, cfg, result_queue))
        p.start()
        procs.append(p)

    results = []
    for _ in procs:
        results.append(result_queue.get())
    for p in procs:
        p.join()

    total_episodes = sum(r["total_episodes"] for r in results)
    total_successes = sum(r["total_successes"] for r in results)
    for r in results:
        if r["error"]:
            print(
                f"[parallel_tasks_across_gpus] WORKER FAILURE on gpu={r['gpu_id']} "
                f"tasks={r['assigned_task_ids']}:\n{r['error']}"
            )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print(
        f"[parallel_tasks_across_gpus] DONE total_episodes={total_episodes} "
        f"total_successes={total_successes} success_rate={final_success_rate:.4f}"
    )
    return final_success_rate


@draccus.wrap()
def eval_libero(cfg: PolicyEvalConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""

    # Set DETERMINISTIC environment variable if on deterministic mode (makes some model operations deterministic)
    assert not (cfg.deterministic and cfg.randomize_seed), (
        "Cannot enable both deterministic mode and randomize seed mode!"
    )

    # Task-level GPU parallelism: spawn one worker per GPU and round-robin assign task IDs.
    # Each worker holds its own model copy on a single GPU and runs ``run_task`` for its task IDs.
    if cfg.parallel_tasks_across_gpus:
        return _eval_libero_parallel_tasks(cfg)
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    # Set multiprocessing start method if using parallel inference
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize T5 text embeddings cache
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # Load Cosmos Policy dataset stats
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    # If using parallel inference, initialize worker pool
    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]  # Only need N parallel workers
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        model = None
        planning_model = None

    # If using serial inference, initialize model and Cosmos config
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        worker_pool = None

        # Initialize model for world model and value function
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg.model_family)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_suite_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)

    # Log parallel inference configuration and start worker pool
    if cfg.use_parallel_inference and worker_pool:
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
        log_message(f"Multiprocessing start method: {mp.get_start_method()}", log_file)

        # Verify GPUs are available
        for gpu_id in available_gpus:
            if gpu_id >= torch.cuda.device_count():
                log_message(
                    f"Warning: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found)", log_file
                )

        # Start worker pool
        try:
            log_message("Starting worker pool...", log_file)
            worker_pool.start_workers()
            log_message("Worker pool started successfully", log_file)
        except Exception as e:
            error_msg = f"Failed to start worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)
            log_message("Disabling parallel inference for this run", log_file)
            worker_pool = None
    else:
        log_message("Using serial inference (parallel inference disabled)", log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Number of tasks: {num_tasks}", log_file)
    selected_task_ids = set(_parse_task_ids_to_run(cfg.task_ids_to_run)) if cfg.task_ids_to_run else None
    if selected_task_ids is None:
        log_message("Running all task IDs in suite", log_file)
    else:
        log_message(f"Running only task IDs: {sorted(selected_task_ids)}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        if selected_task_ids is not None and task_id not in selected_task_ids:
            continue
        (
            total_episodes,
            total_successes,
        ) = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{cfg.task_suite_name}/total": final_success_rate,
                f"num_episodes/{cfg.task_suite_name}/total": total_episodes,
                f"num_successes/{cfg.task_suite_name}/total": total_successes,
            },
        )
        wandb.save(local_log_filepath)

    # Cleanup worker pool
    if worker_pool:
        try:
            worker_pool.shutdown()
        except Exception as e:
            error_msg = f"Error shutting down worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
