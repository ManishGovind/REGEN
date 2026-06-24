# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Offline policy inference on LIBERO training HDF5 demos (no simulator).

Loads ``data/<demo>/obs`` (RGB or JPEG) + ``robot_states`` + ``actions`` (GT), builds observations with
the same ``prepare_observation`` path as ``run_episode`` in ``run_libero_eval``, then runs the same
``get_action`` + open-loop queue logic, and compares predictions to stored actions. Optional
``obs/policy_future_*`` (dense or ``*_jpeg``) are written when ``hdf5_output_path`` is set and the model
returns parallel future images (``use_third_person_image`` / ``use_wrist_image``).

``run_single_demo_replay`` returns the same rollout handles as ``run_episode``: ``replay_images``,
``replay_wrist_images`` (or ``None`` if ``use_wrist_image`` is False), ``future_image_predictions_list``,
and ``collected_data`` (or ``None`` unless ``data_collection``).

With ``--data_collection True``, each replayed demo writes a LIBERO-style HDF5 under
``<local_log_dir>/rollout_data/`` (``data/<demo_key>/actions``, ``dones``, ``obs/``, ``rewards``,
``robot_states``, optional ``states``). With ``--hdf5_save_rollout_videos True``, primary / wrist / future
MP4s are written under ``./rollouts/`` like ``run_libero_eval``.

Usage::

    uv run -m cosmos_policy.experiments.robot.libero.run_libero_hdf5_replay \\
        --config cosmos_predict2_2b_480p_libero_single_task_inference_only \\
        --ckpt_path /path/to/model.pt \\
        --config_file cosmos_policy/config/config.py \\
        --dataset_stats_path .../dataset_statistics.json \\
        --t5_text_embeddings_path .../t5_embeddings.pkl \\
        --task_suite_name libero_goal \\
        --chunk_size 16 --num_open_loop_steps 16 \\
        --use_wrist_image True --use_proprio True \\
        --normalize_proprio True --unnormalize_actions True \\
        --trained_with_image_aug True --flip_images True --use_jpeg_compression True \\
        --num_denoising_steps_action 5 \\
        --hdf5_path /path/to/task_demo.hdf5 \\
        --hdf5_demo demo_0 \\
        --hdf5_demos demo_0,demo_1,demo_2,demo_3,demo_4 \\
        # or: --hdf5_num_demos 5 \\
        --hdf5_output_npz /tmp/pred_vs_gt.npz \\
        --hdf5_output_path /tmp/replay_all_demos.hdf5
"""

import logging
import os
import traceback
from collections import deque
from dataclasses import dataclass

import draccus
import h5py
import numpy as np
import tqdm
import wandb

from cosmos_policy.datasets.dataset_utils import decode_jpeg_bytes_dataset, instruction_from_hdf5_filename
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    PolicyEvalConfig,
    prepare_observation,
    save_episode_collected_data_libero_hdf5,
    validate_config,
)
from cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


@dataclass
class Hdf5ReplayEvalConfig(PolicyEvalConfig):
    """Extends sim eval config with paths for offline HDF5 replay."""

    hdf5_path: str = ""
    #: Single demo when ``hdf5_demos`` is empty and ``hdf5_num_demos`` is 0.
    hdf5_demo: str = "demo_0"
    #: Comma-separated demo keys, e.g. ``demo_0,demo_1``. If non-empty, overrides ``hdf5_demo`` / ``hdf5_num_demos``.
    hdf5_demos: str = ""
    #: If >0 and ``hdf5_demos`` is empty, replay the first N demos in order ``demo_0``, ``demo_1``, ... (by sorted index).
    hdf5_num_demos: int = 0
    hdf5_task_description: str = ""  # If empty, parsed from filename (LIBERODataset heuristic)
    hdf5_output_npz: str = ""
    #: One LIBERO-style HDF5 (like the input): all replayed demos under ``data/<demo>/`` with policy
    #: ``actions`` and ``actions_ground_truth``; only demos you replay are included.
    hdf5_output_path: str = ""
    #: If True, write primary (and optional wrist / future) MP4s under ``./rollouts/`` like ``run_libero_eval``.
    hdf5_save_rollout_videos: bool = False


def _validate_hdf5(cfg: Hdf5ReplayEvalConfig) -> None:
    validate_config(cfg)
    assert cfg.hdf5_path, "hdf5_path is required"
    assert os.path.isfile(cfg.hdf5_path), f"hdf5_path not found: {cfg.hdf5_path}"
    assert cfg.suite == "libero", "HDF5 replay is implemented for suite=libero only"
    if cfg.use_parallel_inference:
        raise ValueError("Set use_parallel_inference=False (serial inference only).")
    if cfg.hdf5_num_demos < 0:
        raise ValueError("hdf5_num_demos must be >= 0")


def _sorted_demo_keys_hdf5(path: str) -> list[str]:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"No top-level group 'data' in {path}")
        demos = list(f["data"].keys())
        if not demos:
            raise ValueError(f"Empty 'data' group in {path}")
        return sorted(demos, key=lambda x: int(x.split("_")[1]))


def resolve_hdf5_demo_keys(cfg: Hdf5ReplayEvalConfig, path: str) -> list[str]:
    path = os.path.abspath(os.path.expanduser(path))
    if cfg.hdf5_demos.strip():
        keys = [s.strip() for s in cfg.hdf5_demos.split(",") if s.strip()]
        if not keys:
            raise ValueError("hdf5_demos is empty after parsing")
        with h5py.File(path, "r") as f:
            for dk in keys:
                if "data" not in f or dk not in f["data"]:
                    raise KeyError(f"{dk!r} not found under data/ in {path}")
        return keys
    if cfg.hdf5_num_demos > 0:
        all_keys = _sorted_demo_keys_hdf5(path)
        return all_keys[: min(cfg.hdf5_num_demos, len(all_keys))]
    with h5py.File(path, "r") as f:
        if cfg.hdf5_demo not in f.get("data", {}):
            raise KeyError(f"{cfg.hdf5_demo!r} not found under data/ in {path}")
    return [cfg.hdf5_demo]


def _output_path_for_demo(base_path: str, demo_key: str, multi: bool) -> str:
    if not base_path or not multi:
        return base_path
    base_path = os.path.abspath(os.path.expanduser(base_path))
    root, ext = os.path.splitext(base_path)
    return f"{root}__{demo_key}{ext}"


def _validate_dataset_stats_for_get_action(cfg: Hdf5ReplayEvalConfig, dataset_stats: dict) -> None:
    """Same keys ``get_action`` / ``unnormalize_actions`` / ``rescale_proprio`` require (no sim-eval changes)."""
    if cfg.unnormalize_actions:
        assert "actions_min" in dataset_stats and "actions_max" in dataset_stats, (
            "dataset_statistics must include actions_min and actions_max when unnormalize_actions is True"
        )
    if cfg.normalize_proprio:
        assert "proprio_min" in dataset_stats and "proprio_max" in dataset_stats, (
            "dataset_statistics must include proprio_min and proprio_max when normalize_proprio is True"
        )


def load_libero_demo_sequences(hdf5_path: str, demo_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if primary.shape[0] != wrist.shape[0] or primary.shape[0] != proprio.shape[0] or primary.shape[0] != actions.shape[0]:
        raise ValueError(
            f"Length mismatch: primary {primary.shape[0]}, wrist {wrist.shape[0]}, "
            f"proprio {proprio.shape[0]}, actions {actions.shape[0]}"
        )
    return primary, wrist, proprio, actions


def _split_libero_robot_states_vec(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split HDF5 ``robot_states`` row into env keys used by ``prepare_observation`` (see ``regenerate_libero_dataset``)."""
    v = np.asarray(rs, dtype=np.float32).ravel()
    d = int(v.shape[0])
    if d == 9:
        return v[0:2].copy(), v[2:5].copy(), v[5:9].copy()
    if d == 8:
        return v[0:1].copy(), v[1:4].copy(), v[4:8].copy()
    raise ValueError(
        f"robot_states must be 8 or 9 floats (gripper + eef_pos + eef_quat), got dim {d}"
    )


def observation_from_training_hdf5_frame(
    primary: np.ndarray,
    wrist: np.ndarray,
    proprio: np.ndarray,
    t: int,
    resize_size,
    flip_images: bool,
) -> dict:
    """Build the same ``observation`` dict as sim eval: ``run_episode`` → ``prepare_observation`` → ``get_action``."""
    g, p, q = _split_libero_robot_states_vec(proprio[t])
    obs = {
        "agentview_image": np.asarray(primary[t], dtype=np.uint8),
        "robot0_eye_in_hand_image": np.asarray(wrist[t], dtype=np.uint8),
        "robot0_gripper_qpos": g,
        "robot0_eef_pos": p,
        "robot0_eef_quat": q,
    }
    return prepare_observation(obs, resize_size, flip_images)


def _expand_future_chunk_to_timesteps(
    chunk_images: list[np.ndarray | None],
    T: int,
    num_open_loop_steps: int,
) -> np.ndarray | None:
    """Repeat each chunk's future image for ``num_open_loop_steps`` steps (same indexing as eval video)."""
    valid = [x for x in chunk_images if x is not None]
    if not valid:
        return None
    n_chunks = len(chunk_images)
    if n_chunks == 0:
        return None
    h0, w0, c0 = valid[0].shape
    if c0 != 3:
        raise ValueError(f"future image must be HxWx3, got {valid[0].shape}")
    out = np.zeros((T, h0, w0, 3), dtype=np.uint8)
    for t in range(T):
        ci = min(t // max(1, num_open_loop_steps), n_chunks - 1)
        img = chunk_images[ci]
        if img is None:
            continue
        img = np.asarray(img, dtype=np.uint8)
        if img.shape[0] != h0 or img.shape[1] != w0:
            raise ValueError(f"Inconsistent future image shapes: expected {(h0, w0)}, got {img.shape[:2]}")
        out[t] = img
    return out


def _build_future_image_timeseries(
    future_image_predictions_list: list[dict],
    T: int,
    num_open_loop_steps: int,
    *,
    use_primary: bool,
    use_wrist: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not future_image_predictions_list:
        return None, None
    prim_chunks: list[np.ndarray | None] = []
    wrist_chunks: list[np.ndarray | None] = []
    for d in future_image_predictions_list:
        if use_primary:
            fi = d.get("future_image") if isinstance(d, dict) else None
            prim_chunks.append(np.asarray(fi, dtype=np.uint8) if fi is not None else None)
        if use_wrist:
            fw = d.get("future_wrist_image") if isinstance(d, dict) else None
            wrist_chunks.append(np.asarray(fw, dtype=np.uint8) if fw is not None else None)
    primary_T = (
        _expand_future_chunk_to_timesteps(prim_chunks, T, num_open_loop_steps)
        if use_primary
        else None
    )
    wrist_T = (
        _expand_future_chunk_to_timesteps(wrist_chunks, T, num_open_loop_steps)
        if use_wrist
        else None
    )
    return primary_T, wrist_T


def _libero_demo_timesteps(demo_group: h5py.Group) -> int:
    obs = demo_group["obs"]
    if "agentview_rgb" in obs:
        return int(obs["agentview_rgb"].shape[0])
    if "agentview_rgb_jpeg" in obs:
        return int(obs["agentview_rgb_jpeg"].shape[0])
    raise KeyError("demo obs missing agentview_rgb / agentview_rgb_jpeg")


def save_libero_style_replay_hdf5_bundle(
    src_path: str,
    dst_path: str,
    demo_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    per_demo_mse: dict[str, float] | None = None,
    per_demo_l1: dict[str, float] | None = None,
    demo_future_obs: dict[str, tuple[np.ndarray | None, np.ndarray | None]] | None = None,
    overwrite: bool = True,
) -> None:
    """Write one LIBERO-style HDF5 with multiple ``data/<demo>/`` groups (training layout).

    Root attributes are copied from ``src_path`` (plus replay metadata). Each key in
    ``demo_predictions`` must exist under ``data/`` in the source. Demos not listed are omitted
    from the output file.
    """
    if not demo_predictions:
        raise ValueError("demo_predictions is empty")

    dst_path = os.path.abspath(os.path.expanduser(dst_path))
    ddir = os.path.dirname(dst_path)
    if ddir:
        os.makedirs(ddir, exist_ok=True)
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"HDF5 output exists (set overwrite or choose another path): {dst_path}")

    keys_sorted = sorted(demo_predictions.keys(), key=lambda x: int(x.split("_")[1]))

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for ak in src.attrs:
            dst.attrs[ak] = src.attrs[ak]
        dst.attrs["replay_source_hdf5"] = os.path.abspath(src_path)
        dst.attrs["replay_demo_keys"] = ",".join(keys_sorted)
        if per_demo_mse and per_demo_l1:
            dst.attrs["replay_mse_mean"] = np.float64(float(np.mean([per_demo_mse[k] for k in keys_sorted])))
            dst.attrs["replay_l1_mean"] = np.float64(float(np.mean([per_demo_l1[k] for k in keys_sorted])))

        if "data" not in src:
            raise KeyError(f"No data group in {src_path}")

        dst.create_group("data")
        for demo_key in keys_sorted:
            pred, gt = demo_predictions[demo_key]
            pred = np.asarray(pred, dtype=np.float32)
            gt = np.asarray(gt, dtype=np.float32)
            if pred.shape != gt.shape:
                raise ValueError(
                    f"{demo_key}: predicted shape {pred.shape} != ground_truth {gt.shape}"
                )
            if demo_key not in src["data"]:
                raise KeyError(f"No data/{demo_key} in {src_path}")

            src.copy(f"data/{demo_key}", dst["data"], name=demo_key)
            g = dst[f"data/{demo_key}"]
            T = _libero_demo_timesteps(g)
            if pred.shape[0] != T:
                raise ValueError(
                    f"{demo_key}: predicted length {pred.shape[0]} != demo obs length {T}"
                )

            if "actions" in g:
                del g["actions"]
            g.create_dataset("actions", data=pred)
            if "actions_ground_truth" in g:
                del g["actions_ground_truth"]
            g.create_dataset("actions_ground_truth", data=gt)

            if per_demo_mse and demo_key in per_demo_mse:
                g.attrs["replay_mse_mean"] = np.float64(per_demo_mse[demo_key])
            if per_demo_l1 and demo_key in per_demo_l1:
                g.attrs["replay_l1_mean"] = np.float64(per_demo_l1[demo_key])

            if demo_future_obs and demo_key in demo_future_obs:
                fut_prim, fut_wrist = demo_future_obs[demo_key]
                obs_grp = g["obs"]
                use_jpeg = "agentview_rgb_jpeg" in obs_grp or "eye_in_hand_rgb_jpeg" in obs_grp
                vlen_dt = h5py.vlen_dtype(np.dtype("uint8")) if use_jpeg else None
                for _old in (
                    "policy_future_agentview_rgb",
                    "policy_future_eye_in_hand_rgb",
                    "policy_future_agentview_rgb_jpeg",
                    "policy_future_eye_in_hand_rgb_jpeg",
                ):
                    if _old in obs_grp:
                        del obs_grp[_old]
                if fut_prim is not None:
                    if fut_prim.shape[0] != T:
                        raise ValueError(
                            f"{demo_key}: policy_future primary length {fut_prim.shape[0]} != T={T}"
                        )
                    if use_jpeg:
                        prim_list = [jpeg_encode_image(fut_prim[t], quality=95) for t in range(T)]
                        obs_grp.create_dataset(
                            "policy_future_agentview_rgb_jpeg", data=prim_list, dtype=vlen_dt
                        )
                    else:
                        obs_grp.create_dataset("policy_future_agentview_rgb", data=fut_prim, dtype=np.uint8)
                if fut_wrist is not None:
                    if fut_wrist.shape[0] != T:
                        raise ValueError(
                            f"{demo_key}: policy_future wrist length {fut_wrist.shape[0]} != T={T}"
                        )
                    if use_jpeg:
                        wrist_list = [jpeg_encode_image(fut_wrist[t], quality=95) for t in range(T)]
                        obs_grp.create_dataset(
                            "policy_future_eye_in_hand_rgb_jpeg", data=wrist_list, dtype=vlen_dt
                        )
                    else:
                        obs_grp.create_dataset(
                            "policy_future_eye_in_hand_rgb", data=fut_wrist, dtype=np.uint8
                        )


def _save_collected_data_rollout_hdf5(
    cfg: Hdf5ReplayEvalConfig,
    collected_data: dict,
    log_file,
    *,
    demo_key: str,
    task_description: str,
) -> None:
    """Write ``collected_data`` in LIBERO training layout under ``<local_log_dir>/rollout_data/``."""
    note = cfg.run_id_note or "replay"
    ep_filename = (
        f"episode_data--hdf5_replay--{DATE_TIME}--demo={demo_key}--success={collected_data['success']}--{note}.hdf5"
    )
    rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data")
    os.makedirs(rollout_data_dir, exist_ok=True)
    ep_filepath = os.path.join(rollout_data_dir, ep_filename)
    save_episode_collected_data_libero_hdf5(
        ep_filepath,
        cfg,
        collected_data,
        task_description,
        bool(collected_data.get("success", True)),
        demo_key=demo_key,
    )
    with h5py.File(ep_filepath, "a") as f:
        f.attrs["hdf5_replay_source"] = os.path.abspath(os.path.expanduser(cfg.hdf5_path))
        f.attrs["hdf5_replay_demo_key"] = demo_key
    log_message(f"Saved collected_data rollout HDF5 {ep_filepath}", log_file)


def save_libero_style_replay_hdf5(
    src_path: str,
    demo_key: str,
    dst_path: str,
    predicted_actions: np.ndarray,
    ground_truth_actions: np.ndarray,
    *,
    mse_mean: float,
    l1_mean: float,
    overwrite: bool = True,
) -> None:
    """Write a single-demo replay file (wrapper around :func:`save_libero_style_replay_hdf5_bundle`)."""
    save_libero_style_replay_hdf5_bundle(
        src_path,
        dst_path,
        {demo_key: (predicted_actions, ground_truth_actions)},
        per_demo_mse={demo_key: mse_mean},
        per_demo_l1={demo_key: l1_mean},
        overwrite=overwrite,
    )


def run_single_demo_replay(
    cfg: Hdf5ReplayEvalConfig,
    model,
    planning_model,
    dataset_stats,
    log_file,
    resize_size,
    path: str,
    demo_key: str,
    task_description: str,
    *,
    output_npz: str,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    list,
    list | None,
    list,
    dict | None,
]:
    primary, wrist, proprio, actions_gt = load_libero_demo_sequences(path, demo_key)

    t_max = primary.shape[0]
    log_message(f"HDF5 replay: demo={demo_key} T={t_max} task={task_description!r}", log_file)

    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    predicted_rows: list[np.ndarray] = []
    future_image_predictions_list: list[dict] = []
    gen_parallel = not (
        cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
    )

    replay_images: list[np.ndarray] = []
    replay_wrist_images: list[np.ndarray] | None = [] if cfg.use_wrist_image else None

    if cfg.data_collection:
        wrist_images_list: list[np.ndarray] = []
        proprio_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []

    pbar_desc = f"hdf5_replay:{demo_key}"
    for t in tqdm.tqdm(range(t_max), desc=pbar_desc):
        if os.environ.get("DETERMINISTIC", "").lower() == "true":
            set_seed_everywhere(0)

        observation = observation_from_training_hdf5_frame(
            primary, wrist, proprio, t, resize_size, cfg.flip_images
        )

        replay_images.append(observation["primary_image"])
        if replay_wrist_images is not None:
            replay_wrist_images.append(observation["wrist_image"])

        if cfg.data_collection:
            wrist_images_list.append(observation["wrist_image"])
            proprio_list.append(observation["proprio"])

        if len(action_queue) == 0:
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
                    generate_future_state_and_value_in_parallel=gen_parallel,
                )
                vp = action_return_dict.get("value_prediction", 0.0)
                if isinstance(vp, (list, tuple)):
                    vp = float(vp[0])
                else:
                    vp = float(vp)
                fut = action_return_dict.get("future_image_predictions")
                seed_to_pack[cfg.seed + query_idx] = (action_return_dict["actions"], fut, vp)

            best_seed, (best_actions, best_future, best_val) = max(
                seed_to_pack.items(), key=lambda x: x[1][2]
            )
            log_message(f"{demo_key} t={t}: selected query seed {best_seed} value={best_val:.4f}", log_file)
            action_queue.extend(best_actions)
            if gen_parallel and best_future is not None:
                future_image_predictions_list.append(best_future)

        action = action_queue.popleft()
        predicted_rows.append(np.asarray(action, dtype=np.float64))
        if cfg.data_collection:
            actions_list.append(np.asarray(action, dtype=np.float32).copy())
        if (t + 1) % max(1, cfg.num_open_loop_steps) == 0:
            log_message(f"{demo_key} t={t + 1}/{t_max} action[0:3]={predicted_rows[-1][:3]}", log_file)

    pred = np.stack(predicted_rows, axis=0)
    if pred.shape != actions_gt.shape:
        log_message(f"{demo_key}: Warning: predicted shape {pred.shape} vs GT {actions_gt.shape}", log_file)
    m = min(pred.shape[0], actions_gt.shape[0])
    pred_m = pred[:m]
    gt_m = actions_gt[:m]
    mse = float(np.mean((pred_m - gt_m) ** 2))
    l1 = float(np.mean(np.abs(pred_m - gt_m)))
    log_message(
        f"{demo_key}: vs demo actions mean MSE={mse:.6f} mean L1={l1:.6f} (over {m} steps)", log_file
    )

    fut_prim_T, fut_wrist_T = _build_future_image_timeseries(
        future_image_predictions_list,
        t_max,
        cfg.num_open_loop_steps,
        use_primary=cfg.use_third_person_image,
        use_wrist=cfg.use_wrist_image,
    )
    if cfg.flip_images:
        if fut_prim_T is not None:
            fut_prim_T = np.ascontiguousarray(np.flip(fut_prim_T, axis=1))
        if fut_wrist_T is not None:
            fut_wrist_T = np.ascontiguousarray(np.flip(fut_wrist_T, axis=1))

    if output_npz:
        out_path = os.path.abspath(os.path.expanduser(output_npz))
        odir = os.path.dirname(out_path)
        # if odir:
        #     os.makedirs(odir, exist_ok=True)
        # save_kw: dict = dict(
        #     predicted_actions=pred,
        #     ground_truth_actions=actions_gt,
        #     mse_mean=np.float64(mse),
        #     l1_mean=np.float64(l1),
        # )
       
        log_message(f"Saved {out_path}", log_file)

    collected_data: dict | None = None
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(replay_images, axis=0),
            wrist_images=np.stack(wrist_images_list, axis=0),
            proprio=np.stack(proprio_list, axis=0),
            actions=np.stack(actions_list, axis=0),
            success=True,
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

    return (
        mse,
        l1,
        pred,
        actions_gt,
        fut_prim_T,
        fut_wrist_T,
        replay_images,
        replay_wrist_images,
        future_image_predictions_list,
        collected_data,
    )


def run_hdf5_replay(
    cfg: Hdf5ReplayEvalConfig,
    model,
    planning_model,
    dataset_stats,
    log_file,
    resize_size,
) -> tuple[float, float]:
    path = os.path.abspath(os.path.expanduser(cfg.hdf5_path))
    demo_keys = resolve_hdf5_demo_keys(cfg, path)
    multi = len(demo_keys) > 1
    log_message(f"HDF5 replay file={path} demos={demo_keys} (n={len(demo_keys)})", log_file)
    if cfg.hdf5_num_demos > 0 and not cfg.hdf5_demos.strip():
        total = len(_sorted_demo_keys_hdf5(path))
        if cfg.hdf5_num_demos > total:
            log_message(
                f"Note: hdf5_num_demos={cfg.hdf5_num_demos} but file has only {total} demo(s); replayed {len(demo_keys)}",
                log_file,
            )

    # task_description = cfg.hdf5_task_description.strip() or instruction_from_hdf5_filename(path)
    task_description = "put the bowl on the plate"
    if not task_description:
        log_message(
            "Warning: empty task_description; set --hdf5_task_description or use a *_demo.hdf5 style name",
            log_file,
        )

    mses: list[float] = []
    l1s: list[float] = []
    demo_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    demo_future_obs: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    per_demo_mse: dict[str, float] = {}
    per_demo_l1: dict[str, float] = {}

    for ep_idx, demo_key in enumerate(demo_keys):
        out_npz = _output_path_for_demo(cfg.hdf5_output_npz, demo_key, multi)
        (
            mse,
            l1,
            pred,
            actions_gt,
            replay_images,
            replay_wrist_images,
            future_image_predictions_list,
            collected_data,
        ) = run_single_demo_replay(
            cfg,
            model,
            planning_model,
            dataset_stats,
            log_file,
            resize_size,
            path,
            demo_key,
            task_description,
            output_npz=out_npz,
        )
        mses.append(mse)
        l1s.append(l1)
        demo_predictions[demo_key] = (pred, actions_gt)

        per_demo_mse[demo_key] = mse
        per_demo_l1[demo_key] = l1
        if cfg.data_collection and collected_data is not None:
            _save_collected_data_rollout_hdf5(
                cfg,
                collected_data,
                log_file,
                demo_key=demo_key,
                task_description=task_description,
            )
        if cfg.hdf5_save_rollout_videos:
            replay_ok = True
            vid_task = f"Replay_{task_description}__{demo_key}"
            # NOTE: LIBERO training-time image augmentation can vertically flip frames
            # (`cfg.flip_images`). The policy should still consume flipped images, but
            # video visualization should use the original orientation.
            replay_images_for_video = (
                [np.flipud(x) for x in replay_images] if cfg.flip_images else replay_images
            )
            replay_wrist_images_for_video = (
                [np.flipud(x) for x in replay_wrist_images]  # type: ignore[arg-type]
                if (cfg.flip_images and replay_wrist_images is not None)
                else replay_wrist_images
            )

            save_rollout_video(
                replay_images_for_video,
                ep_idx + 1,
                success=replay_ok,
                task_description=vid_task,
                log_file=log_file,
            )
            fut_prim_pred = None
            if future_image_predictions_list:
                fut_prim_pred = (
                    [np.flipud(x["future_image"]) for x in future_image_predictions_list]  # type: ignore[index]
                    if cfg.flip_images
                    else [x["future_image"] for x in future_image_predictions_list]
                )
            fut_wrist_pred = None
            if cfg.use_wrist_image and future_image_predictions_list:
                fut_wrist_pred = (
                    [np.flipud(x["future_wrist_image"]) for x in future_image_predictions_list]  # type: ignore[index]
                    if cfg.flip_images
                    else [x["future_wrist_image"] for x in future_image_predictions_list]
                )
            save_rollout_video_with_future_image_predictions(
                replay_images_for_video,
                ep_idx + 1,
                success=replay_ok,
                task_description=vid_task,
                chunk_size=cfg.chunk_size,
                num_open_loop_steps=cfg.num_open_loop_steps,
                rollout_wrist_images=replay_wrist_images_for_video,
                future_primary_image_predictions=fut_prim_pred,
                future_wrist_image_predictions=fut_wrist_pred,
                log_file=log_file,
                show_diff=False,
            )

    if cfg.hdf5_output_path:
        h5_out = os.path.abspath(os.path.expanduser(cfg.hdf5_output_path))
        save_libero_style_replay_hdf5_bundle(
            path,
            h5_out,
            demo_predictions,
            per_demo_mse=per_demo_mse,
            per_demo_l1=per_demo_l1,
            demo_future_obs=demo_future_obs,
            overwrite=True,
        )
        log_message(
            f"Saved LIBERO-format HDF5 {h5_out} with {len(demo_predictions)} demo(s) under data/ "
            "(actions, actions_ground_truth, obs/policy_future_* when images enabled)",
            log_file,
        )

    mean_mse = float(np.mean(mses)) if mses else 0.0
    mean_l1 = float(np.mean(l1s)) if l1s else 0.0
    log_message(
        f"All demos: mean of per-demo MSE={mean_mse:.6f} mean L1={mean_l1:.6f} ({len(demo_keys)} demo(s))",
        log_file,
    )
    return mean_mse, mean_l1


@draccus.wrap()
def main(cfg: Hdf5ReplayEvalConfig) -> float:
    assert not (cfg.deterministic and cfg.randomize_seed), "Cannot use both deterministic and randomize_seed"
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    _validate_hdf5(cfg)
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    _validate_dataset_stats_for_get_action(cfg, dataset_stats)

    model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
        f"Train chunk {cosmos_config.dataloader_train.dataset.chunk_size} != test {cfg.chunk_size}"
    )
    if cfg.planning_model_ckpt_path != "":
        planning_model, _ = get_planning_model(cfg)
    else:
        planning_model = None

    resize_size = get_image_resize_size(cfg.model_family)
    log_file, local_log_filepath, _run_id = setup_logging(
        cfg=cfg,
        task_identifier="hdf5_replay",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Config: {cfg}", log_file)

    try:
        mse, l1 = run_hdf5_replay(cfg, model, planning_model, dataset_stats, log_file, resize_size)
        if cfg.use_wandb:
            wandb.log({"hdf5_replay/mse_mean": mse, "hdf5_replay/l1_mean": l1})
            wandb.save(local_log_filepath)
    except Exception:
        log_message(traceback.format_exc(), log_file)
        raise
    finally:
        if log_file:
            log_file.close()

    return mse


if __name__ == "__main__":
    raise SystemExit(main())
