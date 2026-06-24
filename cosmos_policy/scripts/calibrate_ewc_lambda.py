# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Calibrate ``ewc_lambda`` for the Cosmos Policy.

Choosing ``ewc_lambda`` blindly is not great -- the right value depends on the
**ratio** between the task loss ``L_task`` and the raw EWC quadratic

    Q = (1/2) * sum_i F_i * (theta_i - theta*_i)**2

both of which are model-, dataset-, and dtype-specific. This script measures
that ratio empirically for *your* base-stage checkpoint and computes the
``ewc_lambda`` that targets a given ratio ``alpha`` between the EWC penalty and
the task loss::

    L_ewc = lambda * Q  ~=  alpha * L_task   =>   lambda = alpha * L_task / Q

How it works
------------
1. Load the base-stage checkpoint and the EWC state (Fisher + theta*) computed
   by ``compute_ewc_fisher.py``.
2. Build the *next* CL stage's training dataloader (so the data distribution
   and batch shape match what you'll actually train on).
3. Run a short calibration loop with ``ewc_lambda = 1.0`` so that the
   ``ewc_penalty`` reported by the model is exactly ``Q`` (per-rank local
   shard). Per step we all-reduce ``Q`` across DP ranks (sum) and ``L_task``
   across DP ranks (mean) to recover global metrics.
4. After ``--n_calibration_steps`` steps, average the metrics over the second
   half of the run (skipping warmup where ``Q`` is still tiny because
   ``theta`` has not yet drifted away from ``theta*``) and print recommended
   ``ewc_lambda`` values for a sweep of ``alpha`` targets.

Typical usage::

    uv run --extra cu128 --group libero --python 3.10 \\
        torchrun --nproc_per_node=8 --master_port=12341 \\
        -m cosmos_policy.scripts.calibrate_ewc_lambda \\
        --experiment cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage \\
        --ckpt /workspace/checkpoints/.../iter_000007000 \\
        --ewc_state_path /workspace/ewc_states/libero_goal/base_stage_iter7000 \\
        --n_calibration_steps 100

``--ewc_state_path`` must be the DCP directory from ``compute_ewc_fisher``
(``meta.pt`` + ``ewc_state/``) or a legacy ``.pt`` file — not the inner
``ewc_state/`` subfolder alone.

You only specify the experiment **name** -- the script internally loads
``cosmos_policy/config/config.py`` (the same config router used by
``cosmos_policy.scripts.train``) and applies the experiment + EWC overrides
itself. No Hydra ``--`` separator or ``key=value`` syntax is required on the
command line.

The script does **not** save anything; it just prints recommended values.
Pick one (typically the ``alpha = 1.0`` row) and pass it to your real CL stage
training as ``model.config.ewc_lambda=<value>``.

Note
----
This is a *probe* run -- it actually takes optimizer steps so that
``theta`` drifts away from ``theta*`` (otherwise ``Q = 0`` trivially). Those
parameter updates are discarded when the script exits; nothing is written to
the checkpoint dir.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List

from loguru import logger as logging
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.config import Config, load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_policy.continual.ewc import (
    _DCP_META_FILE,
    _DCP_TENSOR_SUBDIR,
    _resolve_dcp_checkpoint_dir,
)

# Same config router that cosmos_policy.scripts.train uses. We hardcode it so
# users only need to pass --experiment <name> on the CLI.
_CONFIG_PY = "cosmos_policy/config/config.py"


def _ewc_state_path_exists(path: str) -> bool:
    """True if ``path`` is a legacy ``.pt`` file or a complete DCP checkpoint root."""
    path = path.rstrip("/")
    if os.path.isfile(path):
        return True
    if os.path.isdir(path):
        root = _resolve_dcp_checkpoint_dir(path)
        return os.path.isfile(os.path.join(root, _DCP_META_FILE)) and os.path.isdir(
            os.path.join(root, _DCP_TENSOR_SUBDIR)
        )
    return False


def _ewc_state_path_hint(path: str) -> str:
    parent = os.path.dirname(path.rstrip("/")) or "."
    if os.path.isdir(parent):
        siblings = sorted(
            f
            for f in os.listdir(parent)
            if f.startswith("base_stage") or "ewc" in f.lower() or f.endswith(".pt")
        )
        if siblings:
            return f"\nUnder {parent}/ found: {siblings[:8]}"
    return ""


def _build_overrides(args: argparse.Namespace) -> list[str]:
    """Build the Hydra override list for load_config().

    First element MUST be ``"--"`` (load_config / Hydra's contract). Subsequent
    elements are standard ``key=value`` overrides. Critically, this sets
    ``ewc_lambda=1.0`` so that the model's ``ewc_penalty`` term during the
    probe equals the raw quadratic ``Q`` (per-rank-local); this is what makes
    the calibration math work.
    """
    overrides: list[str] = ["--"]
    overrides.append(f"experiment={args.experiment}")
    overrides.append("model.config.ewc_enabled=true")
    overrides.append("model.config.ewc_lambda=1.0")
    overrides.append("model.config.ewc_gamma=1.0")  # irrelevant during probe
    # EWC state is loaded manually after checkpointer.load (not via config path).
    overrides.append("model.config.ewc_prev_state_path=")
    overrides.append("model.config.ewc_save_state_path=")
    overrides.append("model.config.ewc_log_every=0")  # silence per-step EWC log
    if args.ckpt:
        overrides.append(f"checkpoint.load_path={args.ckpt}")
    overrides.append("checkpoint.load_training_state=false")
    overrides.append("checkpoint.strict_resume=false")
    for extra in args.extra_override or []:
        overrides.append(extra)
    return overrides


def _all_reduce_scalar(value: float, op: str) -> float:
    """All-reduce a python float across the DP mesh and return the result.

    ``op`` is either ``"sum"`` (used for Q, which is partitioned across ranks)
    or ``"mean"`` (used for L_task, which is per-rank-local but a sample from
    the population)."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if op == "mean":
        t = t / dist.get_world_size()
    elif op != "sum":
        raise ValueError(f"Unknown reduction op: {op}")
    return float(t.item())


@logging.catch(reraise=True)
def launch(args: argparse.Namespace) -> None:
    if not _ewc_state_path_exists(args.ewc_state_path):
        raise FileNotFoundError(
            f"--ewc_state_path is not a valid EWC checkpoint: {args.ewc_state_path}\n"
            "Expected a DCP directory (meta.pt + ewc_state/) from compute_ewc_fisher "
            "with --format dcp, e.g. ./ewc_states/libero_goal/base_stage_iter7000\n"
            "or a legacy .pt file. Run compute_ewc_fisher first if missing."
            f"{_ewc_state_path_hint(args.ewc_state_path)}"
        )

    overrides = _build_overrides(args)
    config: Config = load_config(_CONFIG_PY, overrides, enable_one_logger=False)
    config.validate()
    config.freeze()  # type: ignore

    with distributed_init():
        distributed.init()

    trainer = config.trainer.type(config)

    with model_init():
        model = instantiate(config.model)

    with data_loader_init():
        dataset = instantiate(config.dataloader_train.dataset)
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            seed=0,
        )
        dataloader_train = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=config.dataloader_train.drop_last,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
            pin_memory_device=config.dataloader_train.pin_memory_device,
            timeout=config.dataloader_train.timeout,
        )

    import torch

    model = model.to("cuda", memory_format=config.trainer.memory_format)  # type: ignore
    model.on_train_start(config.trainer.memory_format)

    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    iteration = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    logging.info(f"Loaded checkpoint at iteration {iteration} from {config.checkpoint.load_path}")

    if model.ewc is None:
        raise RuntimeError(
            "model.ewc is None -- ewc_enabled override didn't take effect. "
            "Check that your config exposes model.config.* through Hydra."
        )
    # Load after checkpoint so Fisher / theta* are re-sharded to match ``net``.
    model.ewc.load(
        args.ewc_state_path,
        net=model.net,
        override_lambda=1.0,
        override_gamma=1.0,
    )
    model.ewc.to("cuda")
    if not model.ewc.has_state():
        raise RuntimeError(
            f"EWC state failed to load from {args.ewc_state_path}. "
            "Run compute_ewc_fisher.py first."
        )

    sampler.set_epoch(0)
    loader_iter = iter(dataloader_train)

    # The model's parameter dtype (typically torch.bfloat16). We cast each
    # batch's float tensors to this so the forward dtype-checks pass; this
    # mirrors what the trainer's LowPrecisionCallback does at every step.
    model_precision = getattr(model, "precision", None)

    # ------------------------------------------------------------------
    # Calibration loop
    # ------------------------------------------------------------------
    n_steps = args.n_calibration_steps
    losses_global: List[float] = []
    qs_global: List[float] = []

    if distributed.is_rank0():
        logging.info(f"[calibrate-ewc] running {n_steps} probe steps with ewc_lambda=1.0")

    model.train()
    for step in range(n_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            sampler.set_epoch(sampler.epoch + 1 if hasattr(sampler, "epoch") else 1)
            loader_iter = iter(dataloader_train)
            batch = next(loader_iter)

        # Mirror what the trainer's LowPrecisionCallback.on_training_step_start
        # does -- cast float-point tensors to model.precision (typically bf16)
        # before the forward, otherwise float32 inputs meet bf16 weights and
        # F.linear raises a dtype mismatch.
        batch = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if model_precision is not None and model_precision != torch.float32:
            batch = {
                k: (v.to(dtype=model_precision)
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                    else v)
                for k, v in batch.items()
            }

        optimizer.zero_grad(set_to_none=True)
        output_batch, total_loss = model.training_step(batch, step)

        # output_batch["ewc_penalty"] is the local-shard quadratic with lambda=1.
        # total_loss = kendall_loss_local + ewc_penalty_local.
        ewc_penalty_local = float(output_batch["ewc_penalty"].detach().item())
        total_loss_local = float(total_loss.detach().item())
        kendall_loss_local = total_loss_local - ewc_penalty_local

        # Reduce to global. Q is partitioned across ranks (sum); kendall is a
        # per-rank sample of the population mean.
        q_global = _all_reduce_scalar(ewc_penalty_local, "sum")
        l_global = _all_reduce_scalar(kendall_loss_local, "mean")
        losses_global.append(l_global)
        qs_global.append(q_global)

        grad_scaler.scale(total_loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        if distributed.is_rank0() and step % max(1, n_steps // 10) == 0:
            logging.info(
                f"[calibrate-ewc] step={step:4d}  L_task={l_global:.4f}  Q={q_global:.4g}  "
                f"L_task/Q={(l_global / q_global) if q_global > 0 else float('inf'):.4g}"
            )

    # ------------------------------------------------------------------
    # Recommend lambda values
    # ------------------------------------------------------------------
    # Skip the first half: that's the warmup phase where theta is still close
    # to theta* and Q is unrepresentatively small. The tail is what an actual
    # CL stage 1 run will look like for most of training.
    n_skip = n_steps // 2
    tail_l = losses_global[n_skip:]
    tail_q = qs_global[n_skip:]
    if not tail_q or sum(tail_q) <= 0:
        raise RuntimeError(
            "Q remained zero throughout calibration -- this should not happen. "
            "Check that the EWC state was loaded and that the optimizer is actually "
            "stepping (try a higher learning rate or more steps)."
        )

    avg_l = sum(tail_l) / len(tail_l)
    avg_q = sum(tail_q) / len(tail_q)
    ratio = avg_l / avg_q

    if distributed.is_rank0():
        logging.info("=" * 72)
        logging.info("[calibrate-ewc] CALIBRATION SUMMARY")
        logging.info("=" * 72)
        logging.info(f"  averaged over the last {len(tail_l)} of {n_steps} steps:")
        logging.info(f"    L_task (mean over DP ranks)         = {avg_l:.6g}")
        logging.info(f"    Q       (sum over DP ranks, lam=1)  = {avg_q:.6g}")
        logging.info(f"    L_task / Q                          = {ratio:.6g}")
        logging.info("")
        logging.info("  recommended ewc_lambda for target ratio alpha = lambda * Q / L_task:")
        for alpha in args.alpha_targets:
            lam = alpha * ratio
            logging.info(f"    alpha = {alpha:>5}  =>  ewc_lambda = {lam:>12.4g}")
        logging.info("=" * 72)
        logging.info(
            "  Pick the row whose alpha matches your forgetting/plasticity tradeoff:\n"
            "    alpha << 1     soft anchor (favours new task, mild EWC)\n"
            "    alpha  ~ 1     balanced (EWC penalty matches task loss in magnitude)\n"
            "    alpha >> 1     stiff anchor (heavy EWC, slow new-task learning)\n"
            "  Then pass model.config.ewc_lambda=<value> to your CL stage train command."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirically calibrate ewc_lambda for the Cosmos Policy "
        "by measuring L_task / Q on a short probe run.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Name of a registered Cosmos Policy experiment config -- this should "
        "be the *next* CL stage (e.g. cosmos_predict2_2b_480p_libero_goal_single_task_cl_stage), "
        "so the dataset and batch shape match what training will actually see.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to the base-stage checkpoint to start from. If omitted, the "
        "experiment's checkpoint.load_path is used.",
    )
    parser.add_argument(
        "--ewc_state_path",
        required=True,
        help="EWC checkpoint from compute_ewc_fisher: DCP directory (no .pt suffix), "
        "e.g. ./ewc_states/libero_goal/base_stage_iter7000, or legacy .pt file.",
    )
    parser.add_argument(
        "--n_calibration_steps",
        type=int,
        default=100,
        help="Number of probe training steps to run. 50-200 is plenty; the "
        "ratio L_task/Q stabilises quickly once theta has drifted from theta*.",
    )
    parser.add_argument(
        "--alpha_targets",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
        help="List of alpha values to recommend ewc_lambda for. "
        "alpha is the target ratio L_ewc / L_task at convergence.",
    )
    parser.add_argument(
        "--extra_override",
        action="append",
        default=None,
        help="Optional extra config override in 'key=value' form (repeatable). "
        "Useful for tweaks like dataloader_train.batch_size=8. "
        "You do NOT need to pass 'experiment=...' here -- use --experiment.",
    )
    args = parser.parse_args()

    try:
        launch(args)
    except Exception:  # pragma: no cover -- top-level error logging
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Usage:
    #   uv run --extra cu128 --group libero --python 3.10 \
    #     torchrun --nproc_per_node=8 --master_port=12341 \
    #     -m cosmos_policy.scripts.calibrate_ewc_lambda \
    #     --experiment <next_stage_experiment_name> \
    #     --ckpt <base_stage_ckpt> \
    #     --ewc_state_path <ewc_state.pt>
    main()
