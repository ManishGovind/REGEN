# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Offline Fisher diagonal computation for online EWC.

This script is the standalone counterpart to the trainer hook in
``cosmos_policy/trainer.py``. It loads a checkpoint, builds the same
data loader the model trained on, and writes ``fisher.pt`` plus ``theta*.pt``
to a **DCP directory** (``meta.pt`` + ``ewc_state/`` shards) that subsequent CL
stages load via ``model.config.ewc_prev_state_path``. Use ``--format dcp``
(default) and a directory path **without** a ``.pt`` suffix.

Typical usage (CL stage 0 -> stage 1):

    uv run --extra cu128 --group libero --python 3.10 \\
        torchrun --nproc_per_node=8 --master_port=12341 \\
        -m cosmos_policy.scripts.compute_ewc_fisher \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_k_1 \\
        --ckpt /workspace/checkpoints/.../iter_000040000 \\
        --output_path /workspace/ewc_states/libero_goal/base_stage_iter7000 \\
        --format dcp \\
        --num_batches 50 \\
        --fisher_dtype auto

You only specify the experiment **name** -- the script internally loads
``cosmos_policy/config/config.py`` (the same config router used by
``cosmos_policy.scripts.train``) and applies the experiment + EWC overrides
itself. No Hydra ``--`` separator or ``key=value`` syntax is required on
the command line.

The ``--ewc_lambda`` / ``--ewc_gamma`` flags are written to the file as metadata
only -- they don't affect Fisher estimation. The values that actually control
training are set on the next stage's launch via
``model.config.ewc_lambda`` / ``model.config.ewc_gamma``.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from loguru import logger as logging
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.config import Config, load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_policy.continual.ewc import _DCP_META_FILE, _DCP_TENSOR_SUBDIR

# Same config router that cosmos_policy.scripts.train uses. We hardcode it so
# users only need to pass --experiment <name> on the CLI.
_CONFIG_PY = "cosmos_policy/config/config.py"


def _normalize_output_path(output_path: str, fmt: str) -> str:
    """Return a path suitable for ``OnlineEWC.save()`` (DCP dir or legacy ``.pt``)."""
    path = output_path.rstrip("/")
    if fmt == "dcp":
        if path.endswith(".pt"):
            raise ValueError(
                "DCP format requires a directory path without a .pt suffix. "
                f"Got: {output_path!r}. Example: ./ewc_states/libero_goal/base_stage_iter7000"
            )
        return path
    if not path.endswith(".pt"):
        return f"{path}.pt"
    return path


def _verify_dcp_checkpoint(dir_path: str) -> None:
    meta = os.path.join(dir_path, _DCP_META_FILE)
    shards = os.path.join(dir_path, _DCP_TENSOR_SUBDIR)
    if not os.path.isfile(meta) or not os.path.isdir(shards):
        raise RuntimeError(
            f"[EWC] DCP checkpoint incomplete at {dir_path} "
            f"(expected {meta} and {shards}/)."
        )


def _build_overrides(args: argparse.Namespace) -> list[str]:
    """Build the Hydra override list for load_config().

    The first element MUST be ``"--"`` (load_config / Hydra's contract).
    Subsequent elements are standard ``key=value`` overrides.
    """
    overrides: list[str] = ["--"]
    overrides.append(f"experiment={args.experiment}")
    overrides.append("model.config.ewc_enabled=true")
    overrides.append(f"model.config.ewc_lambda={args.ewc_lambda}")
    overrides.append(f"model.config.ewc_gamma={args.ewc_gamma}")
    overrides.append(f"model.config.ewc_fisher_dtype={args.fisher_dtype}")
    overrides.append(f"model.config.ewc_num_fisher_batches={args.num_batches}")
    overrides.append(f"model.config.ewc_save_state_path={args.output_path}")
    if args.ckpt:
        overrides.append(f"checkpoint.load_path={args.ckpt}")
    # Don't accidentally resume the optimiser / scheduler -- we only want weights.
    overrides.append("checkpoint.load_training_state=false")
    overrides.append("checkpoint.strict_resume=false")
    # Forward any user-supplied extra overrides verbatim.
    for extra in args.extra_override or []:
        overrides.append(extra)
    return overrides


@logging.catch(reraise=True)
def launch(args: argparse.Namespace) -> None:
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

    # Standard model bring-up (loads the checkpoint into model.net).
    model = model.to("cuda", memory_format=config.trainer.memory_format)  # type: ignore
    model.on_train_start(config.trainer.memory_format)

    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    import torch

    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    iteration = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    logging.info(f"Loaded checkpoint at iteration {iteration} from {config.checkpoint.load_path}")

    if model.ewc is None:
        raise RuntimeError(
            "model.ewc is None -- ewc_enabled override didn't take effect. "
            "Check that your config exposes model.config.* through Hydra."
        )

    # Make sure the sampler is initialised so the loader produces fresh batches.
    sampler.set_epoch(0)

    output_path = _normalize_output_path(args.output_path, args.format)

    model.ewc.compute_fisher(
        model=model,
        net=model.net,
        data_loader=dataloader_train,
        num_batches=args.num_batches,
    )
    model.ewc.save(output_path)

    if args.format == "dcp" and distributed.is_rank0():
        _verify_dcp_checkpoint(output_path)
        logging.success(
            f"[EWC] DCP checkpoint ready at {output_path}/ "
            f"({_DCP_META_FILE} + {_DCP_TENSOR_SUBDIR}/). "
            f"Use model.config.ewc_prev_state_path={output_path} for the next CL stage."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and save an online-EWC Fisher diagonal for the Cosmos Policy.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Name of a registered Cosmos Policy experiment config "
        "(e.g. cosmos_predict2_2b_480p_libero_goal_base_stage_k_1). "
        "This should be the experiment the checkpoint was trained on.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to the checkpoint to load weights from (an iter_XXXXXXX directory). "
        "If omitted, the experiment's checkpoint.load_path is used.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path. For --format dcp (default): a directory without a .pt suffix, "
        "e.g. ./ewc_states/libero_goal/base_stage_iter7000. For --format pt: base path "
        "for a single legacy .pt file. Pass the same path to the next CL stage as "
        "model.config.ewc_prev_state_path.",
    )
    parser.add_argument(
        "--format",
        choices=("dcp", "pt"),
        default="dcp",
        help="Checkpoint format. 'dcp' (default) writes meta.pt + ewc_state/ shards "
        "(recommended; same GPU count / fsdp_shard_size as training). 'pt' writes one "
        "gathered legacy file (only if you need portability across GPU counts).",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Number of batches for the Fisher estimate (mirrors openpi's ewc_max_batches=50).",
    )
    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=50000.0,
        help="EWC lambda metadata (mirrors openpi's TrainConfig default; "
        "not used during Fisher estimation, only stored in the file).",
    )
    parser.add_argument(
        "--ewc_gamma",
        type=float,
        default=0.9,
        help="Online EWC decay metadata (mirrors openpi's TrainConfig default; "
        "not used on first task).",
    )
    parser.add_argument(
        "--fisher_dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Storage dtype for Fisher / theta* tensors. 'auto' (default) "
        "matches the base-stage parameter dtype.",
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
    # Usage (DCP, recommended):
    #   uv run --extra cu128 --group libero --python 3.10 \
    #     torchrun --nproc_per_node=4 --master_port=12341 \
    #     -m cosmos_policy.scripts.compute_ewc_fisher \
    #     --experiment <experiment_name> \
    #     --ckpt <iter_000007000> \
    #     --output_path ./ewc_states/libero_goal/base_stage_iter7000 \
    #     --format dcp
    main()
