# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Export a DCP-format EWC state directory to a single legacy ``.pt`` file.

Use this if you need a portable gathered checkpoint (e.g. for inspection or
tools that only read ``.pt``). Training can load either format automatically.

Example::

    uv run --extra cu128 --group libero --python 3.10 \\
        torchrun --nproc_per_node=4 --master_port=12341 \\
        -m cosmos_policy.scripts.export_ewc_state_to_pt \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_k_1 \\
        --ckpt ./checkpoints/.../iter_000007000 \\
        --dcp_dir ./ewc_states/libero_goal/base_stage_iter7000 \\
        --output_path ./ewc_states/libero_goal/base_stage_iter7000.pt
"""

from __future__ import annotations

import argparse
import sys
import traceback

from loguru import logger as logging
from megatron.core import parallel_state

from cosmos_policy._src.imaginaire.config import Config, load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init

_CONFIG_PY = "cosmos_policy/config/config.py"


def _build_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = ["--"]
    overrides.append(f"experiment={args.experiment}")
    overrides.append("model.config.ewc_enabled=true")
    overrides.append("model.config.ewc_prev_state_path=")
    if args.ckpt:
        overrides.append(f"checkpoint.load_path={args.ckpt}")
    overrides.append("checkpoint.load_training_state=false")
    overrides.append("checkpoint.strict_resume=false")
    return overrides


@logging.catch(reraise=True)
def launch(args: argparse.Namespace) -> None:
    config: Config = load_config(_CONFIG_PY, _build_overrides(args), enable_one_logger=False)
    config.validate()
    config.freeze()  # type: ignore

    with distributed_init():
        from cosmos_policy._src.imaginaire.utils import distributed

        distributed.init()

    trainer = config.trainer.type(config)

    with model_init():
        model = instantiate(config.model)

    model = model.to("cuda", memory_format=config.trainer.memory_format)  # type: ignore
    model.on_train_start(config.trainer.memory_format)

    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    import torch

    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)

    if model.ewc is None:
        raise RuntimeError("model.ewc is None -- enable ewc in the experiment config.")

    model.ewc.load(args.dcp_dir, net=model.net)
    model.ewc._save_pt(args.output_path)
    logging.info(f"Wrote legacy .pt export to {args.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DCP EWC state to a single .pt file.")
    parser.add_argument("--experiment", required=True, help="Experiment used for the EWC state.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (same as when Fisher was computed).")
    parser.add_argument(
        "--dcp_dir",
        required=True,
        help="DCP EWC directory (contains meta.pt and ewc_state/).",
    )
    parser.add_argument("--output_path", required=True, help="Output .pt file path.")
    args = parser.parse_args()

    try:
        launch(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
