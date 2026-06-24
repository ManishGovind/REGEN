# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Reconstruct and export pre-end-prune PackNet masks for LIBERO eval.

Use when a CL stage finished before ``packnet_save_pre_prune_state_path`` was
added: post-prune masks (``cl_stage1/``) do not match the ``iter_2000`` training
checkpoint for the new task. This script rebuilds pre-prune masks by marking
end-pruned slots (post mask==0) that are still non-zero in the training ckpt as
the current CL task label.

Example (CL stage 1, task 6)::

    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.export_packnet_pre_prune_masks \\
        --post_prune_masks /workspace/packnet_states/libero_goal/cl_stage1 \\
        --output /workspace/packnet_states/libero_goal/cl_stage1_pre_prune \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \\
        --ckpt /path/to/packnet_cl_stage1/checkpoints/iter_000002000/model.pt

Then eval with ``--packnet_mask_path`` pointing at ``--output``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch

from cosmos_policy.continual.ewc import _full_tensor, _reshard_full_to_ref
from cosmos_policy.continual.packnet import (
    PackNet,
    _is_legacy_pt_path,
    iter_prunable_weight_params,
    reconstruct_pre_prune_masks_from_post_prune,
)

_CONFIG_PY = "cosmos_policy/config/config.py"


def _load_legacy_pt_masks(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "prune_perc": float(payload.get("prune_perc", 0.25)),
        "task_id": int(payload.get("task_id", 0)),
        "masks": {k: v.to(torch.uint8) for k, v in payload["masks"].items()},
    }


def _label_counts(mask: torch.Tensor) -> Dict[int, int]:
    m = mask.to(torch.uint8)
    return {int(v): int((m == v).sum().item()) for v in m.unique(sorted=True)}


def _print_mask_delta(
    post: Dict[str, torch.Tensor],
    pre: Dict[str, torch.Tensor],
    current_task_label: int,
) -> None:
    post_agg: Dict[int, int] = defaultdict(int)
    pre_agg: Dict[int, int] = defaultdict(int)
    for name in post:
        if name not in pre:
            continue
        for label, c in _label_counts(post[name]).items():
            post_agg[label] += c
        for label, c in _label_counts(pre[name]).items():
            pre_agg[label] += c

    print("Post-prune aggregate labels:", dict(sorted(post_agg.items())))
    print("Pre-prune  aggregate labels:", dict(sorted(pre_agg.items())))
    restored = pre_agg.get(current_task_label, 0) - post_agg.get(current_task_label, 0)
    print(
        f"Restored {restored:,} weights: post label 0 → pre label {current_task_label} "
        f"(non-zero in training ckpt)"
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export pre-end-prune PackNet masks from post-prune masks + training ckpt."
    )
    parser.add_argument(
        "--post_prune_masks",
        required=True,
        help="Post-prune PackNet state (DCP dir or legacy .pt), e.g. cl_stage1/",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path: DCP directory or legacy .pt for pre-end-prune masks.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Inference experiment name (model architecture for ckpt + DCP shard layout).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Last main-training checkpoint (e.g. iter_000002000/model.pt), before end prune.",
    )
    parser.add_argument(
        "--config_file",
        default=_CONFIG_PY,
        help="Config module path.",
    )
    parser.add_argument(
        "--weight_epsilon",
        type=float,
        default=0.0,
        help="Treat |w| <= epsilon as zero when restoring pruned current-task slots.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print reconstruction stats only; do not write output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    print(f"Loading model from {args.ckpt} ...")
    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.ckpt,
        config_file=args.config_file,
        load_ema_to_reg=False,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    prunable_weights = {
        name: _full_tensor(param.data).detach().cpu()
        for name, param in iter_prunable_weight_params(model.net)
    }

    post_path = args.post_prune_masks.rstrip("/")
    if os.path.isfile(post_path) and _is_legacy_pt_path(post_path):
        legacy = _load_legacy_pt_masks(post_path)
        post_masks_cpu = legacy["masks"]
        prune_perc = legacy["prune_perc"]
        task_id = legacy["task_id"]
    else:
        packnet_post = PackNet()
        packnet_post.load(post_path, net=model.net)
        post_masks_cpu = {
            name: _full_tensor(t).detach().cpu().to(torch.uint8) for name, t in packnet_post.masks.items()
        }
        prune_perc = packnet_post.prune_perc
        task_id = packnet_post.task_id

    current_task_label = task_id + 1
    print(
        f"Post-prune state: task_id={task_id}, prune_perc={prune_perc}, "
        f"current_task_label={current_task_label}"
    )

    pre_masks_cpu = reconstruct_pre_prune_masks_from_post_prune(
        post_masks_cpu,
        prunable_weights,
        current_task_label=current_task_label,
        weight_epsilon=args.weight_epsilon,
    )
    _print_mask_delta(post_masks_cpu, pre_masks_cpu, current_task_label)

    if args.dry_run:
        print("Dry run — no output written.")
        return

    name_to_param = dict(model.net.named_parameters())
    packnet_out = PackNet(prune_perc=prune_perc, task_id=task_id)
    packnet_out.masks = {
        name: _reshard_full_to_ref(pre_masks_cpu[name], name_to_param[name], torch.uint8)
        for name in pre_masks_cpu
        if name in name_to_param
    }
    packnet_out.is_initialized = True

    output = args.output.rstrip("/")
    packnet_out.save(output)
    print(f"Saved pre-end-prune masks to {output}")
    print("Use with LIBERO eval:")
    print(f"  --packnet_mask_path {output}")
    print(f"  --ckpt_path {args.ckpt}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
