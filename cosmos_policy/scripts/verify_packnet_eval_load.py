# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Verify PackNet eval load: checkpoint + masks + per-task apply_eval_mask invariants.

Mirrors ``get_model`` + ``apply_packnet_eval_for_libero_task`` in ``cosmos_utils.py``.

Checks after masking for each LIBERO task:
  - weights at mask==0 are exactly zero
  - weights at mask > max_label are exactly zero
  - weights at allowed labels (1..eval_task_id+1) match the loaded checkpoint backup

Example::

    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.verify_packnet_eval_load \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \\
        --ckpt /path/to/iter_000002000/model.pt \\
        --packnet_path /workspace/packnet_states/libero_object/cl_stage1_pre_prune \\
        --libero_tasks 0,6,7 \\
        --first_cl_libero_task 6
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional

import torch

from cosmos_policy.continual.ewc import _full_tensor
from cosmos_policy.continual.packnet import PackNet, iter_prunable_weight_params
from cosmos_policy.experiments.robot.cosmos_utils import (
    apply_packnet_eval_for_libero_task,
    load_packnet_eval_state,
    packnet_eval_task_id_for_libero_task,
)

_CONFIG_PY = "cosmos_policy/config/config.py"


def _check_masked_weights(
    backup: torch.Tensor,
    masked: torch.Tensor,
    mask: torch.Tensor,
    eval_task_id: int,
    *,
    atol: float = 0.0,
) -> dict:
    m = mask.to(torch.uint8)
    max_label = eval_task_id + 1
    allowed = (m >= 1) & (m <= max_label)

    zeros_ok = bool((masked[m == 0].abs() <= atol).all().item()) if (m == 0).any() else True
    future_ok = bool((masked[m > max_label].abs() <= atol).all().item()) if (m > max_label).any() else True

    if allowed.any():
        backup_allowed = backup[allowed]
        masked_allowed = masked[allowed]
        preserved_ok = torch.allclose(backup_allowed, masked_allowed, atol=atol, rtol=0.0)
        allowed_nz_frac = float((masked_allowed.abs() > atol).float().mean().item())
    else:
        preserved_ok = True
        allowed_nz_frac = 0.0

    return {
        "zeros_ok": zeros_ok,
        "future_ok": future_ok,
        "preserved_ok": preserved_ok,
        "allowed_count": int(allowed.sum().item()),
        "allowed_nz_frac": allowed_nz_frac,
        "zero_mask_count": int((m == 0).sum().item()),
        "future_mask_count": int((m > max_label).sum().item()),
    }


def _aggregate_layer_checks(
    packnet: PackNet,
    backup: Dict[str, torch.Tensor],
    net,
    eval_task_id: int,
    *,
    atol: float,
) -> dict:
    name_to_param = dict(net.named_parameters())
    totals = {
        "layers": 0,
        "zeros_ok": 0,
        "future_ok": 0,
        "preserved_ok": 0,
        "allowed_count": 0,
        "allowed_nz": 0,
    }
    bad_layers: List[str] = []

    for name, mask_t in packnet.masks.items():
        if name not in backup or name not in name_to_param:
            continue
        full_w = _full_tensor(name_to_param[name].data).cpu().float()
        full_m = _full_tensor(mask_t).cpu().to(torch.uint8)
        full_b = backup[name].float()
        if full_w.shape != full_m.shape:
            bad_layers.append(f"{name}: shape mismatch w={tuple(full_w.shape)} m={tuple(full_m.shape)}")
            continue

        stats = _check_masked_weights(full_b, full_w, full_m, eval_task_id, atol=atol)
        totals["layers"] += 1
        totals["zeros_ok"] += int(stats["zeros_ok"])
        totals["future_ok"] += int(stats["future_ok"])
        totals["preserved_ok"] += int(stats["preserved_ok"])
        totals["allowed_count"] += stats["allowed_count"]
        totals["allowed_nz"] += int(stats["allowed_nz_frac"] * stats["allowed_count"])

        if not (stats["zeros_ok"] and stats["future_ok"] and stats["preserved_ok"]):
            bad_layers.append(
                f"{name}: zeros_ok={stats['zeros_ok']} future_ok={stats['future_ok']} "
                f"preserved_ok={stats['preserved_ok']}"
            )

    totals["bad_layers"] = bad_layers
    return totals


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Verify PackNet eval checkpoint + mask application.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ckpt", required=True, help="Eval checkpoint (e.g. iter_2000).")
    parser.add_argument("--packnet_path", required=True, help="PackNet mask directory or .pt.")
    parser.add_argument(
        "--libero_tasks",
        default="0,6",
        help="Comma-separated LIBERO task ids to verify (default 0,6).",
    )
    parser.add_argument("--first_cl_libero_task", type=int, default=6)
    parser.add_argument("--config_file", default=_CONFIG_PY)
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for zero / preserve checks.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    libero_tasks = [int(x.strip()) for x in args.libero_tasks.split(",") if x.strip()]

    print(f"Loading checkpoint: {args.ckpt}")
    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.ckpt,
        config_file=args.config_file,
        load_ema_to_reg=False,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Same path as run_libero_eval get_model()
    eval_state = load_packnet_eval_state(model, args.packnet_path)
    packnet = eval_state.packnet

    n_prunable = len(list(iter_prunable_weight_params(model.net)))
    n_mask_layers = len(packnet.masks)
    print(f"Prunable layers in model: {n_prunable}")
    print(f"Mask tensors loaded:      {n_mask_layers}")
    if n_mask_layers == 0:
        print("FAIL: no masks loaded.")
        sys.exit(1)

    backup_nonzero = 0
    backup_total = 0
    for name, full_b in eval_state.weight_backup.items():
        backup_total += full_b.numel()
        backup_nonzero += int((full_b.abs() > args.atol).sum().item())
    print(
        f"Checkpoint backup: {backup_nonzero:,}/{backup_total:,} prunable elements "
        f"non-zero ({100*backup_nonzero/max(backup_total,1):.1f}%)"
    )
    print()

    all_pass = True
    for libero_task in libero_tasks:
        eval_task_id = apply_packnet_eval_for_libero_task(
            model,
            eval_state,
            libero_task,
            first_cl_libero_task=args.first_cl_libero_task,
        )
        expected = packnet_eval_task_id_for_libero_task(libero_task, args.first_cl_libero_task)
        assert eval_task_id == expected

        totals = _aggregate_layer_checks(
            packnet, eval_state.weight_backup, model.net, eval_task_id, atol=args.atol
        )
        nz_pct = 100.0 * totals["allowed_nz"] / max(totals["allowed_count"], 1)

        print(f"LIBERO task {libero_task} → eval_task_id={eval_task_id} (max_label={eval_task_id + 1})")
        print(
            f"  layers OK: zeros={totals['zeros_ok']}/{totals['layers']} "
            f"future_zeroed={totals['future_ok']}/{totals['layers']} "
            f"preserved={totals['preserved_ok']}/{totals['layers']}"
        )
        print(
            f"  active subnet: {totals['allowed_count']:,} weights, "
            f"{nz_pct:.1f}% non-zero (after mask)"
        )

        if totals["bad_layers"]:
            all_pass = False
            print(f"  FAIL ({len(totals['bad_layers'])} layers):")
            for line in totals["bad_layers"][:5]:
                print(f"    {line}")
            if len(totals["bad_layers"]) > 5:
                print(f"    ... +{len(totals['bad_layers']) - 5} more")
        else:
            print("  PASS: mask rules satisfied for this task.")
        print()

    if all_pass:
        print("Overall: PASS — eval load + PackNet masking look correct for all tasks.")
    else:
        print("Overall: FAIL — see layer errors above.")
        print("Common causes:")
        print("  - iter_2000 ckpt with post-prune masks (use cl_stage1_pre_prune)")
        print("  - wrong packnet_path or checkpoint architecture mismatch")
        sys.exit(1)


if __name__ == "__main__":
    main()
