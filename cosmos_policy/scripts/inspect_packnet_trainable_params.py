# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
List ``requires_grad`` and PackNet **effective** trainability per parameter.

PackNet does **not** set ``requires_grad=False``. During training it calls
``zero_grads()`` after backward so only the current task's mask label gets
optimizer updates. Biases and norm layers always get zero grad.

Use this script to see:
  - ``requires_grad`` flag on each ``named_parameter()``
  - For Conv/Linear **weights**: fraction of elements with mask == trainable label
  - For biases / norms: marked as "PackNet grad always zero"

Example (main CL iter_2000, CL stage 1 → train label 2)::

    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.inspect_packnet_trainable_params \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \\
        --ckpt /path/to/iter_000002000/model.pt \\
        --packnet_path /workspace/packnet_states/libero_object/cl_stage1_pre_prune \\
        --trainable_label 2
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn

from cosmos_policy.continual.ewc import _full_tensor
from cosmos_policy.continual.packnet import PackNet, _is_norm_module, iter_prunable_weight_params

_CONFIG_PY = "cosmos_policy/config/config.py"


def _is_prunable_weight_name(name: str, net: nn.Module) -> bool:
    return name in {n for n, _ in iter_prunable_weight_params(net)}


def _effective_trainable_pct(mask: torch.Tensor, trainable_label: int) -> float:
    m = _full_tensor(mask).to(torch.uint8)
    if m.numel() == 0:
        return 0.0
    return 100.0 * float((m == trainable_label).sum().item()) / m.numel()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="List requires_grad and PackNet effective trainability per parameter."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint to inspect (e.g. iter_2000).")
    parser.add_argument("--packnet_path", required=True, help="Pre-prune PackNet masks.")
    parser.add_argument("--config_file", default=_CONFIG_PY)
    parser.add_argument(
        "--trainable_label",
        type=int,
        default=0,
        help="Mask label that receives grad during the phase you care about "
        "(0 = infer from packnet task_id + 1). CL stage 1 main CL → 2.",
    )
    parser.add_argument(
        "--only_trainable",
        action="store_true",
        help="Only print parameters with some trainable elements or requires_grad issues.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Max parameter rows (0 = all).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.ckpt,
        config_file=args.config_file,
        load_ema_to_reg=False,
    )
    model.eval()

    packnet = PackNet()
    packnet.load(args.packnet_path, net=model.net)
    trainable_label = args.trainable_label or (packnet.task_id + 1)

    prunable_names = {n for n, _ in iter_prunable_weight_params(model.net)}
    norm_param_prefixes: set[str] = set()
    for module_name, module in model.net.named_modules():
        if _is_norm_module(module):
            prefix = f"{module_name}." if module_name else ""
            for pname, _ in module.named_parameters(recurse=False):
                norm_param_prefixes.add(f"{prefix}{pname}")

    print(f"Checkpoint:       {args.ckpt}")
    print(f"PackNet masks:    {args.packnet_path}")
    print(f"packnet.task_id:  {packnet.task_id}  → trainable mask label {trainable_label} for main CL")
    print()
    print(
        "NOTE: PackNet keeps requires_grad=True on most params but zeros grads in "
        "on_after_backward(). 'Effective train' = mask label that actually updates."
    )
    print()
    print(f"{'parameter':<72} {'req_grad':>8} {'effective':>20}")
    print("-" * 102)

    n_req_true = 0
    n_req_false = 0
    n_packnet_partial = 0
    n_packnet_frozen = 0
    n_packnet_full = 0
    rows = 0

    for name, param in model.net.named_parameters():
        req = param.requires_grad
        if req:
            n_req_true += 1
        else:
            n_req_false += 1

        if name in prunable_names:
            mask_t = packnet.masks.get(name)
            if mask_t is None:
                effective = "no PackNet mask"
            else:
                pct = _effective_trainable_pct(mask_t, trainable_label)
                if pct <= 0.0:
                    effective = f"frozen (label≠{trainable_label})"
                    n_packnet_frozen += 1
                elif pct >= 100.0:
                    effective = f"100% label {trainable_label}"
                    n_packnet_full += 1
                else:
                    effective = f"{pct:.1f}% label {trainable_label}"
                    n_packnet_partial += 1
        elif name in norm_param_prefixes or name.endswith(".bias"):
            effective = "PackNet: grad always 0"
            n_packnet_frozen += 1
        else:
            effective = "not PackNet-pruned"

        if args.only_trainable and effective.startswith("frozen") and req:
            continue
        if args.only_trainable and effective == "PackNet: grad always 0":
            continue

        print(f"{name:<72} {str(req):>8} {effective:>20}")
        rows += 1
        if args.max_rows and rows >= args.max_rows:
            print("... (truncated)")
            break

    print("-" * 102)
    print(f"requires_grad=True:  {n_req_true}")
    print(f"requires_grad=False: {n_req_false}")
    print(f"Prunable weight tensors: partial={n_packnet_partial}, all-frozen={n_packnet_frozen}, "
          f"100%-train-label={n_packnet_full}")
    print()
    print("Main CL stage 1 summary:")
    print(f"  TRAINED in main CL  → Conv/Linear elements with mask label {trainable_label} (~75%)")
    print(f"  FROZEN in main CL   → mask label 1 (~25% base) + label 0 + all biases/norms")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
