# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Check which weights changed after main CL training using PackNet mask labels.

Use the **pre-end-prune** masks (``cl_stage1_pre_prune/``) with the **iter_2000**
checkpoint — they describe the subnet at the end of main CL.

**Best check (label 1 frozen during main CL):** compare ``--ckpt`` (iter_2000) to a
snapshot taken **after base post-prune retrain, before main CL** (if you saved one).

**Fallback:** compare to the loaded **base** checkpoint (``iter_7000``). Label 2
should change a lot; label 1 changed mostly from base retrain, not main CL.
Biases / norm layers should be unchanged vs base if PackNet grad masking worked.

Example::

    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.inspect_packnet_trained_weights \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \\
        --ckpt /path/to/packnet_cl_stage1/checkpoints/iter_000002000/model.pt \\
        --packnet_path /workspace/packnet_states/libero_object/cl_stage1_pre_prune \\
        --reference_ckpt /path/to/libero_object_base_stage/checkpoints/iter_000010000/model.pt \\
        --delta_threshold 1e-6
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from cosmos_policy.continual.ewc import _full_tensor
from cosmos_policy.continual.packnet import PackNet, iter_prunable_weight_params

_CONFIG_PY = "cosmos_policy/config/config.py"


def _load_prunable_weights(experiment: str, ckpt: str, config_file: str) -> Dict[str, torch.Tensor]:
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(
        experiment_name=experiment,
        s3_checkpoint_dir=ckpt,
        config_file=config_file,
        load_ema_to_reg=False,
    )
    model.eval()
    return {
        name: _full_tensor(param.data).detach().cpu().float()
        for name, param in iter_prunable_weight_params(model.net)
    }


def _load_masks(packnet_path: str, experiment: str, ckpt: str, config_file: str) -> Tuple[PackNet, Dict[str, torch.Tensor]]:
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(
        experiment_name=experiment,
        s3_checkpoint_dir=ckpt,
        config_file=config_file,
        load_ema_to_reg=False,
    )
    packnet = PackNet()
    packnet.load(packnet_path, net=model.net)
    masks = {name: _full_tensor(t).detach().cpu().to(torch.uint8) for name, t in packnet.masks.items()}
    return packnet, masks


def _load_non_prunable_weights(experiment: str, ckpt: str, config_file: str) -> Dict[str, torch.Tensor]:
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(
        experiment_name=experiment,
        s3_checkpoint_dir=ckpt,
        config_file=config_file,
        load_ema_to_reg=False,
    )
    prunable = {n for n, _ in iter_prunable_weight_params(model.net)}
    out: Dict[str, torch.Tensor] = {}
    for name, param in model.net.named_parameters():
        if name in prunable:
            continue
        if "weight" in name or "bias" in name:
            out[name] = _full_tensor(param.data).detach().cpu().float()
    return out


def _label_counts(mask: torch.Tensor) -> Dict[int, int]:
    m = mask.to(torch.uint8)
    return {int(v): int((m == v).sum().item()) for v in m.unique(sorted=True)}


def _stats_for_label(w: torch.Tensor, m: torch.Tensor, label: int) -> dict:
    sel = m == label
    n = int(sel.sum().item())
    if n == 0:
        return {"count": 0, "nonzero_frac": 0.0, "mean_abs": 0.0, "max_abs": 0.0}
    ws = w[sel]
    return {
        "count": n,
        "nonzero_frac": float((ws.abs() > 0).float().mean().item()),
        "mean_abs": float(ws.abs().mean().item()),
        "max_abs": float(ws.abs().max().item()),
    }


def _delta_stats(w_new: torch.Tensor, w_ref: torch.Tensor, m: torch.Tensor, label: int, threshold: float) -> dict:
    sel = m == label
    n = int(sel.sum().item())
    if n == 0:
        return {"count": 0, "changed_frac": 0.0, "mean_abs_delta": 0.0, "max_abs_delta": 0.0}
    delta = (w_new[sel] - w_ref[sel]).abs()
    changed = delta > threshold
    return {
        "count": n,
        "changed_frac": float(changed.float().mean().item()),
        "mean_abs_delta": float(delta.mean().item()),
        "max_abs_delta": float(delta.max().item()),
    }


def _print_header() -> None:
    print(
        f"{'layer':<60} {'label':>5} {'count':>10} {'|w| mean':>10} {'nonzero%':>9} "
        f"{'Δ mean':>10} {'Δ>thr%':>8}"
    )
    print("-" * 120)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PackNet mask labels vs weight changes at iter_2000 (main CL ckpt)."
    )
    parser.add_argument("--experiment", required=True, help="Inference/training experiment name.")
    parser.add_argument("--ckpt", required=True, help="Main CL checkpoint (e.g. iter_000002000).")
    parser.add_argument(
        "--packnet_path",
        required=True,
        help="Pre-end-prune PackNet masks (cl_stage1_pre_prune/).",
    )
    parser.add_argument(
        "--reference_ckpt",
        default="",
        help="Reference checkpoint for deltas (base iter_7000 or before-main-CL snapshot).",
    )
    parser.add_argument("--config_file", default=_CONFIG_PY)
    parser.add_argument(
        "--delta_threshold",
        type=float,
        default=1e-6,
        help="Treat |Δ| > threshold as 'changed' (default 1e-6).",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=0,
        help="Max layers to print (0 = summary only, -1 = all layers).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"CL checkpoint:     {args.ckpt}")
    print(f"PackNet masks:     {args.packnet_path}")
    if args.reference_ckpt:
        print(f"Reference ckpt:    {args.reference_ckpt}")
    else:
        print("Reference ckpt:    (none — weight stats only, no Δ)")
    print(f"Delta threshold:   {args.delta_threshold}")
    print()

    packnet, masks = _load_masks(args.packnet_path, args.experiment, args.ckpt, args.config_file)
    print(f"PackNet task_id={packnet.task_id}  prune_perc={packnet.prune_perc}")
    print(f"Expected main CL train label: {packnet.task_id + 1} (label 2 for CL stage 1)")
    print()

    weights = _load_prunable_weights(args.experiment, args.ckpt, args.config_file)
    ref_weights: Dict[str, torch.Tensor] = {}
    if args.reference_ckpt:
        ref_weights = _load_prunable_weights(args.experiment, args.reference_ckpt, args.config_file)

    agg: Dict[int, dict] = defaultdict(lambda: {"count": 0, "nonzero": 0, "delta_sum": 0.0, "changed": 0})
    names = sorted(set(masks.keys()) & set(weights.keys()))

    if args.max_layers != 0:
        _print_header()
    show_all = args.max_layers < 0
    limit = None if show_all else (args.max_layers if args.max_layers > 0 else 0)

    for i, name in enumerate(names):
        if limit is not None and i >= limit:
            break
        w = weights[name]
        m = masks[name]
        if w.shape != m.shape:
            print(f"SKIP shape mismatch: {name} w={tuple(w.shape)} m={tuple(m.shape)}")
            continue
        w_ref = ref_weights.get(name) if ref_weights else None

        for label in sorted(_label_counts(m).keys()):
            st = _stats_for_label(w, m, label)
            ds = (
                _delta_stats(w, w_ref, m, label, args.delta_threshold)
                if w_ref is not None and w_ref.shape == w.shape
                else None
            )
            agg[label]["count"] += st["count"]
            agg[label]["nonzero"] += int(st["nonzero_frac"] * st["count"])
            if ds is not None:
                agg[label]["delta_sum"] += ds["mean_abs_delta"] * ds["count"]
                agg[label]["changed"] += int(ds["changed_frac"] * ds["count"])

            if args.max_layers != 0:
                d_mean = f"{ds['mean_abs_delta']:.2e}" if ds else "n/a"
                d_pct = f"{100*ds['changed_frac']:.1f}" if ds else "n/a"
                print(
                    f"{name:<60} {label:>5} {st['count']:>10} {st['mean_abs']:>10.4f} "
                    f"{100*st['nonzero_frac']:>8.1f}% {d_mean:>10} {d_pct:>7}%"
                )

    print()
    print("AGGREGATE (prunable Conv/Linear weights):")
    total = sum(a["count"] for a in agg.values())
    for label in sorted(agg.keys()):
        a = agg[label]
        nz = 100.0 * a["nonzero"] / a["count"] if a["count"] else 0.0
        role = {0: "pruned/free", 1: "base (tasks 0-5)", 2: "CL task 6"}.get(label, f"task label {label}")
        line = f"  label {label} ({role}): {a['count']:,} weights ({100*a['count']/total:.1f}%), nonzero {nz:.1f}%"
        if ref_weights and a["count"]:
            mean_d = a["delta_sum"] / a["count"]
            chg = 100.0 * a["changed"] / a["count"]
            line += f", vs ref: mean|Δ|={mean_d:.2e}, changed {chg:.1f}%"
        print(line)

    print()
    print("INTERPRETATION (CL stage 1, main CL → iter_2000):")
    print("  label 2  — should be TRAINED in main CL (nonzero, large |Δ| vs base at those slots).")
    print("  label 1  — frozen in main CL (if ref = before-main-CL: |Δ| ≈ 0; if ref = base ckpt: |Δ| > 0 from retrain).")
    print("  label 0  — should stay zero.")

    if args.reference_ckpt:
        print()
        print("Non-prunable params (biases + norm — should be frozen by PackNet):")
        non_pr = _load_non_prunable_weights(args.experiment, args.ckpt, args.config_file)
        non_ref = _load_non_prunable_weights(args.experiment, args.reference_ckpt, args.config_file)
        max_delta = 0.0
        worst = ""
        n_changed = 0
        n_total = 0
        for name in sorted(non_pr.keys()):
            if name not in non_ref or non_pr[name].shape != non_ref[name].shape:
                continue
            d = (non_pr[name] - non_ref[name]).abs().max().item()
            n_total += non_pr[name].numel()
            n_changed += int((non_pr[name] - non_ref[name]).abs().gt(args.delta_threshold).sum().item())
            if d > max_delta:
                max_delta = d
                worst = name
        print(f"  {n_changed:,}/{n_total:,} elements changed (|Δ| > {args.delta_threshold})")
        print(f"  max |Δ| = {max_delta:.2e}  ({worst})")
        if max_delta <= args.delta_threshold:
            print("  OK: biases/norms unchanged vs reference (PackNet froze them).")
        else:
            print("  WARNING: some biases/norms changed — unexpected for PackNet training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
