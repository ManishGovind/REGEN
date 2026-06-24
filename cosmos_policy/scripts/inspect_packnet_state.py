# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Inspect saved PackNet mask states (DCP directory or legacy ``.pt``).

Mask labels (uint8, per weight element):
    0 — pruned / free capacity
    k — owned by PackNet task ``k - 1`` (base task 0 → label 1, CL stage 1 → label 2, …)

Examples::

    # Metadata only (no GPU / model):
    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.inspect_packnet_state \\
        --packnet_path /workspace/packnet_states/libero_goal/cl_stage1 \\
        --meta_only

    # Legacy .pt masks (no model):
    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.inspect_packnet_state \\
        --packnet_path /path/to/packnet_masks.pt

    # DCP masks (needs model for shard layout; same ckpt architecture as training):
    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.inspect_packnet_state \\
        --packnet_path /workspace/packnet_states/libero_goal/cl_stage1 \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \\
        --ckpt /path/to/iter_000002000/model.pt

    # Export DCP → portable legacy .pt for offline inspection:
    uv run --extra cu128 --group libero --python 3.10 \\
        python -m cosmos_policy.scripts.inspect_packnet_state \\
        --packnet_path /workspace/packnet_states/libero_goal/cl_stage1 \\
        --experiment cosmos_predict2_2b_480p_libero_goal_base_stage_inference_only \\
        --ckpt /path/to/iter_000002000/model.pt \\
        --export_legacy_pt /tmp/cl_stage1_masks.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch

from cosmos_policy.continual.ewc import _full_tensor
from cosmos_policy.continual.packnet import (
    PackNet,
    _DCP_META_FILE,
    _is_legacy_pt_path,
    _resolve_dcp_checkpoint_dir,
)

_CONFIG_PY = "cosmos_policy/config/config.py"


def _load_meta(packnet_path: str) -> dict:
    path = packnet_path.rstrip("/")
    if os.path.isfile(path) and _is_legacy_pt_path(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "format": payload.get("format", "pt"),
            "prune_perc": float(payload.get("prune_perc", 0.0)),
            "task_id": int(payload.get("task_id", 0)),
            "param_names": sorted(payload.get("masks", {}).keys()),
        }
    dcp_dir = _resolve_dcp_checkpoint_dir(path)
    meta_path = os.path.join(dcp_dir, _DCP_META_FILE)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No PackNet meta at {meta_path}")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    meta.setdefault("format", "dcp")
    meta["param_names"] = list(meta.get("param_names", []))
    return meta


def _print_meta(meta: dict, packnet_path: str) -> None:
    print(f"PackNet path: {packnet_path}")
    print(f"  format:      {meta.get('format')}")
    print(f"  task_id:     {meta.get('task_id')}  (saved after this CL stage)")
    print(f"  prune_perc:  {meta.get('prune_perc')}  (fraction pruned per layer)")
    print(f"  num layers:  {len(meta.get('param_names', []))}")
    names = meta.get("param_names", [])
    if names:
        print(f"  first layer: {names[0]}")
        print(f"  last layer:  {names[-1]}")


def _label_counts(mask: torch.Tensor) -> Dict[int, int]:
    m = mask.to(torch.uint8)
    return {int(v): int((m == v).sum().item()) for v in m.unique(sorted=True)}


def _format_pct(counts: Dict[int, int], total: int) -> str:
    parts = []
    for label in sorted(counts):
        pct = 100.0 * counts[label] / total if total else 0.0
        parts.append(f"{label}:{counts[label]} ({pct:.1f}%)")
    return ", ".join(parts)


def _summarize_masks(
    masks: Dict[str, torch.Tensor],
    *,
    max_layers: Optional[int] = None,
    layer_filter: Optional[str] = None,
) -> None:
    names = sorted(masks.keys())
    if layer_filter:
        names = [n for n in names if layer_filter in n]
    if max_layers is not None:
        names = names[:max_layers]

    agg: Dict[int, int] = defaultdict(int)
    total_weights = 0

    print()
    print(f"{'layer':<72} {'shape':<22} label counts")
    print("-" * 120)
    for name in names:
        full_m = _full_tensor(masks[name]).to(torch.uint8)
        total = full_m.numel()
        counts = _label_counts(full_m)
        for label, c in counts.items():
            agg[label] += c
        total_weights += total
        shape_str = str(tuple(full_m.shape))
        print(f"{name:<72} {shape_str:<22} {_format_pct(counts, total)}")

    if max_layers is not None and len(sorted(masks.keys())) > len(names):
        print(f"... ({len(masks) - len(names)} layers omitted; use --max_layers 0 to show all)")

    print("-" * 120)
    print(f"AGGREGATE over {len(masks)} layers, {total_weights:,} weights:")
    print(f"  {_format_pct(dict(agg), total_weights)}")
    print()
    print("Label key: 0=pruned/free, 1=base (task 0), 2=CL stage 1 (task 6), 3=CL stage 2, ...")


def _load_legacy_pt_masks(path: str) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "format": payload.get("format", "pt"),
        "prune_perc": float(payload.get("prune_perc", 0.0)),
        "task_id": int(payload.get("task_id", 0)),
        "param_names": sorted(payload.get("masks", {}).keys()),
    }
    masks = {k: v.to(torch.uint8) for k, v in payload["masks"].items()}
    return meta, masks


def _load_masks_with_model(
    packnet_path: str,
    experiment: str,
    ckpt: str,
    config_file: str,
) -> tuple[dict, Dict[str, torch.Tensor], PackNet]:
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(
        experiment_name=experiment,
        s3_checkpoint_dir=ckpt,
        config_file=config_file,
        load_ema_to_reg=False,
    )
    model.eval()
    model = model.to("cuda")

    meta_pre = _load_meta(packnet_path)
    packnet = PackNet(
        prune_perc=float(meta_pre.get("prune_perc", 0.25)),
        task_id=int(meta_pre.get("task_id", 0)),
    )
    packnet.load(packnet_path, net=model.net)

    masks = {name: _full_tensor(t).detach().cpu().to(torch.uint8) for name, t in packnet.masks.items()}
    meta = {
        "format": meta_pre.get("format"),
        "prune_perc": packnet.prune_perc,
        "task_id": packnet.task_id,
        "param_names": sorted(masks.keys()),
    }
    return meta, masks, packnet


def _export_legacy_pt(packnet: PackNet, output_path: str) -> None:
    rank = 0
    gathered = {name: _full_tensor(t).detach().cpu().to(torch.uint8) for name, t in packnet.masks.items()}
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "format": "pt",
            "prune_perc": packnet.prune_perc,
            "task_id": packnet.task_id,
            "masks": gathered,
        }
        torch.save(payload, output_path)
        print(f"Exported legacy .pt to {output_path} ({len(gathered)} tensors)")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect PackNet saved mask states.")
    parser.add_argument(
        "--packnet_path",
        required=True,
        help="DCP directory (meta.pt + packnet_state/) or legacy .pt file.",
    )
    parser.add_argument(
        "--meta_only",
        action="store_true",
        help="Print meta.pt / header fields only (no model, no GPU).",
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="Inference experiment name (required for DCP mask load without legacy .pt).",
    )
    parser.add_argument(
        "--ckpt",
        default="",
        help="Model checkpoint path (same architecture as when masks were saved).",
    )
    parser.add_argument(
        "--config_file",
        default=_CONFIG_PY,
        help="Config module path.",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=0,
        help="Max layers to print (0 = all).",
    )
    parser.add_argument(
        "--layer_filter",
        default="",
        help="Only print layers whose name contains this substring.",
    )
    parser.add_argument(
        "--export_legacy_pt",
        default="",
        help="If set with --experiment and --ckpt, export gathered masks to this .pt path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    packnet_path = args.packnet_path.rstrip("/")
    max_layers = None if args.max_layers == 0 else args.max_layers
    layer_filter = args.layer_filter or None

    try:
        if args.meta_only:
            meta = _load_meta(packnet_path)
            _print_meta(meta, packnet_path)
            return

        if os.path.isfile(packnet_path) and _is_legacy_pt_path(packnet_path):
            meta, masks = _load_legacy_pt_masks(packnet_path)
            _print_meta(meta, packnet_path)
            _summarize_masks(masks, max_layers=max_layers, layer_filter=layer_filter)
            return

        if not args.experiment or not args.ckpt:
            meta = _load_meta(packnet_path)
            _print_meta(meta, packnet_path)
            print()
            print(
                "DCP mask tensors require --experiment and --ckpt to restore shard layout.\n"
                "Re-run with those flags, or use --meta_only for a quick check."
            )
            sys.exit(0)

        meta, masks, packnet = _load_masks_with_model(
            packnet_path,
            experiment=args.experiment,
            ckpt=args.ckpt,
            config_file=args.config_file,
        )
        _print_meta(meta, packnet_path)
        _summarize_masks(masks, max_layers=max_layers, layer_filter=layer_filter)

        if args.export_legacy_pt:
            _export_legacy_pt(packnet, args.export_legacy_pt)

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
