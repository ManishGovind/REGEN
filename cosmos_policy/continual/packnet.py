# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
PackNet for the Cosmos Policy (Mallya & Lazebnik, 2018).

PackNet is a dynamic-architecture continual-learning method: after each task the
network is pruned (lowest-magnitude weights among the current task's allocation
are removed), the surviving weights are fine-tuned, then frozen for future tasks.
New tasks train only on previously pruned (mask == 0) capacity.

This implementation follows the LIBERO lifelong-learning reference:
https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/algos/packnet.py

FSDP / DTensor handling mirrors ``cosmos_policy.continual.ewc``: masks are stored
per-parameter with the same shard layout as model weights; pruning ranks weights
on the **global** (gathered) tensor, then re-shards the updated mask/weights.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from cosmos_policy._src.imaginaire.utils import distributed, log
from cosmos_policy.continual.ewc import (
    TensorDict,
    _align_ewc_state_to_net,
    _empty_shell_like,
    _full_tensor,
    _is_dtensor,
    _is_legacy_pt_path,
    _local_shard_from_full,
    _reshard_full_to_ref,
    _to_local,
)

_MASK_PREFIX = "mask."
_DCP_TENSOR_SUBDIR = "packnet_state"
_DCP_META_FILE = "meta.pt"

NamedParams = Iterable[Tuple[str, nn.Parameter]]


def _is_norm_module(module: nn.Module) -> bool:
    name = module.__class__.__name__
    return "BatchNorm" in name or "LayerNorm" in name or "GroupNorm" in name or "RMSNorm" in name


def iter_prunable_weight_params(net: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Return ``(param_name, weight_param)`` for Conv2d / Linear weights only."""
    out: List[Tuple[str, nn.Parameter]] = []
    for module_name, module in net.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        param_name = f"{module_name}.weight" if module_name else "weight"
        out.append((param_name, module.weight))
    return out


def _resolve_dcp_checkpoint_dir(path: str) -> str:
    path = path.rstrip("/")
    meta_path = os.path.join(path, _DCP_META_FILE)
    tensor_dir = os.path.join(path, _DCP_TENSOR_SUBDIR)
    if os.path.isfile(meta_path) and os.path.isdir(tensor_dir):
        return path
    if os.path.basename(path) == _DCP_TENSOR_SUBDIR:
        parent = os.path.dirname(path)
        parent_meta = os.path.join(parent, _DCP_META_FILE)
        parent_tensors = os.path.join(parent, _DCP_TENSOR_SUBDIR)
        if os.path.isfile(parent_meta) and os.path.isdir(parent_tensors):
            if distributed.is_rank0():
                log.warning(
                    f"[PackNet] packnet_prev_state_path pointed at inner '{_DCP_TENSOR_SUBDIR}/'; "
                    f"using parent {parent}"
                )
            return parent
    return path


def _flatten_masks(masks: TensorDict) -> Dict[str, torch.Tensor]:
    return {f"{_MASK_PREFIX}{name}": tensor for name, tensor in masks.items()}


def _unflatten_masks(state: Dict[str, torch.Tensor]) -> TensorDict:
    prefix_len = len(_MASK_PREFIX)
    masks: TensorDict = {}
    for key, tensor in state.items():
        if key.startswith(_MASK_PREFIX):
            masks[key[prefix_len:]] = tensor
    return masks


def _align_masks_to_net(masks: TensorDict, net: nn.Module) -> TensorDict:
    aligned, _ = _align_ewc_state_to_net(masks, {}, net)
    return aligned


def _copy_full_weight_into_param(param: nn.Parameter, full_weight: torch.Tensor) -> None:
    """Write a global weight tensor into ``param`` (handles DTensor / FSDP shards).

    FSDP-2 weights may be DTensors or Parameters whose ``.data`` is a DTensor.
    In-place ``to_local().copy_(...)`` raises autograd view errors; re-shard via
    ``_reshard_full_to_ref`` and assign / copy through ``.data`` instead.
    """
    with torch.no_grad():
        if isinstance(param, nn.Parameter):
            storage = param.data
        else:
            storage = param

        global_shape = tuple(_full_tensor(storage).shape)
        if full_weight.shape != global_shape:
            raise RuntimeError(
                f"[PackNet] Weight shape mismatch: got {tuple(full_weight.shape)}, "
                f"expected {global_shape}"
            )

        store_dtype = _to_local(storage).dtype
        new_storage = _reshard_full_to_ref(full_weight.detach().cpu(), param, store_dtype)

        if isinstance(param, nn.Parameter):
            param.data = new_storage
        else:
            # ``param`` is a bare DTensor (``module.weight`` under FSDP-2).
            _to_local(param).data.copy_(_to_local(new_storage))


class PackNet:
    """
    PackNet mask manager for continual-learning stages.

    Mask semantics (uint8, per weight element):
        0 — pruned / free capacity for a future task
        k — owned by task ``k - 1`` (task 0 uses value 1, task 1 uses 2, ...)
    """

    def __init__(self, *, prune_perc: float = 0.25, task_id: int = 0) -> None:
        if not 0.0 < prune_perc < 1.0:
            raise ValueError(f"packnet prune_perc must be in (0, 1), got {prune_perc}")
        if task_id < 0:
            raise ValueError(f"packnet task_id must be >= 0, got {task_id}")

        self.prune_perc = float(prune_perc)
        self.task_id = int(task_id)
        self.masks: TensorDict = {}
        self.is_initialized: bool = False

    @property
    def current_task_label(self) -> int:
        """Mask value for the active task (task_id + 1)."""
        return self.task_id + 1

    def has_state(self) -> bool:
        return bool(self.masks)

    def init_masks(self, net: nn.Module) -> None:
        """Create zero masks for every prunable weight (first task / fresh run)."""
        self.masks = {}
        for name, param in iter_prunable_weight_params(net):
            full = torch.zeros(tuple(_full_tensor(param).shape), dtype=torch.uint8)
            self.masks[name] = _reshard_full_to_ref(full, param, torch.uint8)
        self.is_initialized = True

    def mark_all_weights_as_task(self, net: nn.Module, task_id: int) -> None:
        """Assign every prunable weight to ``task_id`` (already-trained / frozen task)."""
        if task_id < 0:
            raise ValueError(f"task_id must be >= 0, got {task_id}")
        label = task_id + 1
        if not self.masks:
            self.init_masks(net)
        name_to_param = dict(net.named_parameters())
        for name in list(self.masks.keys()):
            param = name_to_param[name]
            full = torch.full(tuple(_full_tensor(param).shape), label, dtype=torch.uint8)
            self.masks[name] = _reshard_full_to_ref(full.cpu(), param, torch.uint8)
        self.is_initialized = True
        if distributed.is_rank0():
            log.info(f"[PackNet] mark_all_weights_as_task: assigned all weights to task_id={task_id} (label={label})")

    def init_from_pretrained_base(self, net: nn.Module, *, base_task_id: int = 0) -> None:
        """Treat a loaded checkpoint as an already-finished PackNet base task, then open pruned slots.

        Canonical flow when continual learning starts from a multi-task *base* checkpoint:

        1. Mark every weight as owned by ``base_task_id`` (default 0 → mask label 1).
        2. Prune the lowest-magnitude ``prune_perc`` fraction of those base weights.
        3. Assign the freed (mask==0) slots to the current ``self.task_id`` via ``start_task``.

        Requires ``self.task_id >= 1`` so the new CL task does not overwrite the base label.
        """
        if self.task_id < 1:
            raise ValueError(
                f"init_from_pretrained_base requires packnet_task_id >= 1 (got {self.task_id}). "
                "The base checkpoint occupies task_id=0; the first CL stage should use task_id=1."
            )

        if distributed.is_rank0():
            log.critical(
                f"[PackNet] init_from_pretrained_base: treating loaded weights as base task "
                f"{base_task_id}, pruning ~{100 * self.prune_perc:.1f}%, then opening slots for "
                f"CL task_id={self.task_id}."
            )

        self.init_masks(net)
        self.mark_all_weights_as_task(net, base_task_id)

        cl_task_id = self.task_id
        self.task_id = base_task_id
        self.prune(net)
        self.task_id = cl_task_id

        self.start_task(net)

    def start_task(self, net: nn.Module) -> None:
        """Assign all pruned (0) slots to the current task; keep prior tasks frozen."""
        if not self.masks:
            self.init_masks(net)
        label = self.current_task_label
        for name in list(self.masks.keys()):
            full_mask = _full_tensor(self.masks[name]).to(torch.uint8)
            full_mask[full_mask == 0] = label
            ref = dict(net.named_parameters())[name]
            self.masks[name] = _reshard_full_to_ref(full_mask.cpu(), ref, torch.uint8)
        self.apply_weight_mask(net)
        if distributed.is_rank0():
            log.info(f"[PackNet] start_task: task_id={self.task_id} (mask label={label})")

    def set_norm_layers_eval(self, net: nn.Module) -> None:
        """PackNet does not train normalization layers (LIBERO convention)."""
        for module in net.modules():
            if _is_norm_module(module):
                module.eval()

    def apply_weight_mask(self, net: nn.Module) -> None:
        """Zero weights whose mask is 0."""
        name_to_param = dict(net.named_parameters())
        for name, mask_t in self.masks.items():
            if name not in name_to_param:
                continue
            param = name_to_param[name]
            full_w = _full_tensor(param.data).clone()
            full_m = _full_tensor(mask_t).to(torch.uint8)
            full_w[full_m == 0] = 0
            _copy_full_weight_into_param(param, full_w)

    def zero_grads(self, net: nn.Module) -> None:
        """Mask gradients: only current-task weights update; biases and norm layers frozen."""
        if not self.masks:
            return
        label = self.current_task_label
        name_to_param = dict(net.named_parameters())

        for name, mask_t in self.masks.items():
            param = name_to_param.get(name)
            if param is None or param.grad is None:
                continue
            full_mask = _full_tensor(mask_t).to(torch.uint8)
            grad_local = _to_local(param.grad)
            mask_local = _local_shard_from_full(full_mask.cpu(), param).to(
                device=grad_local.device, dtype=torch.uint8
            )
            grad_local[mask_local != label] = 0

        for module_name, module in net.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
                if module.bias.grad is not None:
                    module.bias.grad.zero_()
            elif _is_norm_module(module):
                if getattr(module, "weight", None) is not None and module.weight.grad is not None:
                    module.weight.grad.zero_()
                if getattr(module, "bias", None) is not None and module.bias is not None and module.bias.grad is not None:
                    module.bias.grad.zero_()

    def _pruning_mask_for_layer(
        self,
        weights: torch.Tensor,
        previous_mask: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Rank by magnitude; prune the lowest ``prune_perc`` fraction of current-task weights."""
        label = self.current_task_label
        previous_mask = previous_mask.to(torch.uint8)
        current = weights[previous_mask == label]
        if current.numel() == 0:
            return previous_mask

        abs_tensor = current.abs()
        cutoff_rank = max(1, round(self.prune_perc * current.numel()))
        cutoff_rank = min(cutoff_rank, current.numel())
        cutoff_value = abs_tensor.reshape(-1).cpu().kthvalue(cutoff_rank).values.item()

        remove_mask = weights.abs().le(cutoff_value) & (previous_mask == label)
        new_mask = previous_mask.clone()
        new_mask[remove_mask] = 0

        if distributed.is_rank0():
            pruned = int(remove_mask.sum().item())
            log.info(
                f"[PackNet] {layer_name}: pruned {pruned}/{current.numel()} "
                f"({100.0 * pruned / current.numel():.2f}%) of current-task weights"
            )
        return new_mask

    def prune(self, net: nn.Module) -> None:
        """Prune lowest-magnitude current-task weights and zero them in ``net``."""
        if not self.masks:
            raise RuntimeError("[PackNet] prune() called before masks were initialized.")

        if distributed.is_rank0():
            log.critical(
                f"[PackNet] Pruning each layer: removing ~{100 * self.prune_perc:.1f}% "
                f"of current-task weights (task_id={self.task_id})"
            )

        name_to_param = dict(net.named_parameters())
        new_masks: TensorDict = {}
        for name, mask_t in self.masks.items():
            param = name_to_param[name]
            full_w = _full_tensor(param.data)
            full_prev = _full_tensor(mask_t).to(torch.uint8)
            full_new = self._pruning_mask_for_layer(full_w, full_prev, name)
            full_w[full_new == 0] = 0
            _copy_full_weight_into_param(param, full_w)
            new_masks[name] = _reshard_full_to_ref(full_new.cpu(), param, torch.uint8)

        self.masks = new_masks
        self.is_initialized = True

    def save(self, path: str) -> None:
        if _is_legacy_pt_path(path):
            self._save_pt(path)
        else:
            self._save_dcp(path)

    def _save_pt(self, path: str) -> None:
        rank = distributed.get_rank() if dist.is_initialized() else 0
        gathered: TensorDict = {}
        for name, t in self.masks.items():
            gathered[name] = _full_tensor(t).detach().cpu().to(torch.uint8)

        if rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            payload = {
                "version": 1,
                "format": "pt",
                "prune_perc": self.prune_perc,
                "task_id": self.task_id,
                "masks": gathered,
            }
            torch.save(payload, path)
            log.success(f"[PackNet] Saved masks (legacy .pt) to {path} ({len(gathered)} tensors)")

        if dist.is_initialized():
            dist.barrier()

    def _save_dcp(self, dir_path: str) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemWriter

        rank = distributed.get_rank() if dist.is_initialized() else 0
        dir_path = dir_path.rstrip("/")
        tensor_dir = os.path.join(dir_path, _DCP_TENSOR_SUBDIR)

        if rank == 0:
            os.makedirs(dir_path, exist_ok=True)
            torch.save(
                {
                    "version": 1,
                    "format": "dcp",
                    "prune_perc": self.prune_perc,
                    "task_id": self.task_id,
                    "param_names": list(self.masks.keys()),
                },
                os.path.join(dir_path, _DCP_META_FILE),
            )

        if dist.is_initialized():
            dist.barrier()

        dcp.save(_flatten_masks(self.masks), storage_writer=FileSystemWriter(tensor_dir))

        if rank == 0:
            log.success(f"[PackNet] Saved masks (DCP) to {dir_path}/ ({len(self.masks)} tensors)")

        if dist.is_initialized():
            dist.barrier()

    def load(self, path: str, net: nn.Module, *, strict: bool = False) -> None:
        path = path.rstrip("/")
        if os.path.isfile(path) and _is_legacy_pt_path(path):
            self._load_pt(path, net, strict=strict)
        elif os.path.isdir(path):
            self._load_dcp(_resolve_dcp_checkpoint_dir(path), net, strict=strict)
        else:
            raise FileNotFoundError(
                f"[PackNet] state not found at {path} (expected DCP directory or .pt file)"
            )

    def _load_pt(self, path: str, net: nn.Module, *, strict: bool) -> None:
        rank = distributed.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            log.info(f"[PackNet] Loading masks (legacy .pt) from {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.prune_perc = float(payload.get("prune_perc", self.prune_perc))

        name_to_param = dict(net.named_parameters())
        loaded: TensorDict = {}
        missing: list = []
        for name, full_m in payload["masks"].items():
            if name not in name_to_param:
                missing.append(name)
                continue
            ref = name_to_param[name]
            loaded[name] = _reshard_full_to_ref(full_m.to(torch.uint8), ref, torch.uint8)

        if strict and missing:
            raise RuntimeError(f"[PackNet] {len(missing)} mask tensors missing from model: {missing[:8]}...")
        if missing and rank == 0:
            log.warning(f"[PackNet] {len(missing)} mask tensors missing from model (skipped).")

        self.masks = _align_masks_to_net(loaded, net)
        self.is_initialized = True
        if rank == 0:
            log.success(f"[PackNet] Loaded {len(self.masks)} mask tensors from {path}")

        if dist.is_initialized():
            dist.barrier()

    def _load_dcp(self, dir_path: str, net: nn.Module, *, strict: bool) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader

        rank = distributed.get_rank() if dist.is_initialized() else 0
        meta_path = os.path.join(dir_path, _DCP_META_FILE)
        tensor_dir = os.path.join(dir_path, _DCP_TENSOR_SUBDIR)
        if not os.path.isfile(meta_path) or not os.path.isdir(tensor_dir):
            raise FileNotFoundError(
                f"[PackNet] DCP state incomplete at {dir_path} "
                f"(need {_DCP_META_FILE} and {_DCP_TENSOR_SUBDIR}/)"
            )

        if rank == 0:
            log.info(f"[PackNet] Loading masks (DCP) from {dir_path}")

        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        self.prune_perc = float(meta.get("prune_perc", self.prune_perc))

        name_to_param = dict(net.named_parameters())
        param_names = meta.get("param_names", list(name_to_param.keys()))
        load_sd: Dict[str, torch.Tensor] = {}
        missing: list = []
        for name in param_names:
            if name not in name_to_param:
                missing.append(name)
                continue
            ref = name_to_param[name]
            load_sd[f"{_MASK_PREFIX}{name}"] = _empty_shell_like(ref)

        dcp.load(load_sd, storage_reader=FileSystemReader(tensor_dir))
        loaded = _unflatten_masks(load_sd)
        for name, t in list(loaded.items()):
            loaded[name] = t.to(torch.uint8)

        if strict and missing:
            raise RuntimeError(f"[PackNet] {len(missing)} mask tensors missing from model: {missing[:8]}...")
        if missing and rank == 0:
            log.warning(f"[PackNet] {len(missing)} mask tensors missing from model (skipped).")

        self.masks = _align_masks_to_net(loaded, net)
        self.is_initialized = True
        if rank == 0:
            log.success(f"[PackNet] Loaded {len(self.masks)} mask tensors (DCP) from {dir_path}")

        if dist.is_initialized():
            dist.barrier()

    def to(self, device: torch.device) -> "PackNet":
        """Move mask tensors to ``device`` without changing FSDP sharding."""
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:  # pragma: no cover
            from torch.distributed._tensor import DTensor  # type: ignore

        for name in list(self.masks.keys()):
            t = self.masks[name]
            if isinstance(t, DTensor):
                local = t.to_local().to(device)
                self.masks[name] = DTensor.from_local(
                    local, device_mesh=t.device_mesh, placements=t.placements
                )
            else:
                self.masks[name] = t.to(device)
        return self

    def apply_eval_mask(self, net: nn.Module, eval_task_id: int) -> None:
        """Zero weights reserved for tasks after ``eval_task_id`` (inference / eval)."""
        max_label = eval_task_id + 1
        name_to_param = dict(net.named_parameters())
        for name, mask_t in self.masks.items():
            if name not in name_to_param:
                continue
            param = name_to_param[name]
            full_w = _full_tensor(param.data).clone()
            full_m = _full_tensor(mask_t).to(torch.uint8)
            full_w[full_m == 0] = 0
            full_w[full_m > max_label] = 0
            _copy_full_weight_into_param(param, full_w)
