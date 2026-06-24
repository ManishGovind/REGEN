# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Online Elastic Weight Consolidation (EWC) for the Cosmos Policy.

This module is heavily inspired by:
- continual-openpi (online EWC, JAX/Flax):
  https://github.com/Continual-VLAs/continual-openpi/blob/main/src/openpi/continual/ewc.py
- Lotus (canonical PyTorch EWC):
  https://github.com/UT-Austin-RPL/Lotus/blob/master/lotus/lifelong/algos/ewc.py

It maintains a single Fisher diagonal `F` and a single parameter snapshot `theta*`
across continual-learning (CL) stages:

    F      <-  gamma * F + F_new
    theta* <-  theta_new

During training on a new task, the EWC penalty

    L_ewc(theta) = (lambda / 2) * sum_i F_i * (theta_i - theta*_i)**2

is added to the regular task loss to prevent forgetting of previous tasks.

FSDP / DTensor handling
-----------------------
Cosmos Policy uses PyTorch FSDP-2 (`fully_shard`), so model parameters are stored
as `DTensor`s sharded across the data-parallel mesh. For correct per-shard
gradients, the penalty is computed locally on each rank using the **local shard**
of every parameter / Fisher / theta* tensor. Each rank therefore adds its own
portion of the EWC penalty to the loss; FSDP backward then routes each shard's
gradient to the rank that owns it -- which is exactly what we want.

Save / load (default: DCP)
--------------------------
By default Fisher / theta* are saved in **PyTorch Distributed Checkpoint (DCP)**
format under a directory (same idea as model checkpoints)::

    ewc_states/libero_goal/base_stage_iter7000/
        meta.pt          # lambda, gamma, param names (rank 0)
        ewc_state/       # per-rank sharded DTensors (Fisher + theta*)

Each rank writes only its local FSDP shards -- no full-tensor gather. Pass a
path **without** a ``.pt`` suffix for DCP. Legacy single-file ``.pt`` is still
supported for loading (and optional export via ``export_ewc_state_to_pt.py``).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
    from torch.distributed.tensor.placement_types import Replicate, Shard
except ImportError:  # pragma: no cover -- older PyTorch fallback
    from torch.distributed._tensor import DTensor, distribute_tensor  # type: ignore
    from torch.distributed._tensor.placement_types import Replicate, Shard  # type: ignore

from cosmos_policy._src.imaginaire.utils import distributed, log, misc

NamedParams = Iterable[Tuple[str, nn.Parameter]]
TensorDict = Dict[str, torch.Tensor]

_FISHER_PREFIX = "fisher."
_PREV_PREFIX = "prev_params."
_DCP_TENSOR_SUBDIR = "ewc_state"
_DCP_META_FILE = "meta.pt"


def _is_dtensor(t: torch.Tensor) -> bool:
    return isinstance(t, DTensor)


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Return the local shard if `t` is a DTensor, otherwise `t` itself."""
    return t.to_local() if _is_dtensor(t) else t


def _from_local_like(local: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Build a DTensor from `local` matching the placement/mesh of `ref`."""
    if _is_dtensor(ref):
        return DTensor.from_local(
            local,
            device_mesh=ref.device_mesh,
            placements=ref.placements,
        )
    return local


def _full_tensor(t: torch.Tensor) -> torch.Tensor:
    """Materialise the full (un-sharded) tensor on every rank."""
    return t.full_tensor() if _is_dtensor(t) else t


def _local_shard_from_full(full_t: torch.Tensor, ref: nn.Parameter) -> torch.Tensor:
    """Slice a global tensor into this rank's local shard matching ``ref``.

    ``distribute_tensor`` can produce the wrong local size when the GPU count
  differs from the run that saved the EWC file (e.g. Fisher on 8 GPUs, load on
    4). Slicing with ``ref``'s mesh coordinate is reliable across GPU counts.
    """
    ref_local = _to_local(ref)
    if full_t.shape == ref_local.shape:
        return full_t

    if not _is_dtensor(ref):
        return full_t

    global_shape = tuple(ref.shape)
    if full_t.shape != global_shape:
        raise RuntimeError(
            f"[EWC] Saved tensor shape {tuple(full_t.shape)} does not match the "
            f"model's global shape {global_shape}."
        )

    mesh = ref.device_mesh
    placements = ref.placements
    coord = mesh.get_coordinate()
    if coord is None:
        raise RuntimeError("[EWC] Current rank is not part of the parameter device mesh.")

    local = full_t
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Replicate):
            continue
        if not isinstance(placement, Shard):
            continue
        shard_dim = placement.dim
        mesh_size = mesh.size(mesh_dim)
        if mesh_size <= 1:
            continue
        rank_idx = coord[mesh_dim]
        dim_size = local.size(shard_dim)
        if dim_size % mesh_size != 0:
            raise RuntimeError(
                f"[EWC] Cannot shard dim {shard_dim} of size {dim_size} across "
                f"{mesh_size} ranks (not evenly divisible)."
            )
        chunk_size = dim_size // mesh_size
        start = rank_idx * chunk_size
        local = local.narrow(shard_dim, start, chunk_size).contiguous()

    return local


def _is_legacy_pt_path(path: str) -> bool:
    return path.endswith(".pt")


def _resolve_dcp_checkpoint_dir(path: str) -> str:
    """Resolve the DCP *root* directory (contains ``meta.pt`` + ``ewc_state/``).

    Accepts either the root or the inner ``ewc_state/`` shard directory (common mistake).
    """
    path = path.rstrip("/")
    meta_path = os.path.join(path, _DCP_META_FILE)
    tensor_dir = os.path.join(path, _DCP_TENSOR_SUBDIR)
    if os.path.isfile(meta_path) and os.path.isdir(tensor_dir):
        return path

    # User passed .../ewc_state (inner DCP shard dir) instead of the parent root.
    if os.path.basename(path) == _DCP_TENSOR_SUBDIR:
        parent = os.path.dirname(path)
        parent_meta = os.path.join(parent, _DCP_META_FILE)
        parent_tensors = os.path.join(parent, _DCP_TENSOR_SUBDIR)
        if os.path.isfile(parent_meta) and os.path.isdir(parent_tensors):
            if distributed.is_rank0():
                log.warning(
                    f"[EWC] ewc_prev_state_path pointed at the inner '{_DCP_TENSOR_SUBDIR}/' "
                    f"folder; using parent directory {parent}"
                )
            return parent

    return path


def _flatten_ewc_tensors(fisher: TensorDict, prev_params: TensorDict) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in fisher.items():
        state[f"{_FISHER_PREFIX}{name}"] = tensor
    for name, tensor in prev_params.items():
        state[f"{_PREV_PREFIX}{name}"] = tensor
    return state


def _unflatten_ewc_tensors(state: Dict[str, torch.Tensor]) -> Tuple[TensorDict, TensorDict]:
    fisher: TensorDict = {}
    prev_params: TensorDict = {}
    fisher_prefix_len = len(_FISHER_PREFIX)
    prev_prefix_len = len(_PREV_PREFIX)
    for key, tensor in state.items():
        if key.startswith(_FISHER_PREFIX):
            fisher[key[fisher_prefix_len:]] = tensor
        elif key.startswith(_PREV_PREFIX):
            prev_params[key[prev_prefix_len:]] = tensor
    return fisher, prev_params


def _empty_shell_like(ref: nn.Parameter) -> torch.Tensor:
    """Allocate a DTensor shell with the same layout as ``ref`` (for ``dcp.load``)."""
    local = torch.zeros_like(_to_local(ref))
    return _from_local_like(local, ref)


def _fisher_dtype_label(fisher_dtype: Optional[torch.dtype]) -> str:
    if fisher_dtype is None:
        return "auto"
    return str(fisher_dtype).replace("torch.", "")


def _ewc_local_aligned(stored: torch.Tensor, param: nn.Parameter) -> torch.Tensor:
    """Return the EWC tensor's local shard aligned with ``param``'s local shard."""
    p_local = _to_local(param)
    s_local = _to_local(stored)
    if s_local.shape == p_local.shape:
        return s_local

    # Stored value is likely a gathered global tensor (legacy .pt) or a shard from
    # a different GPU count / layout. Re-slice to this rank's local shard.
    full_t = _full_tensor(stored).detach()
    if full_t.device != p_local.device:
        full_t = full_t.to(p_local.device)
    return _local_shard_from_full(full_t, param)


def _reshard_full_to_ref(full_t: torch.Tensor, ref: nn.Parameter, store_dtype: torch.dtype) -> torch.Tensor:
    """Build a Fisher / theta* tensor on ``ref``'s FSDP layout from a global CPU tensor."""
    full_t = full_t.to(store_dtype)
    if not _is_dtensor(ref):
        return full_t.to(ref.device)

    ref_local = _to_local(ref)
    if full_t.shape == ref_local.shape:
        return DTensor.from_local(
            full_t.to(ref.device).contiguous(),
            device_mesh=ref.device_mesh,
            placements=ref.placements,
            run_check=False,
        )

    local = _local_shard_from_full(full_t, ref)
    ref_local = ref.to_local()
    if local.shape != ref_local.shape:
        # Last-resort fallback for unusual placement layouts.
        dt = distribute_tensor(
            full_t.to(ref.device),
            device_mesh=ref.device_mesh,
            placements=ref.placements,
        )
        local = dt.to_local()
    if local.shape != ref_local.shape:
        raise RuntimeError(
            f"[EWC] Failed to align EWC shard with parameter: saved global "
            f"{tuple(full_t.shape)}, model local {tuple(ref_local.shape)}, "
            f"sliced local {tuple(local.shape)}."
        )
    return DTensor.from_local(
        local.to(ref.device).contiguous(),
        device_mesh=ref.device_mesh,
        placements=ref.placements,
        run_check=False,
    )


def _align_ewc_state_to_net(
    fisher: TensorDict,
    prev_params: TensorDict,
    net: nn.Module,
) -> Tuple[TensorDict, TensorDict]:
    """Re-align Fisher / theta* with ``net``'s current FSDP shard layout (in-place safe)."""
    name_to_param = dict(net.named_parameters())
    aligned_fisher: TensorDict = {}
    aligned_prev: TensorDict = {}
    n_fixed = 0

    for name in fisher:
        if name not in name_to_param:
            aligned_fisher[name] = fisher[name]
            continue
        param = name_to_param[name]
        f_local = _to_local(fisher[name])
        p_local = _to_local(param)
        if f_local.shape != p_local.shape:
            store_dtype = fisher[name].dtype
            aligned_fisher[name] = _reshard_full_to_ref(
                _full_tensor(fisher[name]).detach().cpu(), param, store_dtype
            )
            n_fixed += 1
        else:
            aligned_fisher[name] = fisher[name]

    for name in prev_params:
        if name not in name_to_param:
            aligned_prev[name] = prev_params[name]
            continue
        param = name_to_param[name]
        t_local = _to_local(prev_params[name])
        p_local = _to_local(param)
        if t_local.shape != p_local.shape:
            store_dtype = prev_params[name].dtype
            aligned_prev[name] = _reshard_full_to_ref(
                _full_tensor(prev_params[name]).detach().cpu(), param, store_dtype
            )
            n_fixed += 1
        else:
            aligned_prev[name] = prev_params[name]

    if n_fixed and distributed.is_rank0():
        log.warning(
            f"[EWC] Re-aligned {n_fixed} Fisher/theta* tensors to the current FSDP shard layout "
            f"(common when loading legacy .pt across different GPU counts)."
        )
    return aligned_fisher, aligned_prev


class OnlineEWC:
    """
    Memory-efficient *online* Elastic Weight Consolidation helper.

    Parameters
    ----------
    lambda_:
        Regularisation strength :math:`\\lambda`.
    gamma:
        Decay applied to the running Fisher diagonal after each task.
        :math:`\\gamma=1` is canonical EWC with O(1) memory; :math:`\\gamma<1`
        gradually forgets very old tasks.
    fisher_dtype:
        Storage dtype for Fisher / theta* tensors. ``None`` (default) means
        "match the dtype of the corresponding model parameter" -- so the EWC
        state inherits the base-stage checkpoint's precision exactly.
        Pass an explicit ``torch.dtype`` only if you want a different
        storage precision (e.g. ``torch.bfloat16`` to shrink the file).
    """

    def __init__(
        self,
        *,
        lambda_: float,
        gamma: float = 1.0,
        fisher_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"EWC gamma must be in [0, 1], got {gamma}")
        if lambda_ < 0:
            raise ValueError(f"EWC lambda must be >= 0, got {lambda_}")

        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        # `None` means "auto: match each parameter's dtype on the fly".
        self.fisher_dtype: Optional[torch.dtype] = fisher_dtype

        self.fisher: TensorDict = {}
        self.prev_params: TensorDict = {}
        self.is_initialized: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_dtype(self, ref: torch.Tensor) -> torch.dtype:
        """Dtype to store Fisher/theta* in: explicit override or ``ref.dtype``."""
        return self.fisher_dtype if self.fisher_dtype is not None else ref.dtype

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def has_state(self) -> bool:
        """True iff Fisher / theta* are populated and the penalty is non-trivial."""
        return self.is_initialized and len(self.fisher) > 0 and self.lambda_ > 0.0

    def __repr__(self) -> str:  # pragma: no cover -- cosmetic
        return (
            f"OnlineEWC(lambda_={self.lambda_}, gamma={self.gamma}, "
            f"initialised={'yes' if self.is_initialized else 'no'}, "
            f"num_params={len(self.fisher)})"
        )

    # ------------------------------------------------------------------
    # Penalty (added to the task loss every training step)
    # ------------------------------------------------------------------

    def penalty(self, named_params: NamedParams) -> torch.Tensor:
        """
        Compute :math:`(\\lambda/2) \\sum_i F_i (\\theta_i - \\theta^*_i)^2`
        over the **local shard** owned by this rank.

        Notes
        -----
        With FSDP-2 each rank only owns a slice of every parameter. Adding the
        local penalty contribution to the task loss is sufficient: each rank's
        backward pass populates ``param.grad`` only for its local shard, which
        is precisely the slice the penalty's gradient targets.
        """
        if not self.has_state():
            # Return a zero scalar that is connected to nothing -- safe no-op.
            return torch.zeros((), device="cuda")

        total = torch.zeros((), device="cuda", dtype=torch.float32)
        for name, param in named_params:
            if name not in self.fisher or name not in self.prev_params:
                continue

            # Local shards (autograd-aware for `param`).
            p_local = _to_local(param)
            f_local = _ewc_local_aligned(self.fisher[name], param).detach()
            t_local = _ewc_local_aligned(self.prev_params[name], param).detach()

            # Skip empty local shards (can happen with uneven sharding).
            if p_local.numel() == 0:
                continue

            if f_local.shape != p_local.shape or t_local.shape != p_local.shape:
                raise RuntimeError(
                    f"[EWC] Shard shape mismatch for {name}: param local {tuple(p_local.shape)}, "
                    f"fisher {tuple(f_local.shape)}, theta* {tuple(t_local.shape)}. "
                    "Re-run compute_ewc_fisher with the same fsdp_shard_size and GPU count, "
                    "or use a DCP directory (no .pt) saved from this training layout."
                )

            # Compute in fp32 for numerical stability.
            diff = p_local.to(torch.float32) - t_local.to(torch.float32)
            total = total + (f_local.to(torch.float32) * diff * diff).sum()

        return 0.5 * self.lambda_ * total

    # ------------------------------------------------------------------
    # Fisher computation -- run after a task finishes
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def compute_fisher(
        self,
        *,
        model: nn.Module,
        net: nn.Module,
        data_loader,
        num_batches: int = 200,
        loss_fn: Optional[Callable] = None,
        name_filter: Optional[Callable[[str], bool]] = None,
    ) -> int:
        """
        Estimate the empirical Fisher diagonal via gradient squared.

        Parameters
        ----------
        model:
            The full ImaginaireModel (used to call ``training_step``).
        net:
            The trainable sub-module whose parameters the Fisher is computed
            over (typically ``model.net`` -- the DiT backbone).
        data_loader:
            Yields task-specific training batches identical in shape to those
            used during regular training.
        num_batches:
            Maximum number of batches consumed (set 0 for the full loader).
        loss_fn:
            Optional callable ``loss_fn(model, batch, iteration) -> Tensor``.
            Defaults to the task-only Kendall/EDM loss from ``training_step``
            with ``include_ewc_penalty=False`` (Fisher must not include an
            already-loaded EWC penalty from a prior stage).
        name_filter:
            Optional predicate ``name -> bool``; only matching parameters get
            Fisher entries. Defaults to all trainable parameters.

        Returns
        -------
        Number of batches actually consumed.
        """
        rank = distributed.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            log.info(
                f"[EWC] Computing Fisher diagonal over up to {num_batches or 'ALL'} batches "
                f"(lambda={self.lambda_}, gamma={self.gamma})"
            )

        # We want gradients but no optimiser step. Keep the model in train()
        # mode so the loss matches the training distribution exactly.
        was_training = model.training
        model.train()

        # Default loss -- matches the regular training loss exactly.
        if loss_fn is None:
            def loss_fn(_model, _batch, _iter):
                _, _loss = _model.training_step(_batch, _iter, include_ewc_penalty=False)
                return _loss

        # Initialise per-parameter Fisher accumulators on local shards.
        fisher_new: TensorDict = {}
        ref_params: Dict[str, nn.Parameter] = {}
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if name_filter is not None and not name_filter(name):
                continue
            ref_params[name] = param
            local = _to_local(param)
            fisher_new[name] = torch.zeros_like(local, dtype=torch.float32)

        # The trainer's `LowPrecisionCallback.on_training_step_start` casts
        # all floating-point inputs to ``model.precision`` (typically bfloat16)
        # before the forward pass. We bypass the callback machinery here by
        # calling ``model.training_step`` directly, so we have to mirror the
        # cast manually -- otherwise float32 inputs would meet bfloat16 weights
        # in the first F.linear and torch raises a dtype mismatch.
        model_precision = getattr(model, "precision", None)

        n_done = 0
        for batch_idx, data_batch in enumerate(data_loader):
            if num_batches > 0 and batch_idx >= num_batches:
                break

            data_batch = misc.to(data_batch, device="cuda")
            if model_precision is not None and model_precision != torch.float32:
                data_batch = {
                    k: (v.to(dtype=model_precision)
                        if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                        else v)
                    for k, v in data_batch.items()
                }

            # Zero grads on every relevant parameter.
            for p in net.parameters():
                if p.grad is not None:
                    p.grad = None

            loss = loss_fn(model, data_batch, batch_idx)
            loss.backward()

            for name, param in ref_params.items():
                if param.grad is None:
                    continue
                grad_local = _to_local(param.grad).detach().to(torch.float32)
                fisher_new[name].add_(grad_local * grad_local)

            n_done += 1
            if rank == 0 and (batch_idx + 1) % 10 == 0:
                log.info(f"[EWC] Fisher batch {batch_idx + 1} / {num_batches or '?'}")

        # Final cleanup of grads we created.
        for p in net.parameters():
            if p.grad is not None:
                p.grad = None

        if n_done == 0:
            raise RuntimeError("[EWC] compute_fisher consumed zero batches; check the data loader.")

        # Average across batches.
        for name in fisher_new:
            fisher_new[name].div_(n_done)

        # Wrap each local accumulator back into the same DTensor layout as
        # `param`, casting to the per-parameter storage dtype (defaults to the
        # parameter's own dtype so the EWC state matches the base-stage model).
        new_fisher: TensorDict = {}
        for name, accum in fisher_new.items():
            ref = ref_params[name]
            target_dtype = self._store_dtype(ref)
            new_fisher[name] = _from_local_like(accum.to(target_dtype), ref)

        # Online update: F <- gamma * F + F_new ; theta* <- current params.
        if not self.fisher:
            self.fisher = new_fisher
        else:
            merged: TensorDict = {}
            for name, f_new in new_fisher.items():
                ref = ref_params[name]
                target_dtype = self._store_dtype(ref)
                if name in self.fisher:
                    f_old_local = _to_local(self.fisher[name]).to(torch.float32)
                    f_new_local = _to_local(f_new).to(torch.float32)
                    combined = (self.gamma * f_old_local + f_new_local).to(target_dtype)
                    merged[name] = _from_local_like(combined, ref)
                else:
                    merged[name] = f_new
            self.fisher = merged

        # Snapshot current parameters as theta* in the parameter's native dtype.
        self.prev_params = {}
        for name, param in ref_params.items():
            target_dtype = self._store_dtype(param)
            local = _to_local(param).detach().to(target_dtype).clone()
            self.prev_params[name] = _from_local_like(local, param)

        self.is_initialized = True

        # Restore prior train/eval state.
        if not was_training:
            model.eval()

        if rank == 0:
            log.info(f"[EWC] Fisher computed over {n_done} batches; tracked {len(self.fisher)} tensors.")
            _log_fisher_diagnostics(ref_params, self.fisher, n_done)
        return n_done

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _meta_dict(self) -> dict:
        return {
            "version": 3,
            "format": "dcp",
            "lambda_": self.lambda_,
            "gamma": self.gamma,
            "fisher_dtype": _fisher_dtype_label(self.fisher_dtype),
            "param_names": list(self.fisher.keys()),
        }

    def _apply_meta(
        self,
        meta: dict,
        *,
        override_lambda: Optional[float],
        override_gamma: Optional[float],
    ) -> None:
        if override_lambda is None:
            self.lambda_ = float(meta.get("lambda_", self.lambda_))
        else:
            self.lambda_ = float(override_lambda)
        if override_gamma is None:
            self.gamma = float(meta.get("gamma", self.gamma))
        else:
            self.gamma = float(override_gamma)

    def save(self, path: str) -> None:
        """
        Persist Fisher / theta* and hyper-parameters.

        - **Directory path** (no ``.pt`` suffix): DCP format (default). Each rank
          writes its local FSDP shards under ``<path>/ewc_state/``.
        - **``*.pt`` file path**: legacy single-file format (gathered full tensors).
        """
        if _is_legacy_pt_path(path):
            self._save_pt(path)
        else:
            self._save_dcp(path)

    def _save_pt(self, path: str) -> None:
        """Legacy: gather full tensors; rank 0 writes one ``.pt`` file."""
        rank = distributed.get_rank() if dist.is_initialized() else 0

        gathered_fisher: TensorDict = {}
        gathered_prev: TensorDict = {}
        for name, t in self.fisher.items():
            gathered_fisher[name] = _full_tensor(t).detach().cpu()
        for name, t in self.prev_params.items():
            gathered_prev[name] = _full_tensor(t).detach().cpu()

        if rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            payload = {
                "version": 2,
                "format": "pt",
                "lambda_": self.lambda_,
                "gamma": self.gamma,
                "fisher_dtype": _fisher_dtype_label(self.fisher_dtype),
                "fisher": gathered_fisher,
                "prev_params": gathered_prev,
            }
            torch.save(payload, path)
            log.success(
                f"[EWC] Saved EWC state (legacy .pt) to {path} "
                f"({len(gathered_fisher)} tensors, lambda={self.lambda_}, gamma={self.gamma})"
            )

        if dist.is_initialized():
            dist.barrier()

    def _save_dcp(self, dir_path: str) -> None:
        """Save sharded Fisher / theta* via PyTorch DCP (no full-tensor gather)."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemWriter

        rank = distributed.get_rank() if dist.is_initialized() else 0
        dir_path = dir_path.rstrip("/")
        tensor_dir = os.path.join(dir_path, _DCP_TENSOR_SUBDIR)

        if rank == 0:
            os.makedirs(dir_path, exist_ok=True)
            torch.save(self._meta_dict(), os.path.join(dir_path, _DCP_META_FILE))

        if dist.is_initialized():
            dist.barrier()

        state_dict = _flatten_ewc_tensors(self.fisher, self.prev_params)
        dcp.save(state_dict, storage_writer=FileSystemWriter(tensor_dir))

        if rank == 0:
            log.success(
                f"[EWC] Saved EWC state (DCP) to {dir_path}/ "
                f"({len(self.fisher)} tensors, lambda={self.lambda_}, gamma={self.gamma})"
            )

        if dist.is_initialized():
            dist.barrier()

    def load(
        self,
        path: str,
        net: nn.Module,
        *,
        override_lambda: Optional[float] = None,
        override_gamma: Optional[float] = None,
        strict: bool = False,
    ) -> None:
        """
        Load Fisher / theta* from disk and align them with ``net``.

        - **Directory**: DCP checkpoint at ``<path>/ewc_state/`` (recommended).
        - **``*.pt`` file**: legacy gathered format (re-shards to current FSDP layout).
        """
        path = path.rstrip("/")
        if os.path.isfile(path) and _is_legacy_pt_path(path):
            self._load_pt(path, net, override_lambda=override_lambda, override_gamma=override_gamma, strict=strict)
        elif os.path.isdir(path):
            dcp_root = _resolve_dcp_checkpoint_dir(path)
            self._load_dcp(dcp_root, net, override_lambda=override_lambda, override_gamma=override_gamma, strict=strict)
        else:
            raise FileNotFoundError(
                f"[EWC] state not found at {path} (expected a directory for DCP or a .pt file)"
            )

    def _load_pt(
        self,
        path: str,
        net: nn.Module,
        *,
        override_lambda: Optional[float],
        override_gamma: Optional[float],
        strict: bool,
    ) -> None:
        rank = distributed.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            log.info(f"[EWC] Loading EWC state (legacy .pt) from {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._apply_meta(payload, override_lambda=override_lambda, override_gamma=override_gamma)

        name_to_param = dict(net.named_parameters())
        loaded_fisher: TensorDict = {}
        loaded_prev: TensorDict = {}
        missing: list = []

        for name, full_t in payload["fisher"].items():
            if name not in name_to_param:
                missing.append(name)
                continue
            ref = name_to_param[name]
            loaded_fisher[name] = _reshard_full_to_ref(full_t, ref, self._store_dtype(ref))

        for name, full_t in payload["prev_params"].items():
            if name not in name_to_param:
                continue
            ref = name_to_param[name]
            loaded_prev[name] = _reshard_full_to_ref(full_t, ref, self._store_dtype(ref))

        if strict and missing:
            raise RuntimeError(
                f"[EWC] {len(missing)} tensors in the state file are missing from the model: {missing[:8]}..."
            )
        if missing and rank == 0:
            log.warning(f"[EWC] {len(missing)} tensors in the state file are missing from the model (skipped).")

        self.fisher, self.prev_params = _align_ewc_state_to_net(loaded_fisher, loaded_prev, net)
        self.is_initialized = True

        if rank == 0:
            log.success(
                f"[EWC] Loaded EWC state: {len(self.fisher)} tensors, "
                f"lambda={self.lambda_}, gamma={self.gamma}"
            )

        if dist.is_initialized():
            dist.barrier()

    def _load_dcp(
        self,
        dir_path: str,
        net: nn.Module,
        *,
        override_lambda: Optional[float],
        override_gamma: Optional[float],
        strict: bool,
    ) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader

        rank = distributed.get_rank() if dist.is_initialized() else 0
        dir_path = _resolve_dcp_checkpoint_dir(dir_path)
        meta_path = os.path.join(dir_path, _DCP_META_FILE)
        tensor_dir = os.path.join(dir_path, _DCP_TENSOR_SUBDIR)
        if not os.path.isfile(meta_path) or not os.path.isdir(tensor_dir):
            hint = (
                f"Expected layout:\n  {dir_path}/{_DCP_META_FILE}\n  {dir_path}/{_DCP_TENSOR_SUBDIR}/\n"
                "If you only have a legacy Fisher file from compute_ewc_fisher, set "
                "ewc_prev_state_path to the .pt file (e.g. .../base_stage_iter7000.pt), not a directory."
            )
            # Helpful sibling check: same folder may contain a .pt from an older Fisher run.
            parent = os.path.dirname(dir_path) or "."
            pt_candidates = [
                os.path.join(parent, f)
                for f in os.listdir(parent)
                if f.endswith(".pt") and os.path.isfile(os.path.join(parent, f))
            ] if os.path.isdir(parent) and rank == 0 else []
            if pt_candidates:
                hint += f"\nFound legacy .pt file(s) nearby: {pt_candidates[:3]}"
            raise FileNotFoundError(
                f"[EWC] DCP state incomplete at {dir_path} (need {_DCP_META_FILE} and {_DCP_TENSOR_SUBDIR}/). {hint}"
            )

        if rank == 0:
            log.info(f"[EWC] Loading EWC state (DCP) from {dir_path}")

        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        self._apply_meta(meta, override_lambda=override_lambda, override_gamma=override_gamma)

        name_to_param = dict(net.named_parameters())
        param_names = meta.get("param_names", list(name_to_param.keys()))
        load_sd: Dict[str, torch.Tensor] = {}
        missing: list = []

        for name in param_names:
            if name not in name_to_param:
                missing.append(name)
                continue
            ref = name_to_param[name]
            load_sd[f"{_FISHER_PREFIX}{name}"] = _empty_shell_like(ref)
            load_sd[f"{_PREV_PREFIX}{name}"] = _empty_shell_like(ref)

        dcp.load(load_sd, storage_reader=FileSystemReader(tensor_dir))

        loaded_fisher, loaded_prev = _unflatten_ewc_tensors(load_sd)
        self.fisher, self.prev_params = _align_ewc_state_to_net(loaded_fisher, loaded_prev, net)

        if strict and missing:
            raise RuntimeError(
                f"[EWC] {len(missing)} tensors in the DCP state are missing from the model: {missing[:8]}..."
            )
        if missing and rank == 0:
            log.warning(f"[EWC] {len(missing)} tensors in the DCP state are missing from the model (skipped).")

        self.is_initialized = True

        if rank == 0:
            log.success(
                f"[EWC] Loaded EWC state (DCP): {len(self.fisher)} tensors, "
                f"lambda={self.lambda_}, gamma={self.gamma}"
            )

        if dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "OnlineEWC":
        """Move Fisher and theta* to a target device without changing FSDP sharding."""
        for store in (self.fisher, self.prev_params):
            for k in list(store.keys()):
                t = store[k]
                if _is_dtensor(t):
                    local = t.to_local().to(device)
                    store[k] = DTensor.from_local(
                        local,
                        device_mesh=t.device_mesh,
                        placements=t.placements,
                        run_check=False,
                    )
                else:
                    store[k] = t.to(device)
        return self


def _log_fisher_diagnostics(
    ref_params: Dict[str, nn.Parameter],
    fisher: TensorDict,
    n_batches: int,
) -> None:
    """Rank-0 summary to sanity-check Fisher estimation."""
    zero_fisher_names: list[str] = []
    fisher_vals: list[float] = []

    for name, param in ref_params.items():
        if name not in fisher:
            continue
        f_local = _to_local(fisher[name]).detach().float()
        fisher_vals.append(f_local.mean().item())
        if f_local.abs().max().item() == 0.0:
            zero_fisher_names.append(name)
    if not fisher_vals:
        log.warning("[EWC] Fisher diagnostics: no Fisher tensors to summarize.")
        return

    import statistics

    mean_f = statistics.mean(fisher_vals)
    max_f = max(fisher_vals)
    log.info(
        f"[EWC] Fisher diagnostics ({n_batches} batches): "
        f"mean(F)≈{mean_f:.4g}, max(mean F per tensor)≈{max_f:.4g}, "
        f"num_tensors={len(fisher_vals)}, zero_F={len(zero_fisher_names)}"
    )
    if mean_f < 1e-8:
        log.warning(
            "[EWC] Fisher values are extremely small — penalty may be negligible vs task loss. "
            "Consider ewc_fisher_dtype=float32, more Fisher batches, or verify the experiment "
            "dataloader matches the checkpoint (base Fisher needs base-task data)."
        )
    if len(zero_fisher_names) > 0:
        log.warning(
            f"[EWC] {len(zero_fisher_names)} tensors have zero Fisher "
            f"(showing up to 3): {zero_fisher_names[:3]}"
        )


def parse_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """
    Resolve a dtype string for the EWC config.

    - ``None`` / ``""`` / ``"auto"`` / ``"match"`` -> ``None`` (sentinel meaning
      "match the model parameter's dtype on the fly").
    - Any other recognised name (``"bfloat16"``, ``"float32"``, ...) returns
      the corresponding ``torch.dtype`` and forces that storage precision.
    """
    if name is None:
        return None
    name = str(name).lower().replace("torch.", "").strip()
    if name in ("", "auto", "match", "none"):
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unknown EWC dtype: {name!r}. Use 'auto' to match the model dtype.")
    return mapping[name]
