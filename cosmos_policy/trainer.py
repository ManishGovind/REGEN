# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extended Trainer for Cosmos Policy with epoch tracking, online-EWC, and PackNet support.

This trainer extends the base ImaginaireTrainer to add:
- Epoch tracking and sampler epoch setting for proper distributed sampling
- Optional end-of-training Fisher diagonal computation + save for online EWC.
  When ``model.config.ewc_enabled`` is True and ``model.config.ewc_save_state_path``
  is set, the Fisher diagonal is estimated over the current task's training data
  immediately after the final iteration (before the trainer shuts down) and then
  persisted alongside the regular checkpoint. The saved file can be passed to a
  subsequent CL stage via ``model.config.ewc_prev_state_path``.
- Optional end-of-training PackNet prune + post-prune fine-tune + mask save.
  When ``model.config.packnet_enabled`` is True, pre-end-prune masks are saved first
  (for eval with the last training checkpoint), then masks are pruned after training
  and optionally fine-tuned for ``packnet_post_prune_iters`` iterations. When
  ``packnet_save_post_prune_ckpt`` is True (default), a checkpoint is written at
  ``main_iter + packnet_post_prune_iters`` after post-prune fine-tune.
- Optional base post-prune retrain after ``init_from_pretrained_base`` for
  ``packnet_base_post_prune_iters`` iterations on ``dataloader_packnet_base_retrain``.
"""

import signal

import torch
import torch.utils.data

from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_policy._src.imaginaire.utils import distributed, log, misc
from cosmos_policy._src.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling


class CosmosPolicyTrainer(ImaginaireTrainer):
    """
    Extended Trainer for Cosmos Policy.

    Adds special handling for:
    - Epoch tracking to properly set dataloader sampler epochs (needed for distributed training)
    - Simplified initial validation check (removes run_validation_on_start requirement)
    - End-of-training online-EWC Fisher computation + save (see module docstring)
    - End-of-training PackNet prune / post-prune fine-tune / mask save
    """

    def __init__(self, config):
        super().__init__(config)

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
        **kwargs,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        # EWC must be loaded after weights are on ``net`` (FSDP shard layout is final).
        if hasattr(model, "load_ewc_state_if_configured"):
            model.load_ewc_state_if_configured()
        if hasattr(model, "load_packnet_state_if_configured"):
            model.load_packnet_state_if_configured()
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        dataloader_packnet_base_retrain = kwargs.get("dataloader_packnet_base_retrain")

        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)
        self._maybe_packnet_base_post_prune_retrain(
            model,
            model_ddp,
            dataloader_packnet_base_retrain,
            optimizer,
            scheduler,
            grad_scaler,
            iteration=iteration,
        )
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0 and self.config.trainer.run_validation_on_start:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            epoch = 0
            while True:
                dataloader_train.sampler.set_epoch(epoch)
                dataloader_train_iter = iter(dataloader_train)
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # Do the following when an actual optimizer (update) step has been made.
                    iteration += 1
                    # Save checkpoint.
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    self.straggler_detector.generate_report(iteration)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                epoch += 1
                if _end_training:
                    break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)

        # Online EWC: estimate Fisher on the task we just finished and persist it.
        self._maybe_finalize_ewc(model, dataloader_train, iteration=iteration)

        # PackNet: prune, optional post-prune fine-tune, persist masks.
        self._maybe_finalize_packnet(
            model,
            model_ddp,
            dataloader_train,
            optimizer,
            scheduler,
            grad_scaler,
            iteration=iteration,
        )

        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    # ------------------------------------------------------------------
    # Online-EWC end-of-task hook
    # ------------------------------------------------------------------

    def _maybe_finalize_ewc(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        iteration: int,
    ) -> None:
        """Run ``OnlineEWC.compute_fisher`` + ``save`` if the config requests it.

        This is a no-op unless ``model.config.ewc_enabled`` is True and
        ``model.config.ewc_save_state_path`` is a non-empty string.
        """
        cfg = getattr(model, "config", None)
        if cfg is None:
            return
        if not getattr(cfg, "ewc_enabled", False):
            return
        save_path = getattr(cfg, "ewc_save_state_path", None)
        if not save_path:
            log.info(
                "[EWC] ewc_enabled=True but ewc_save_state_path is empty -- skipping "
                "Fisher computation at end of training."
            )
            return
        if getattr(model, "ewc", None) is None:
            log.warning("[EWC] model.ewc is None despite ewc_enabled=True -- skipping.")
            return

        log.critical(
            f"[EWC] Finalising EWC for this task: computing Fisher over up to "
            f"{cfg.ewc_num_fisher_batches} batches, saving to {save_path}."
        )

        # Make sure the dataloader's sampler is reset so Fisher batches are drawn
        # from a fresh epoch (avoids reusing the very last batch over and over).
        if hasattr(dataloader_train, "sampler") and hasattr(dataloader_train.sampler, "set_epoch"):
            dataloader_train.sampler.set_epoch(iteration + 1)

        try:
            model.ewc.compute_fisher(
                model=model,
                net=model.net,
                data_loader=dataloader_train,
                num_batches=cfg.ewc_num_fisher_batches,
            )
            model.ewc.save(save_path)
        except Exception:  # noqa: BLE001 -- never fail the trainer because of EWC
            log.exception("[EWC] compute_fisher / save failed; continuing shutdown.")

    def _maybe_finalize_packnet(
        self,
        model: ImaginaireModel,
        model_ddp: torch.nn.Module,
        dataloader_train: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save pre-end-prune masks, prune, optionally post-prune fine-tune, save post-prune masks."""
        cfg = getattr(model, "config", None)
        if cfg is None or not getattr(cfg, "packnet_enabled", False):
            return
        if getattr(model, "packnet", None) is None:
            log.warning("[PackNet] packnet_enabled=True but model.packnet is None -- skipping.")
            return

        save_path = getattr(cfg, "packnet_save_state_path", None)
        pre_prune_path = getattr(cfg, "packnet_save_pre_prune_state_path", None)
        if not pre_prune_path and save_path:
            pre_prune_path = save_path.rstrip("/") + "_pre_prune"
        if pre_prune_path:
            try:
                log.critical(
                    f"[PackNet] Saving pre-end-prune masks (for eval with last training ckpt) to {pre_prune_path}"
                )
                model.packnet.save(pre_prune_path)
            except Exception:  # noqa: BLE001
                log.exception("[PackNet] pre-end-prune save() failed; continuing shutdown.")

        log.critical(
            f"[PackNet] Finalising task {cfg.packnet_task_id}: pruning "
            f"~{100 * cfg.packnet_prune_perc:.1f}% of current-task weights."
        )
        try:
            model.packnet.prune(model.net)
        except Exception:  # noqa: BLE001
            log.exception("[PackNet] prune() failed; continuing shutdown.")
            return

        post_iters = int(getattr(cfg, "packnet_post_prune_iters", 0) or 0)
        if post_iters > 0:
            log.critical(f"[PackNet] Post-prune fine-tuning for {post_iters} iterations.")
            self._run_packnet_post_prune_finetune(
                model,
                model_ddp,
                dataloader_train,
                optimizer,
                scheduler,
                grad_scaler,
                num_iters=post_iters,
                start_iteration=iteration,
                phase_name="post-prune",
            )
            if getattr(cfg, "packnet_save_post_prune_ckpt", True):
                post_prune_iter = iteration + post_iters
                log.critical(
                    f"[PackNet] Saving post-prune fine-tune checkpoint at iter {post_prune_iter} "
                    f"(matches post-prune masks in {save_path or 'packnet_save_state_path'})."
                )
                try:
                    self.checkpointer.save(
                        model, optimizer, scheduler, grad_scaler, iteration=post_prune_iter
                    )
                except Exception:  # noqa: BLE001
                    log.exception("[PackNet] post-prune checkpoint save failed; continuing shutdown.")

        if save_path:
            try:
                log.critical(f"[PackNet] Saving post-prune masks (for next CL stage) to {save_path}")
                model.packnet.save(save_path)
            except Exception:  # noqa: BLE001
                log.exception("[PackNet] save() failed; continuing shutdown.")

    def _maybe_packnet_base_post_prune_retrain(
        self,
        model: ImaginaireModel,
        model_ddp: torch.nn.Module,
        dataloader_base: torch.utils.data.DataLoader | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Retrain the pruned base subnet on prior-task data (PackNet paper step before new-task training)."""
        cfg = getattr(model, "config", None)
        if cfg is None or not getattr(cfg, "packnet_enabled", False):
            return
        if getattr(model, "packnet", None) is None:
            return
        if not getattr(cfg, "packnet_init_from_base", False):
            return
        if iteration != 0:
            return

        base_iters = int(getattr(cfg, "packnet_base_post_prune_iters", 0) or 0)
        if base_iters <= 0:
            return
        if dataloader_base is None:
            log.warning(
                "[PackNet] packnet_base_post_prune_iters > 0 but no dataloader_packnet_base_retrain "
                "was provided -- skipping base post-prune retrain."
            )
            return

        base_task_id = int(getattr(cfg, "packnet_base_task_id", 0))
        cl_task_id = model.packnet.task_id
        log.critical(
            f"[PackNet] Base post-prune retrain for {base_iters} iterations on task "
            f"{base_task_id} data (kept base subnet only)."
        )
        model.packnet.task_id = base_task_id
        try:
            self._run_packnet_post_prune_finetune(
                model,
                model_ddp,
                dataloader_base,
                optimizer,
                scheduler,
                grad_scaler,
                num_iters=base_iters,
                start_iteration=iteration,
                phase_name="base post-prune",
            )
        finally:
            model.packnet.task_id = cl_task_id

    def _run_packnet_post_prune_finetune(
        self,
        model: ImaginaireModel,
        model_ddp: torch.nn.Module,
        dataloader_train: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        num_iters: int,
        start_iteration: int,
        phase_name: str = "post-prune",
    ) -> None:
        """Fine-tune the pruned subnet (mask zeros stay frozen; LIBERO post_prune_epochs)."""
        if hasattr(dataloader_train, "sampler") and hasattr(dataloader_train.sampler, "set_epoch"):
            dataloader_train.sampler.set_epoch(start_iteration + 1)

        data_iter = iter(dataloader_train)
        grad_accum_iter = 0
        for step in range(num_iters):
            try:
                data_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader_train)
                data_batch = next(data_iter)

            data_batch = misc.to(data_batch, device="cuda")
            if not model.training:
                model_ddp.train()
            step_iteration = start_iteration + step + 1
            # Minimal callbacks only: full step_end triggers timer/wandb callbacks that
            # expect the main training loop's dataloading/forward timers (empty here).
            self.callbacks.on_training_step_start(model, data_batch, iteration=step_iteration)
            output_batch, loss, grad_accum_iter = self.training_step(
                model_ddp,
                optimizer,
                scheduler,
                grad_scaler,
                data_batch,
                iteration=step_iteration,
                grad_accum_iter=grad_accum_iter,
            )
            if distributed.is_rank0() and step % 50 == 0:
                log.info(
                    f"[PackNet] {phase_name} step {step}/{num_iters} loss={loss.detach().item():.4f}"
                )
