"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import logging
import os
from collections import defaultdict
from typing import Optional
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import ase
import numpy as np
import torch
from tqdm import tqdm
import torch_geometric
from torch_scatter import scatter

from adsorbdiff.utils import distutils
from adsorbdiff.utils.registry import registry
from adsorbdiff.utils.utils import check_traj_files
from adsorbdiff.utils.utils import load_state_dict, save_checkpoint
from adsorbdiff.modules.normalizer import Normalizer
from adsorbdiff.modules.evaluator import Evaluator
from adsorbdiff.utils.typing import assert_is_instance
from adsorbdiff.datasets.lmdb_dataset import data_list_collater
from adsorbdiff.modules.scaling.util import ensure_fitted
from adsorbdiff.relaxation.ml_relaxation import ml_diffuse
from adsorbdiff.modules.normalizer import Normalizer
from adsorbdiff.modules.scaling.compat import load_scales_compat
from adsorbdiff.modules.scaling.util import ensure_fitted
from adsorbdiff.models.equiformer_v2.trainers.lr_scheduler import LRScheduler
from adsorbdiff.trainers import OCPTrainer
from adsorbdiff.utils import rot_utils
from adsorbdiff.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)


@torch.no_grad()
def pbc_correction(noise_vec, batch):
    corr_noise_vec = torch.zeros(noise_vec.shape, device=noise_vec.device)
    for batch_idx in range(batch.batch.max() + 1):
        if noise_vec.shape[0] == batch.batch.shape[0]:
            mask = batch.batch == batch_idx
        else:
            mask = batch_idx

        fractional = torch.linalg.solve(
            batch.cell[batch_idx].t().double(), noise_vec[mask].t().double()
        ).t()
        fractional %= 1.0
        fractional %= 1.0
        assert fractional.max() <= 1.0 and fractional.min() >= 0.0
        fractional[fractional > 0.5] -= 1
        corr_noise_vec[mask] = torch.matmul(
            fractional.float(), batch.cell[batch_idx].float()
        )
    return corr_noise_vec


def tr_so3_schedule(batch, denoise_pos_params):
    ads_std_low = denoise_pos_params["ads_std_low"]
    ads_std_high = denoise_pos_params["ads_std_high"]
    free_std_low = denoise_pos_params["free_std_low"]
    free_std_high = denoise_pos_params["free_std_high"]
    rot_std_low = denoise_pos_params["rot_std_low"]
    rot_std_high = denoise_pos_params["rot_std_high"]
    num_steps = denoise_pos_params["num_steps"]

    t = torch.rand(size=(batch.natoms.size(0),), device=batch.pos.device)
    tr_sigma = ads_std_low ** (1 - t) * ads_std_high**t
    rot_sigma = rot_std_low ** (1 - t) * rot_std_high**t

    tags = batch.tags
    batch.tr_sigma = tr_sigma[:, None]  # (natoms, 1)
    batch.rot_sigma = rot_sigma[:, None]  # (natoms, 1)

    ads_center = scatter(
        batch.pos[tags == 2], batch.batch[tags == 2], dim=0, reduce="mean"
    )

    ads_center_noise_vec = torch.zeros(
        ads_center.shape, device=batch.pos.device
    )
    ads_center_noise_vec = ads_center_noise_vec.normal_() * tr_sigma[:, None]
    ads_center_noise_vec = pbc_correction(ads_center_noise_vec, batch)

    # Add noise only to XY coordinate
    ads_center_noise_vec[:, -1] = 0

    # fractional = torch.linalg.solve(batch.cell, ads_center + ads_center_noise_vec)
    # fractional %= 1
    # fractional %= 1
    # ads_center_noise_vec = torch.einsum("bi,bij->bj", fractional, batch.cell.transpose(1, 2)) - ads_center

    ads_pos = batch.pos[tags == 2]

    batch_rot_score, new_ads_pos = [], []
    for batch_idx in range(batch.batch.max() + 1):
        batch_mask = batch.batch[tags == 2] == batch_idx
        rot_update = rot_utils.sample_vec(eps=rot_sigma[batch_idx].item())
        rot_mat = rot_utils.axis_angle_to_matrix(
            torch.tensor(rot_update)
        ).float()
        rot_score = (
            torch.from_numpy(
                rot_utils.score_vec(
                    vec=rot_update, eps=rot_sigma[batch_idx].item()
                )
            )
            .float()
            .unsqueeze(0)
        )
        batch_rot_score.append(rot_score)
        new_ads_pos.append(
            (ads_pos[batch_mask] - ads_center[batch_idx]) @ rot_mat.T
            + ads_center_noise_vec[batch_idx]
            + ads_center[batch_idx]
        )
    new_ads_pos = torch.cat(new_ads_pos)

    # Move the adsorbate up by roughly 1A
    new_ads_pos[:, -1] += 1

    batch.rot_score = torch.cat(batch_rot_score)
    batch.pos[tags == 2] = new_ads_pos
    batch.ads_center_noise_vec = ads_center_noise_vec
    batch.tr_score = -ads_center_noise_vec / tr_sigma[:, None] ** 2
    return batch


def ads_COM_gaussian_schedule(batch, denoise_pos_params):
    ads_std_low = denoise_pos_params["ads_std_low"]
    ads_std_high = denoise_pos_params["ads_std_high"]
    num_steps = denoise_pos_params["num_steps"]

    t_tr = torch.rand(size=(batch.natoms.size(0),), device=batch.pos.device)
    tr_sigma = ads_std_low ** (1 - t_tr) * ads_std_high**t_tr

    tags = batch.tags
    batch.tr_sigma = tr_sigma[:, None]  # (natoms, 1)

    ads_center = scatter(
        batch.pos[tags == 2], batch.batch[tags == 2], dim=0, reduce="mean"
    )

    ads_center_noise_vec = torch.zeros(
        ads_center.shape, device=batch.pos.device
    )
    ads_center_noise_vec = ads_center_noise_vec.normal_() * tr_sigma[:, None]

    # Add noise only to XY coordinate
    ads_center_noise_vec[:, -1] = 0

    ads_center += ads_center_noise_vec

    fractional = torch.linalg.solve(batch.cell, ads_center)
    fractional %= 1
    fractional %= 1
    ads_center = torch.einsum(
        "bi,bij->bj", fractional, batch.cell.transpose(1, 2)
    )

    # Move the adsorbate up by roughly 1A
    ads_center[:, -1] += 1

    batch.pos[tags == 2] = ads_center[batch.batch][tags == 2]

    batch.ads_center_noise_vec = ads_center_noise_vec
    batch.tr_score = -ads_center_noise_vec / tr_sigma[:, None] ** 2
    return batch


class DenoisingTrainer(OCPTrainer):
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        name="ocp",
    ):
        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_fns=loss_fns,
            eval_metrics=eval_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            name=name,
        )

        self.evaluator = Evaluator(task=name)
        # for denoising positions
        self.use_denoising_pos = self.config["optim"]["use_denoising_pos"]
        if (
            "denoising_pos_params" in self.config["optim"]
            and self.config["optim"]["denoising_pos_params"] is not None
        ):
            self.denoising_pos_params = self.config["optim"][
                "denoising_pos_params"
            ]

    def load_extras(self) -> None:
        def multiply(obj, num):
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = obj[i] * num
            else:
                obj = obj * num
            return obj

        self.config["optim"]["scheduler_params"]["epochs"] = self.config[
            "optim"
        ]["max_epochs"]
        self.config["optim"]["scheduler_params"]["lr"] = self.config["optim"][
            "lr_initial"
        ]

        # convert epochs into number of steps
        if self.train_loader is None:
            logging.warning("Skipping scheduler setup. No training set found.")
            self.scheduler = None
        else:
            n_iter_per_epoch = len(self.train_loader)
            scheduler_params = self.config["optim"]["scheduler_params"]
            for k in scheduler_params.keys():
                if "epochs" in k:
                    if isinstance(scheduler_params[k], (int, float)):
                        scheduler_params[k] = int(
                            multiply(scheduler_params[k], n_iter_per_epoch)
                        )
                    elif isinstance(scheduler_params[k], list):
                        scheduler_params[k] = [
                            int(x)
                            for x in multiply(
                                scheduler_params[k], n_iter_per_epoch
                            )
                        ]
            self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        evaluator = Evaluator(
            task=self.name,
            # eval_metrics=self.evaluation_metrics.get(
            #     "metrics", Evaluator.task_metrics.get(self.name, {})
            # ),
        )

        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            if hasattr(batch, "pos_relaxed"):
                batch.pos = batch.pos_relaxed

            if self.config["model_attributes"].get("so3_denoising", False):
                batch = tr_so3_schedule(batch, self.denoising_pos_params)
            else:
                batch = ads_COM_gaussian_schedule(
                    batch, self.denoising_pos_params
                )

            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward_denoising(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    def train(self, disable_eval_tqdm=False):
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )

        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        nan_count = 0

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                if hasattr(batch, "pos_relaxed"):
                    batch.pos = batch.pos_relaxed

                if self.config["model_attributes"].get("so3_denoising", False):
                    batch = tr_so3_schedule(batch, self.denoising_pos_params)
                else:
                    batch = ads_COM_gaussian_schedule(
                        batch, self.denoising_pos_params
                    )

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward_denoising(batch)
                    loss = self._compute_loss(out, batch)

                if torch.isnan(loss).any():
                    logging.warning("NaN loss detected, skipping step")
                    nan_count += 1
                    continue
                else:
                    nan_count = 0

                if loss > 1e6:
                    logging.warning(f"Loss too high: {loss.item()}")
                    break
                if nan_count > 10:
                    logging.warning("Too many NaN losses, stopping training")
                    break
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    or i == 0
                    or i == (len(self.train_loader) - 1)
                ) and distutils.is_master():
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0 or i == (
                    len(self.train_loader) - 1
                ):
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val", disable_tqdm=disable_eval_tqdm
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            traj_dir = self.config["task"]["relax_opt"][
                                "traj_dir"
                            ]
                            traj_dir = os.join(traj_dir, self.identifier)
                            traj_dir += f"_step{self.step}"
                            self.config["task"]["relax_opt"][
                                "traj_dir"
                            ] = traj_dir
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward_denoising(self, batch):

        if not self.config["model_attributes"].get("so3_denoising", False):
            out_positions = self.model(batch.to(self.device))
            out = {"positions": out_positions}
        else:
            # this first position gives the prediction for ads COM and
            # the second position gives the prediction for other atoms
            out_positions1, out_positions2 = self.model(batch.to(self.device))
            out = {
                "positions": out_positions1,
                "positions_free": out_positions2,
            }

        return out

    @torch.no_grad()
    def predict_denoising(
        self,
        data_loader,
        per_image: bool = True,
        results_file=None,
        disable_tqdm: bool = False,
    ):

        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [data_loader]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        predictions = {"id": [], "positions": [], "chunk_idx": []}
        if self.config["model_attributes"].get("so3_denoising", False):
            predictions["positions_free"] = []

        for i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward_denoising(batch)

            if per_image:
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(batch.sid.tolist(), batch.fid.tolist())
                ]
                predictions["id"].extend(systemids)
                batch_natoms = batch.natoms
                batch_fixed = batch.fixed

                # total energy target requires predictions to be saved in float32
                # default is float16

                predictions["positions"].extend(
                    out["positions"].cpu().detach().to(torch.float16).numpy()
                )
                positions_free = (
                    out["positions_free"].cpu().detach().to(torch.float16)
                )
                per_image_positions_free = torch.split(
                    positions_free, batch_natoms.tolist()
                )
                per_image_positions_free = [
                    force.numpy() for force in per_image_positions_free
                ]
                # evalAI only requires forces on free atoms
                if results_file is not None:
                    _per_image_fixed = torch.split(
                        batch_fixed, batch_natoms.tolist()
                    )
                    _per_image_free_positions = [
                        force[(fixed == 0).tolist()]
                        for force, fixed in zip(
                            per_image_positions_free, _per_image_fixed
                        )
                    ]
                    _chunk_idx = np.array(
                        [
                            free_force.shape[0]
                            for free_force in _per_image_free_positions
                        ]
                    )
                    per_image_positions_free = _per_image_free_positions
                    predictions["chunk_idx"].extend(_chunk_idx)
                predictions["positions_free"].extend(per_image_positions_free)
            else:
                predictions["positions"] = out["positions"].detach()
                if "positions_free" in out:
                    predictions["positions_free"] = out[
                        "positions_free"
                    ].detach()
                if self.ema:
                    self.ema.restore()
                return predictions
        if "positions_free" in predictions:
            predictions["positions_free"] = np.array(
                predictions["positions_free"], dtype=object
            )
            predictions["chunk_idx"] = np.array(
                predictions["chunk_idx"],
            )
        predictions["positions"] = np.array(predictions["positions"])
        predictions["id"] = np.array(
            predictions["id"],
        )
        self.save_results(
            predictions,
            results_file,
            keys=["positions", "positions_free", "chunk_idx"],
        )

        if self.ema:
            self.ema.restore()

        return predictions

    def _compute_loss(self, out, batch):
        loss = []

        ads_noise_target = batch.tr_score.to(self.device)

        denoising_pos_mult = self.config["optim"].get(
            "denoising_pos_coefficient", 1
        )

        out["positions"] = scatter(
            out["positions"][batch.tags == 2],
            batch.batch[batch.tags == 2],
            dim=0,
            reduce="mean",
        )
        out["positions"] /= batch.tr_sigma
        out["positions"][:, -1] = 0

        energy_mask = torch.ones((batch.batch.max() + 1,), device=self.device)

        loss.append(
            (
                (out["positions"] - ads_noise_target) ** 2
                * batch.tr_sigma**2
                * energy_mask.unsqueeze(-1)
            ).mean()
        )

        if self.config["model_attributes"].get("so3_denoising", False):
            rot_target = batch.rot_score.to(self.device)
            out["positions_free"] = scatter(
                out["positions_free"][batch.tags == 2],
                batch.batch[batch.tags == 2],
                dim=0,
                reduce="mean",
            )
            out["positions_free"] /= batch.rot_sigma
            rot_score_norm = rot_utils.score_norm(batch.rot_sigma.cpu()).to(
                self.device
            )
            loss.append(
                (
                    ((out["positions_free"] - rot_target) / rot_score_norm)
                    ** 2
                    * energy_mask.unsqueeze(-1)
                ).mean()
            )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch, evaluator, metrics={}):
        target = {
            "natoms": torch.ones(out["positions"].shape[0], dtype=torch.long),
            "cell": batch.cell,
            "pbc": torch.tensor([True, True, True]),
        }

        target["positions"] = batch.ads_center_noise_vec

        out["natoms"] = target["natoms"]
        out["cell"] = target["cell"]
        out["pbc"] = target["pbc"]

        metrics = evaluator.eval(
            out,
            target,
            prev_metrics=metrics,
        )
        return metrics

    def run_relaxations(self, split: str = "val") -> None:
        ensure_fitted(self._unwrapped_model)

        # When set to true, uses deterministic CUDA scatter ops, if available.
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # Only implemented for GemNet-OC currently.
        registry.register(
            "set_deterministic_scatter",
            self.config["task"].get("set_deterministic_scatter", False),
        )

        logging.info("Running ML-diffusion")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()
            
        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                try:
                    logging.info(f"Skipping batch: {batch.sid.tolist()}")
                except:
                    logging.info(f"Skipping batch: {batch.sid}")
                continue

            relaxed_batch = ml_diffuse(
                batch=batch,
                model=self,
                denoising_pos_params=self.denoising_pos_params,
                traj_dir=self.config["task"]["relax_opt"].get(
                    "traj_dir", None
                ),
                save_full_traj=self.config["task"].get("save_full_traj", True),
                device=self.device,
                transform=None,
                logger=self.logger,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()

        registry.unregister("set_deterministic_scatter")
