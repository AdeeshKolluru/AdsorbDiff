import logging
from collections import deque
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import ase
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter

from adsorbdiff.relaxation.ase_utils import batch_to_atoms
from adsorbdiff.utils.utils import radius_graph_pbc
from adsorbdiff.utils.rot_utils import axis_angle_to_matrix


class Denoiser:
    def __init__(
        self,
        batch: Batch,
        model: "TorchCalc",
        denoising_pos_params: dict,
        device: str = "cuda:0",
        save_full_traj: bool = True,
        traj_dir: Optional[Path] = None,
        traj_names=None,
        early_stop_batch: bool = False,
        logger=None,
    ) -> None:
        self.batch = batch
        self.model = model
        self.device = device
        self.save_full = save_full_traj
        self.traj_dir = traj_dir
        self.traj_names = traj_names
        self.early_stop_batch = early_stop_batch
        self.otf_graph = model.model._unwrapped_model.otf_graph
        self.denoising_pos_params = denoising_pos_params

        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"
        logging.info("Step   Fmax(eV/A)")

        if not self.otf_graph and "edge_index" not in batch:
            self.model.update_graph(self.batch)

        if "saved_traj_dir" in denoising_pos_params:
            self.get_pos_from_traj(denoising_pos_params["saved_traj_dir"])

    def set_positions(self, update, update_mask) -> None:
        if not self.early_stop_batch:
            update = torch.where(update_mask.unsqueeze(1), update, 0.0)

        self.batch.pos += update.to(dtype=torch.float32)

        if not self.otf_graph:
            self.model.update_graph(self.batch)

    def run(self):

        self.trajectories = None
        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            self.trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj_tmp", mode="w")
                for name in self.traj_names
            ]

        self.reverse_sde_sampling_rot()

        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()

        if self.trajectories is not None:
            for traj in self.trajectories:
                traj.close()
            for name in self.traj_names:
                traj_fl = Path(self.traj_dir / f"{name}.traj_tmp", mode="w")
                traj_fl.rename(traj_fl.with_suffix(".traj"))

        return self.batch

    def get_pos_from_traj(self, saved_traj_dir):
        import glob

        self.sid_to_pos = {}
        traj_paths = glob.glob(os.path.join(traj_dir, "*.traj"))
        for traj in traj_paths:
            sid = traj.split("/")[-1].split(".")[0]
            pos = ase.io.read(traj).get_positions()
            self.sid_to_pos[sid] = torch.tensor(pos)

    def reverse_sde_sampling(self):
        if "ads_std_low" in self.denoising_pos_params:
            std_low = self.denoising_pos_params["ads_std_low"]
            std_high = self.denoising_pos_params["ads_std_high"]
            num_steps = self.denoising_pos_params["num_steps"]

            tr_schedule = torch.tensor(
                np.linspace(1, 0, num_steps + 1)[:-1],
                dtype=torch.float32,
                device=self.batch.pos.device,
            )

            noise = torch.rand(self.batch.batch.max() + 1, 3)

            ads_com_noise = torch.einsum(
                "bi,bij->bj", noise, self.batch.cell.transpose(1, 2)
            )

            ads_init_pos = self.batch.pos[self.batch.tags == 2]
            ads_init_com = self._get_ads_output(self.batch.pos)

            # Only for train subsplit start by 1A up
            # ads_init_com[:, -1] += 1
            # ads_init_pos[:, -1] += 1

            ads_com_noise[:, -1] = ads_init_com[:, -1]
            ads_init_pos_rel = (
                ads_init_pos
                - ads_init_com[self.batch.batch][self.batch.tags == 2]
            )
            ads_pos = (
                ads_init_pos_rel
                + ads_com_noise[self.batch.batch][self.batch.tags == 2]
            )
            self.batch.pos[self.batch.tags == 2] = ads_pos

            cvg_count = 0
            logging.info("Starting reverse sde samling:")
            for t_idx in tqdm(range(num_steps)):
                sigma = (
                    std_low ** (1 - tr_schedule[t_idx])
                    * std_high ** tr_schedule[t_idx]
                )
                tr_g = sigma * (2 * np.log(std_high / std_low)) ** 0.5
                dt_tr = (
                    tr_schedule[t_idx] - tr_schedule[t_idx + 1]
                    if t_idx < num_steps - 1
                    else tr_schedule[t_idx]
                )
                noise_pred, _ = self.model.get_denoising_prediction(self.batch)
                noise_pred = self._get_ads_output(noise_pred)

                # ODE case
                updated_ads_pos = 0.5 * tr_g**2 * dt_tr * noise_pred

                # PBC handling
                ads_com_pos = self._get_ads_output(self.batch.pos)
                updated_ads_pos[:, -1] = 0
                fractional = torch.linalg.solve(
                    self.batch.cell, ads_com_pos + updated_ads_pos
                )
                fractional %= 1
                fractional %= 1
                updated_ads_pos = (
                    torch.einsum(
                        "bi,bij->bj",
                        fractional,
                        self.batch.cell.transpose(1, 2),
                    )
                    - ads_com_pos
                )

                if torch.allclose(
                    updated_ads_pos,
                    torch.zeros_like(updated_ads_pos),
                    rtol=1e-3,
                    atol=1e-3,
                ):
                    cvg_count += 1
                    if cvg_count == 10:
                        break
                # update positions
                updated_pos = torch.zeros_like(self.batch.pos)
                updated_pos[self.batch.tags == 2] += updated_ads_pos[
                    self.batch.batch
                ][self.batch.tags == 2]
                self.set_positions(
                    updated_pos.double(),
                    torch.ones_like(self.batch.pos[:, 0]).bool(),
                )

                self.write(
                    torch.zeros(
                        self.batch.batch.max() + 1,
                        device=self.batch.pos.device,
                    ),
                    torch.zeros(
                        self.batch.pos.shape, device=self.batch.pos.device
                    ),
                    torch.ones_like(self.batch.pos[:, 0]).bool(),
                )

    def reverse_sde_sampling_rot(self):
        if "ads_std_low" in self.denoising_pos_params:
            ads_std_low = self.denoising_pos_params["ads_std_low"]
            ads_std_high = self.denoising_pos_params["ads_std_high"]
            rot_std_low = self.denoising_pos_params["rot_std_low"]
            rot_std_high = self.denoising_pos_params["rot_std_high"]

            ode = self.denoising_pos_params.get("ode", True)

            num_steps = self.denoising_pos_params["num_steps"]

            tr_schedule = torch.tensor(
                np.linspace(1, 0, num_steps + 1)[:-1],
                dtype=torch.float32,
                device=self.batch.pos.device,
            )

            noise = torch.rand(self.batch.batch.max() + 1, 3)
            ads_com_noise = torch.einsum(
                "bi,bij->bj", noise, self.batch.cell.transpose(1, 2)
            )

            ads_init_pos = self.batch.pos[self.batch.tags == 2]
            ads_init_com = self._get_ads_output(self.batch.pos)

            ads_com_noise[:, -1] = ads_init_com[:, -1]
            ads_init_pos_rel = (
                ads_init_pos
                - ads_init_com[self.batch.batch][self.batch.tags == 2]
            )
            ads_pos = (
                ads_init_pos_rel
                + ads_com_noise[self.batch.batch][self.batch.tags == 2]
            )
            self.batch.pos[self.batch.tags == 2] = ads_pos
            
            cvg_count = 0
            logging.info("Starting reverse sde samling:")
            for t_idx in tqdm(range(num_steps)):
                tr_sigma = (
                    ads_std_low ** (1 - tr_schedule[t_idx])
                    * ads_std_high ** tr_schedule[t_idx]
                )
                rot_sigma = (
                    rot_std_low ** (1 - tr_schedule[t_idx])
                    * rot_std_high ** tr_schedule[t_idx]
                )

                tr_g = (
                    tr_sigma * (2 * np.log(ads_std_high / ads_std_low)) ** 0.5
                )
                rot_g = (
                    2
                    * rot_sigma
                    * torch.sqrt(
                        torch.tensor(np.log(rot_std_high / rot_std_low))
                    )
                )

                dt = (
                    tr_schedule[t_idx] - tr_schedule[t_idx + 1]
                    if t_idx < num_steps - 1
                    else tr_schedule[t_idx]
                )

                noise_pred, rot_pred = self.model.get_denoising_prediction(
                    self.batch
                )
                noise_pred = self._get_ads_output(noise_pred)
                rot_pred = self._get_ads_output(rot_pred)

                if ode:
                    # ODE case
                    updated_ads_pos = 0.5 * tr_g**2 * dt * noise_pred
                    updated_rot_vec = 0.5 * rot_pred * dt * rot_g**2
                else:
                    tr_z = torch.normal(
                        mean=0,
                        std=1,
                        size=noise_pred.shape,
                        device=self.device,
                    )
                    updated_ads_pos = (
                        tr_g**2 * dt * noise_pred + tr_g * np.sqrt(dt) * tr_z
                    )

                    rot_z = torch.normal(
                        mean=0,
                        std=1,
                        size=noise_pred.shape,
                        device=self.device,
                    )
                    updated_rot_vec = (
                        rot_pred * dt * rot_g**2
                        + rot_g * np.sqrt(dt) * rot_z
                    )

                # PBC handling
                ads_com_pos = self._get_ads_output(self.batch.pos)
                updated_ads_pos[:, -1] = 0
                fractional = torch.linalg.solve(
                    self.batch.cell, ads_com_pos + updated_ads_pos
                )
                fractional %= 1
                fractional %= 1
                updated_ads_pos = (
                    torch.einsum(
                        "bi,bij->bj",
                        fractional,
                        self.batch.cell.transpose(1, 2),
                    )
                    - ads_com_pos
                )

                if torch.allclose(
                    updated_ads_pos,
                    torch.zeros_like(updated_ads_pos),
                    rtol=1e-3,
                    atol=1e-3,
                ):
                    cvg_count += 1
                    if cvg_count == 10:
                        break

                new_ads_pos = []
                ads_pos = self.batch.pos[self.batch.tags == 2]
                for batch_idx in range(self.batch.batch.max() + 1):
                    batch_mask = (
                        self.batch.batch[self.batch.tags == 2] == batch_idx
                    )
                    rot_mat = axis_angle_to_matrix(
                        updated_rot_vec[batch_idx]
                    ).float()
                    new_ads_pos.append(
                        (ads_pos[batch_mask] - ads_com_pos[batch_idx])
                        @ rot_mat.T
                        + updated_ads_pos[batch_idx]
                        + ads_com_pos[batch_idx]
                    )

                new_ads_pos = torch.cat(new_ads_pos)

                # update positions
                updated_pos = torch.zeros_like(self.batch.pos)
                updated_pos[self.batch.tags == 2] += updated_ads_pos[
                    self.batch.batch
                ][self.batch.tags == 2]

                if not self.early_stop_batch:
                    update_mask = torch.ones_like(self.batch.pos[:, 0]).bool()
                    update = torch.where(
                        update_mask.unsqueeze(1), updated_pos.double(), 0.0
                    )

                # self.batch.pos += updated_pos.to(dtype=torch.float32)
                self.batch.pos[self.batch.tags == 2] = new_ads_pos

                if not self.otf_graph:
                    self.model.update_graph(self.batch)

                self.write(
                    torch.zeros(
                        self.batch.batch.max() + 1,
                        device=self.batch.pos.device,
                    ),
                    torch.zeros(
                        self.batch.pos.shape, device=self.batch.pos.device
                    ),
                    torch.ones_like(self.batch.pos[:, 0]).bool(),
                )

    def langevin_dynamics(self):
        if "ads_std_low" in self.denoising_pos_params:
            std_low = self.denoising_pos_params["ads_std_low"]
            std_high = self.denoising_pos_params["ads_std_high"]
            num_steps = self.denoising_pos_params["num_steps"]
            n_step_each = self.denoising_pos_params["n_step_each"]

            sigmas = torch.tensor(
                np.exp(
                    np.linspace(np.log(std_high), np.log(std_low), num_steps)
                ),
                dtype=torch.float32,
                device=self.batch.pos.device,
            )

            noise = torch.rand(self.batch.batch.max() + 1, 3)
            ads_com_noise = torch.einsum(
                "bi,bij->bj", noise, self.batch.cell.transpose(1, 2)
            )

            ads_init_pos = self.batch.pos[self.batch.tags == 2]
            ads_init_com = self._get_ads_output(self.batch.pos)
            ads_com_noise[:, -1] = ads_init_com[:, -1]
            ads_init_pos_rel = (
                ads_init_pos
                - ads_init_com[self.batch.batch][self.batch.tags == 2]
            )
            ads_pos = (
                ads_init_pos_rel
                + ads_com_noise[self.batch.batch][self.batch.tags == 2]
            )
            self.batch.pos[self.batch.tags == 2] = ads_pos

            logging.info("Starting langevin dynamics:")
            for sigma in tqdm(sigmas, total=sigmas.shape[0]):

                step_size = (
                    self.denoising_pos_params["step_lr"]
                    * (sigma / sigmas[-1]) ** 2
                )
                for step in range(n_step_each):

                    self.batch.denoising_pos_forward = True
                    noise_pred = self.model.get_denoising_prediction(
                        self.batch
                    )
                    noise_pred = self._get_ads_output(noise_pred)

                    noise = torch.randn_like(noise_pred) * torch.sqrt(
                        step_size * 2
                    )
                    updated_ads_pos = step_size * noise_pred + noise

                    ads_com_pos = self._get_ads_output(self.batch.pos)
                    updated_ads_pos[:, -1] = 0
                    fractional = torch.linalg.solve(
                        self.batch.cell, ads_com_pos + updated_ads_pos
                    )

                    fractional %= 1
                    fractional %= 1
                    updated_ads_pos = (
                        torch.einsum(
                            "bi,bij->bj",
                            fractional,
                            self.batch.cell.transpose(1, 2),
                        )
                        - ads_com_pos
                    )

                    updated_pos = torch.zeros_like(self.batch.pos)
                    updated_pos[self.batch.tags == 2] += updated_ads_pos[
                        self.batch.batch
                    ][self.batch.tags == 2]

                    self.set_positions(
                        updated_pos.double(),
                        torch.ones_like(self.batch.pos[:, 0]).bool(),
                    )

                    self.write(
                        torch.zeros(
                            self.batch.batch.max() + 1,
                            device=self.batch.pos.device,
                        ),
                        torch.zeros(
                            self.batch.pos.shape, device=self.batch.pos.device
                        ),
                        torch.ones_like(self.batch.pos[:, 0]).bool(),
                    )

    def _get_ads_output(self, pred):
        ads_pred = scatter(
            pred[self.batch.tags == 2],
            self.batch.batch[self.batch.tags == 2],
            dim=0,
            reduce="mean",
        )
        return ads_pred

    def write(self, energy, forces, update_mask) -> None:
        self.batch.y, self.batch.force = energy, forces
        atoms_objects = batch_to_atoms(self.batch)
        update_mask_ = torch.split(update_mask, self.batch.natoms.tolist())
        for atm, traj, mask in zip(
            atoms_objects, self.trajectories, update_mask_
        ):
            if mask[0] or not self.save_full:
                traj.write(atm)

    def compute_metrics(self, positions):
        return

    def log(self):
        return


class DiffTorchCalc:
    def __init__(self, model, transform=None) -> None:
        self.model = model
        self.transform = transform

    def get_denoising_prediction(self, atoms, apply_constraint: bool = True):
        predictions = self.model.predict_denoising(
            atoms, per_image=False, disable_tqdm=True
        )
        positions = predictions["positions"]
        if "positions_free" in predictions and apply_constraint:
            positions_free = predictions["positions_free"]
            positions_free[atoms.fixed == 1] = 0
            return positions, positions_free
        return positions

    def update_graph(self, atoms):
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(
            atoms, 6, 50
        )
        atoms.edge_index = edge_index
        atoms.cell_offsets = cell_offsets
        atoms.neighbors = num_neighbors
        if self.transform is not None:
            atoms = self.transform(atoms)
        return atoms
