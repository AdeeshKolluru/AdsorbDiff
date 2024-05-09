import copy
import logging
from typing import Dict, Optional

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from adsorbdiff.utils.registry import registry
from adsorbdiff.utils.utils import (
    load_config,
    setup_imports,
    setup_logging,
    update_config,
)
from adsorbdiff.datasets import data_list_collater
from adsorbdiff.utils.atoms_to_graphs import AtomsToGraphs
from adsorbdiff.relaxation.ml_relaxation import ml_diffuse
from adsorbdiff.modules.scaling.util import ensure_fitted

from .ase_utils import batch_to_atoms


class AdsorbDiffCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        config_yml: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        trainer: Optional[str] = None,
        cutoff: int = 12,
        max_neighbors: int = 50,
        cpu: bool = True,
        seed: Optional[int] = 0,
    ):
        """
        AdsorbDiff-ASE Calculator

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint_path (str):
                Path to trained checkpoint.
            trainer (str):
                AdsorbDiff/OCP trainer to be used. "forces" for S2EF, "energy" for IS2RE.
            cutoff (int):
                Cutoff radius to be used for data preprocessing.
            max_neighbors (int):
                Maximum amount of neighbors to store for a given atom.
            cpu (bool):
                Whether to load and run the model on CPU. Set `False` for GPU.
        """
        setup_imports()
        setup_logging()
        Calculator.__init__(self)

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint_path is not None

        checkpoint = None
        if config_yml is not None:
            if isinstance(config_yml, str):
                config, duplicates_warning, duplicates_error = load_config(
                    config_yml
                )
                if len(duplicates_warning) > 0:
                    logging.warning(
                        f"Overwritten config parameters from included configs "
                        f"(non-included parameters take precedence): {duplicates_warning}"
                    )
                if len(duplicates_error) > 0:
                    raise ValueError(
                        f"Conflicting (duplicate) parameters in simultaneously "
                        f"included configs: {duplicates_error}"
                    )
            else:
                config = config_yml

            # Only keeps the train data that might have normalizer values
            if isinstance(config["dataset"], list):
                config["dataset"] = config["dataset"][0]
            elif isinstance(config["dataset"], dict):
                config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu")
            )
            config = checkpoint["config"]

        if trainer is not None:
            config["trainer"] = trainer
        else:
            config["trainer"] = config.get("trainer", "ocp")

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # for checkpoints with relaxation datasets defined, remove to avoid
        # unnecesarily trying to load that dataset
        if "relax_dataset" in config["task"]:
            del config["task"]["relax_dataset"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        config = update_config(config)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint_path
        del config["dataset"]["src"]

        self.trainer = registry.get_trainer_class(config["trainer"])(
            task=config["task"],
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_fns=config["loss_fns"],
            eval_metrics=config["eval_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=config.get("amp", False),
        )

        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path=checkpoint_path, checkpoint=checkpoint
            )

        seed = seed if seed is not None else self.trainer.config["cmd"]["seed"]
        if seed is None:
            logging.warning(
                "No seed has been set in modelcheckpoint or AdsorbDiffCalculator! Results may not be reproducible on re-run"
            )
        else:
            self.trainer.set_seed(seed)

        self.a2g = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )

    def load_checkpoint(self, checkpoint_path: str, checkpoint: Dict = {}):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path, checkpoint)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms: Atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object], otf_graph=True)

        predictions = self.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )

        for key in predictions:
            _pred = predictions[key]
            _pred = _pred.item() if _pred.numel() == 1 else _pred.cpu().numpy()
            self.results[key] = _pred

    def run_diffusion(self, atoms: Atoms, trajectory="trajs/"):
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object], otf_graph=True)
        assert "denoising_pos_params" in self.config["optim"]
        if "sid" not in batch:
            batch.sid = [0]
        ensure_fitted(self.trainer._unwrapped_model)
        self.trainer.model.eval()
        if self.trainer.ema:
            self.trainer.ema.store()
            self.trainer.ema.copy_to()

        relaxed_batch = ml_diffuse(
            batch=batch,
            model=self.trainer,
            denoising_pos_params=self.config["optim"]["denoising_pos_params"],
            traj_dir=trajectory,
            save_full_traj=self.config["task"].get("save_full_traj", True),
            device=torch.device("cuda:0") if not self.trainer.cpu else "cpu",
            transform=None,
            logger=None,
        )
        
        if self.trainer.ema:
            self.trainer.ema.restore()
            
        atoms = batch_to_atoms(relaxed_batch)
        if len(atoms) == 1:
            return atoms[0]
        else:
            return atoms
