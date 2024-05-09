"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.



Utilities to interface OCP models/trainers with the Atomic Simulation
Environment (ASE)
"""

import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.constraints import FixAtoms


def batch_to_atoms(batch):
    n_systems = batch.natoms.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell
    energies = batch.y.tolist()

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].long().tolist()),
            pbc=[True, True, True],
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects
