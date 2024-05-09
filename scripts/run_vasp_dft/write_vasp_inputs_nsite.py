import numpy as np
import ase.io
from tqdm import tqdm
import lmdb, time, copy, shutil, glob, random, sys, datetime, pickle, lmdb, os
from ocdata.utils.vasp import write_vasp_input_files
from adsorbdiff.placement import DetectTrajAnomaly
import os

# Link to the directory with all simulations for an adslab system
TRAJ_INPUT_PATH = ""

# Add link to the tags.pkl file
tag_path = ""

VASP_FLAGS = {
    "ibrion": 2,
    "nsw": 0,
    "isif": 0,
    "isym": 0,
    "lreal": "Auto",
    "ediffg": -0.03,
    "symprec": 1e-10,
    "encut": 350.0,
    "laechg": True,
    "lwave": False,
    "ncore": 4,
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
}

with open(os.path.join(tag_path), "rb") as h:
    tags_map = pickle.load(h)

traj_paths = glob.glob(
    f"{TRAJ_INPUT_PATH}/*/*.traj"
)


def anomalous_structure(traj, sid):
    initial_atoms = traj[0]
    final_atoms = traj[-1]
    atom_tags = tags_map[sid]
    detector = DetectTrajAnomaly(initial_atoms, final_atoms, atom_tags)
    anom = np.array(
        [
            detector.is_adsorbate_dissociated(),
            detector.is_adsorbate_desorbed(),
            detector.has_surface_changed(),
            detector.is_adsorbate_intercalated(),
        ]
    )
    return anom

uniques_sids = {}
for traj_path in tqdm(traj_paths):
    #traj = ase.io.read(traj_path, ":")
    
    if traj_path.split("/")[-1].count("_") == 3:
        sid, fid = traj_path.split("/")[-1].split(".")[0].rsplit("_", 1)
    elif traj_path.split("/")[-1].count("_") == 2:

        sid = traj_path.split("/")[-1].split(".")[0]
        fid = 0
    
    if sid in uniques_sids:
        continue
    else:
        uniques_sids[sid] = 1

    files_per_sid = glob.glob(f"{TRAJ_INPUT_PATH}/*/relaxations/{sid}*.traj")

    # get the minimum energy structure
    energies = np.array(
        list(map(lambda x: ase.io.read(x).get_potential_energy(), files_per_sid))
    )
    sorted_energy_idx = np.argsort(energies)
    
    count = 0
    while count < len(sorted_energy_idx):
        traj = ase.io.read(files_per_sid[sorted_energy_idx[0]], ":")
        if anomalous_structure(traj, sid).any():
            sorted_energy_idx = sorted_energy_idx[1:]
        else:
            break
     
    if count == len(sorted_energy_idx):
        print("All structures are anomalous for ", sid)
        continue

    relaxed_struct = traj[-1]

    # set constraints based on tags
    tags = tags_map[sid]
    fixed_atoms = np.where(tags == 2)[0]
    relaxed_struct.set_constraint(ase.constraints.FixAtoms(fixed_atoms))
    
    os.makedirs(f"{TRAJ_INPUT_PATH}/vasp", exist_ok=True)
    write_vasp_input_files(
        relaxed_struct,
        outdir=f"{TRAJ_INPUT_PATH}/vasp/{sid}_{fid}",
        vasp_flags=VASP_FLAGS,
    )
