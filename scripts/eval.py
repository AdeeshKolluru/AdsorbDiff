import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import glob
import ase.io
import os

from adsorbdiff.placement import DetectTrajAnomaly
import multiprocessing as mp
import matplotlib.pyplot as plt


def get_success_from_trajs_rewrite(traj_path, dft_targets):
    traj_files = glob.glob(traj_path + "/*.traj")
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    with tqdm(total=len(dft_targets)) as pbar:
        for traj_file in traj_files:
            if traj_file.split("/")[-1].count("_") == 3:
                sid, fid = (
                    traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
                )
            elif traj_file.split("/")[-1].count("_") == 2:
                sid = traj_file.split("/")[-1].split(".")[0]
                fid = 0
            if sid in minE_dict:

                continue
            pbar.update(1)
            traj_paths_by_sid = glob.glob(f"{traj_path}/{sid}*.traj")
            minE = float("inf")
            for traj_per_sid in traj_paths_by_sid:
                traj = ase.io.read(traj_per_sid, ":")
                mlE = traj[-1].get_potential_energy()

                anom = anomalous_structure(traj)
                total_anoms += anom
                if anom.any():
                    anom_dict[sid] = {fid: anom}
                    continue
                if mlE < minE:
                    minE = mlE
            minE_dict[sid] = minE
            success_rate += is_successful(minE, dft_targets[sid])

    success_rate /= len(minE_dict)
    print(len(minE_dict))
    if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
        with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
            pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_train_trajs(traj_path, train_minE_pos):
    traj_files = glob.glob(traj_path + "/*.traj")
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    with tqdm(total=len(train_minE_pos)) as pbar:
        for traj_file in traj_files:
            if traj_file.split("/")[-1].count("_") == 3:
                sid, fid = (
                    traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
                )
            elif traj_file.split("/")[-1].count("_") == 2:
                sid = traj_file.split("/")[-1].split(".")[0]
                fid = 0
            if sid in minE_dict:

                continue
            pbar.update(1)
            traj_paths_by_sid = glob.glob(f"{traj_path}/{sid}*.traj")
            minE = float("inf")
            for traj_per_sid in traj_paths_by_sid:
                traj = ase.io.read(traj_per_sid, ":")
                mlE = traj[-1].get_potential_energy()

                anom = anomalous_structure(traj)
                total_anoms += anom
                if anom.any():
                    anom_dict[sid] = {fid: anom}
                    continue
                assert len(mlE) == 1
                if mlE[0] < minE:
                    minE = mlE[0]
            minE_dict[sid] = minE
            success_rate += is_successful(minE, train_minE_pos[sid]["energy"])

    success_rate /= 44
    print(len(minE_dict))
    if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
        with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
            pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_dft_valood(traj_path, dft_targets):
    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_ref_energies.pkl",
        "rb",
    ) as f:
        ref_energies = pickle.load(f)
    traj_files = glob.glob(traj_path + "/vasp/*")
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    count = 0
    with tqdm(total=44) as pbar:
        for traj_file in traj_files:
            try:
                if traj_file.split("/")[-1].count("_") == 3:
                    sid, fid = (
                        traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
                    )
                elif traj_file.split("/")[-1].count("_") == 2:
                    sid = traj_file.split("/")[-1].split(".")[0]
                    fid = 0

                if sid in minE_dict:

                    continue
                pbar.update(1)
                outcars_by_sid = glob.glob(f"{traj_path}/vasp/{sid}*/OUTCAR")
                minE = float("inf")

                for outcar_per_sid in outcars_by_sid:
                    file = ase.io.read(outcar_per_sid)
                    mlE = file.get_potential_energy()
                    mlE -= ref_energies[sid]
                    # anom = anomalous_structure(traj)
                    # total_anoms += anom
                    # if anom.any():
                    #    anom_dict[sid] = {fid: anom}
                    #    continue

                    if mlE < minE:
                        minE = mlE
                minE_dict[sid] = minE
                count += 1
                success_rate += is_successful(minE, dft_targets[sid])
            except Exception as e:
                print(e)
                print("Error with ", traj_file)
                # removes the outcar file for rerunning DFT on them
                import shutil

                shutil.rmtree(traj_file)

    success_rate /= 50
    print(count)
    # if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
    #     with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
    #         pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    # print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_dft_train(traj_path, train_minE_pos):
    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_ref_energies.pkl",
        "rb",
    ) as f:
        ref_energies = pickle.load(f)
    traj_files = glob.glob(traj_path + "/vasp/*")
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    count = 0
    with tqdm(total=44) as pbar:
        for traj_file in traj_files:
            try:
                if traj_file.split("/")[-1].count("_") == 3:
                    sid, fid = (
                        traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
                    )
                elif traj_file.split("/")[-1].count("_") == 2:
                    sid = traj_file.split("/")[-1].split(".")[0]
                    fid = 0

                if sid in minE_dict:

                    continue
                pbar.update(1)
                outcars_by_sid = glob.glob(f"{traj_path}/vasp/{sid}*/OUTCAR")
                minE = float("inf")

                for outcar_per_sid in outcars_by_sid:
                    file = ase.io.read(outcar_per_sid)
                    mlE = file.get_potential_energy()
                    mlE -= ref_energies[sid]
                    # anom = anomalous_structure(traj)
                    # total_anoms += anom
                    # if anom.any():
                    #    anom_dict[sid] = {fid: anom}
                    #    continue

                    if mlE < minE:
                        minE = mlE
                minE_dict[sid] = minE
                count += 1
                success_rate += is_successful(
                    minE, train_minE_pos[sid]["energy"]
                )
            except Exception as e:
                print(e)
                print("Error with ", traj_file)
                # removes the outcar file for rerunning DFT on them
                import shutil

                shutil.rmtree(traj_file)

    success_rate /= 44
    print(count)
    # if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
    #     with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
    #         pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    # print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_dft(traj_path, dft_targets):
    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_ref_energies.pkl",
        "rb",
    ) as f:
        ref_energies = pickle.load(f)
    traj_files = glob.glob(traj_path + "/vasp/*")
    import pdb

    pdb.set_trace()
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    with tqdm(total=len(dft_targets)) as pbar:
        for traj_file in traj_files:
            try:
                if traj_file.split("/")[-1].count("_") == 3:
                    sid, fid = (
                        traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
                    )
                elif traj_file.split("/")[-1].count("_") == 2:
                    sid = traj_file.split("/")[-1].split(".")[0]
                    fid = 0

                if sid in minE_dict:

                    continue
                pbar.update(1)
                outcars_by_sid = glob.glob(f"{traj_path}/vasp/{sid}*/OUTCAR")
                minE = float("inf")

                for outcar_per_sid in outcars_by_sid:
                    file = ase.io.read(outcar_per_sid)
                    mlE = file.get_potential_energy()
                    mlE -= ref_energies[sid]
                    # anom = anomalous_structure(traj)
                    # total_anoms += anom
                    # if anom.any():
                    #    anom_dict[sid] = {fid: anom}
                    #    continue

                    if mlE < minE:
                        minE = mlE
                minE_dict[sid] = minE
                success_rate += is_successful(minE, dft_targets[sid])
            except Exception as e:
                print(e)
                print("Error with ", traj_file)
    success_rate /= len(minE_dict)
    print(len(minE_dict))
    # if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
    #     with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
    #         pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    # print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_trajs_parallel(traj_path, dft_targets, num_workers=8):
    traj_files = glob.glob(traj_path + "/*.traj")
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)

    def success_per_sid(sid):
        traj_paths_by_sid = glob.glob(f"{traj_path}/{sid}_*.traj")
        minE = float("inf")
        for traj_per_sid in traj_paths_by_sid:
            traj = ase.io.read(traj_per_sid, ":")
            mlE = traj[-1].get_potential_energy()
            anom = anomalous_structure(traj)
            total_anoms += anom
            if anom.any():
                anom_dict[sid] = {fid: anom}
                continue
            if mlE < minE:
                minE = mlE
        minE_dict[sid] = minE
        return is_successful(minE, dft_targets[sid])

    success_rate = mp.Pool(num_workers).map(
        success_per_sid, dft_targets.keys()
    )
    success_rate = sum(success_rate)

    success_rate /= len(minE_dict)
    print(len(minE_dict))
    if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
        with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
            pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_noisy_relax_trajs(traj_path, dft_targets):
    # traj_dirs = glob.glob(traj_path)
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    with tqdm(total=len(dft_targets)) as pbar:
        traj_dir = os.path.join(traj_path, "0")
        traj_paths = glob.glob(traj_dir + "/*.traj")
        for traj_file in traj_paths:
            sid, fid = traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
            files_per_sid = glob.glob(f"{traj_path}/*/{sid}*.traj")
            minE = float("inf")
            for traj_per_sid in files_per_sid:
                traj = ase.io.read(traj_per_sid, ":")
                mlE = traj[-1].get_potential_energy()
                anom = anomalous_structure(traj)
                total_anoms += anom
                if anom.any():
                    anom_dict[sid] = {fid: anom}
                    continue
                if mlE < minE:
                    minE = mlE
            minE_dict[sid] = minE
            success_rate += is_successful(minE, dft_targets[sid])
            pbar.update(1)
    success_rate /= len(minE_dict)
    print(len(minE_dict))
    if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
        with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
            pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_train_trajs_nsite(traj_path, train_minE_pos):
    # traj_dirs = glob.glob(traj_path)
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    with tqdm(total=len(train_minE_pos)) as pbar:
        traj_dir = os.path.join(traj_path, "0")
        traj_paths = glob.glob(traj_dir + "/*.traj")
        for traj_file in traj_paths:
            if traj_file.split("/")[-1].count("_") == 3:
                sid, fid = (
                    traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
                )
            elif traj_file.split("/")[-1].count("_") == 2:
                sid = traj_file.split("/")[-1].split(".")[0]
                fid = 0
            files_per_sid = glob.glob(f"{traj_path}/*/relaxations/{sid}*.traj")
            minE = float("inf")
            for traj_per_sid in files_per_sid:
                traj = ase.io.read(traj_per_sid, ":")
                mlE = traj[-1].get_potential_energy()
                anom = anomalous_structure(traj)
                total_anoms += anom
                if anom.any():
                    anom_dict[sid] = {fid: anom}
                    continue
                if mlE < minE:
                    minE = mlE
            minE_dict[sid] = minE
            success_rate += is_successful(minE, train_minE_pos[sid]["energy"])
            pbar.update(1)
    success_rate /= len(minE_dict)
    print(len(minE_dict))
    if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
        with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
            pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_noisy_relax_train_trajs(traj_path, train_minE_pos):
    # traj_dirs = glob.glob(traj_path)
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    with tqdm(total=len(train_minE_pos)) as pbar:
        traj_dir = os.path.join(traj_path, "0")
        traj_paths = glob.glob(traj_dir + "/*.traj")
        for traj_file in traj_paths:
            if traj_file.split("/")[-1].count("_") == 3:
                sid, fid = (
                    traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
                )
            elif traj_file.split("/")[-1].count("_") == 2:
                sid = traj_file.split("/")[-1].split(".")[0]
                fid = 0
            files_per_sid = glob.glob(f"{traj_path}/*/{sid}*.traj")
            minE = float("inf")
            for traj_per_sid in files_per_sid:
                traj = ase.io.read(traj_per_sid, ":")
                mlE = traj[-1].get_potential_energy()
                anom = anomalous_structure(traj)
                total_anoms += anom
                if anom.any():
                    anom_dict[sid] = {fid: anom}
                    continue
                if mlE < minE:
                    minE = mlE
            minE_dict[sid] = minE
            success_rate += is_successful(minE, train_minE_pos[sid]["energy"])
            pbar.update(1)
    success_rate /= len(minE_dict)
    print(len(minE_dict))
    if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
        with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
            pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_npz_energies(npz_path, traj_path, dft_targets):
    npz_results = np.load(npz_path)
    npz_results = {
        k: v for k, v in zip(npz_results["ids"], npz_results["energy"])
    }

    traj_files = glob.glob(traj_path + "/*.traj")
    anom_dict = {}
    total_anoms = np.zeros(4)
    success_rate = 0.0
    minE_dict = defaultdict(dict)
    with tqdm(total=len(dft_targets)) as pbar:
        for traj_file in traj_files:
            sid, fid = traj_file.split("/")[-1].split(".")[0].rsplit("_", 1)
            if sid in minE_dict:

                continue
            pbar.update(1)
            traj_paths_by_sid = glob.glob(f"{traj_path}/{sid}_*.traj")
            minE = float("inf")
            for traj_per_sid in traj_paths_by_sid:
                traj = ase.io.read(traj_per_sid, ":")
                # mlE = traj[-1].get_potential_energy()
                mlE = npz_results[f"{sid}_{fid}"]
                anom = anomalous_structure(traj)
                total_anoms += anom
                if anom.any():
                    anom_dict[sid] = {fid: anom}
                    continue
                if mlE < minE:
                    minE = mlE
            minE_dict[sid] = minE
            success_rate += is_successful(minE, dft_targets[sid])

    success_rate /= len(minE_dict)
    print(len(minE_dict))
    if not Path(f"{traj_path}/anomalous_structures_new.pkl").exists():
        with open(f"{traj_path}/anomalous_structures_new.pkl", "wb") as f:
            pickle.dump(anom_dict, f)
    if not Path(f"{traj_path}/success_new.txt").exists():
        with open(f"{traj_path}/success_new.txt", "w") as f:
            f.write(str(success_rate * 100))
    print("Anomalies", total_anoms)
    print("Success rate: ", success_rate * 100)


def get_success_from_pkl(pkl_path, dft_targets):
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
    success_rate = 0.0
    success_rate_with_dft = 0.0

    c = 0
    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/non_metal_sids.pkl",
        "rb",
    ) as f:
        non_metal_sids = pickle.load(f)

    for sid in tqdm(dft_targets):
        # if sid in non_metal_sids:
        #     continue
        c += 1
        minE = float("inf")
        for config in pkl[sid]:
            mlE = pkl[sid][config]["ml_energy"]
            if mlE < minE:
                minE = mlE
                ml_dft_E = pkl[sid][config]["ml+dft_energy"]

        success = is_successful(minE, dft_targets[sid])
        success_rate += success
        dft_E = dft_targets[sid]
        if abs(dft_E - ml_dft_E) > 0.1:
            success_rate_with_dft += success
        else:
            if success:
                print(
                    f"Success but not with dft: sid {sid}, ML {minE}, DFT {dft_E}, ML+DFT {ml_dft_E}"
                )
    print(c)
    print("Success rate: ", success_rate / c * 100)
    print("Success rate with dft: ", success_rate_with_dft / c * 100)


def calculate_success_metrics(minE_dict, dft_targets, traj_paths):
    success_rate = 0.0
    print("Calculating success rate ...")
    for traj in tqdm(traj_paths):
        sid, fid = traj.split("/")[-1].split(".")[0].rsplit("_", 1)[0]
        success = is_successful(minE_dict[sid][0], dft_targets[sid])
        success_rate += success
    success_rate /= len(traj_paths)
    print("Success rate: ", success_rate * 100)


def anomalous_structure(traj):
    initial_atoms = traj[0]
    final_atoms = traj[-1]
    atom_tags = initial_atoms.get_tags()
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


def is_successful(best_pred_energy, best_dft_energy, SUCCESS_THRESHOLD=0.1):
    diff = best_pred_energy - best_dft_energy
    success_parity = diff <= SUCCESS_THRESHOLD

    return success_parity


# def get_dft_data(targets):
#     dft_data = defaultdict(dict)
#     for system in targets:
#         minE = float("inf")
#         for adslab in targets[system]:
#             if adslab[1] < minE:
#                 dft_data[system] = adslab[1]
#                 minE = adslab[1]

#     with open(
#         "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/dft_val_minE_old.pkl", "wb"
#     ) as f:
#         pickle.dump(dft_data, f)
#     return dft_data
def get_dft_data(targets):
    dft_data = defaultdict(dict)
    final_dft_data = {}
    for system in targets:
        for adslab in targets[system]:
            dft_data[system][adslab[0]] = adslab[1]
        final_dft_data[system] = min(list(dft_data[system].values()))

    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/dft_val_minE_new.pkl",
        "wb",
    ) as f:
        pickle.dump(final_dft_data, f)
    return dft_data


def get_minE_pos(targets):
    dft_data = defaultdict(dict)
    for system in targets:
        minE = float("inf")
        for adslab in targets[system]:
            if adslab[1] < minE:
                dft_data[system] = adslab[0]
                minE = adslab[1]
        traj = ase.io.read(
            f"/home/jovyan/shared-scratch/adeesh/data/oc20_dense/dft_val_trajectories/{system}/{system}_{dft_data[system]}.traj"
        )
        dft_data[system] = traj.get_positions()
    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/dft_val_minE_pos.pkl",
        "wb",
    ) as f:
        pickle.dump(dft_data, f)
    return dft_data


def get_mean_distances_from_traj(path):
    sid, fid = path.split("/")[-1].split(".")[0].rsplit("_", 1)
    pos_target = dft_pos_target[sid]
    traj_predicted = ase.io.read(path, ":")

    try:
        ml_relaxed = traj_predicted[global_step]
        pos_prediction = ml_relaxed.get_positions()
        cell = ml_relaxed.get_cell(complete=True)
        diff = pos_target - pos_prediction
        return np.mean(
            np.linalg.norm(min_diff(diff, cell)[tags_map[sid] > 0], axis=1)
        )
    except:
        return np.nan


def store_train_minE_pos():
    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_ref_energies.pkl",
        "rb",
    ) as f:
        ref_energies = pickle.load(f)
    train_minE_pos = defaultdict(dict)
    sid_dirs = glob.glob(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/trajs/*"
    )

    for traj_dir in sid_dirs:
        sid = traj_dir.split("/")[-1]
        traj_files = glob.glob(traj_dir + "/*[!surface].traj")
        traj_dir = (
            f"/home/jovyan/shared-scratch/adeesh/data/oc20_dense/trajs/{sid}"
        )
        traj_files = glob.glob(traj_dir + "/*[!surface].traj")
        energies = np.array(
            list(
                map(
                    lambda x: ase.io.read(x).get_potential_energy(), traj_files
                )
            )
        )
        minE_idx = np.argmin(energies)
        traj_path = traj_files[minE_idx]
        adslab = ase.io.read(traj_path)
        pos_target = adslab.get_positions()
        energy_target = adslab.get_potential_energy()
        ref_energy = ref_energies[sid]
        energy_target -= ref_energy

        train_minE_pos[sid]["pos"] = pos_target
        train_minE_pos[sid]["energy"] = energy_target
    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/train_minE_pos.pkl",
        "wb",
    ) as f:
        pickle.dump(train_minE_pos, f)


def get_mean_distances_from_traj_single(path):
    try:
        sid = path.split("/")[-1].split(".")[0]
        if sid.count("_") == 3:
            sid, fid = sid.rsplit("_", 1)
        traj_dir = (
            f"/home/jovyan/shared-scratch/adeesh/data/oc20_dense/trajs/{sid}"
        )
        traj_files = glob.glob(traj_dir + "/*[!surface].traj")
        energies = np.array(
            list(
                map(
                    lambda x: ase.io.read(x).get_potential_energy(), traj_files
                )
            )
        )
        minE_idx = np.argmin(energies)
        traj_path = traj_files[minE_idx]
        adslab = ase.io.read(traj_path)
        pos_target = adslab.get_positions()
        traj_predicted = ase.io.read(path, ":")

        ml_relaxed = traj_predicted[global_step]
        pos_prediction = ml_relaxed.get_positions()
        cell = ml_relaxed.get_cell(complete=True)

        diff = pos_target - pos_prediction
        ads_com_diff = pos_target[tags_map[sid] == 2].mean(
            axis=0
        ) - pos_prediction[tags_map[sid] == 2].mean(axis=0)

        ads_com_diff[-1] = 0
        return np.linalg.norm(min_diff(ads_com_diff, cell))
    except:
        return 0


def parallelized_adwt(traj_dir, step, nprocs=8):
    global global_step
    global_step = step
    systems = glob.glob(
        traj_dir + "/*.traj",
    )
    pool = mp.Pool(processes=nprocs)
    mean_distance = pool.map(get_mean_distances_from_traj, systems)
    # mean_distance = []
    # for i in range(100):
    #    mean_distance.append(get_mean_distances_from_traj(systems[i]))
    dwt, adwt = compute_metrics(mean_distance)
    print(f"Step: {step}, DwT: {dwt}, ADwT: {adwt}")
    return dwt, adwt


def compute_metrics(distances):
    distances = np.array(distances)

    dwts = []
    intv = np.arange(0.01, 0.5, 0.001)
    for i in intv:
        count = sum(distances < i)
        dwts.append(100 * count / len(distances))
    adwt = np.mean(dwts)
    dwt = 100 * sum(distances < 0.1) / len(distances)

    return dwt, adwt


def min_diff(diff, cell):
    positions = diff
    fractional = np.linalg.solve(cell.T, positions.T).T

    # for i, periodic in enumerate(atoms_init.pbc):
    #     if periodic:
    # Yes, we need to do it twice.
    # See the scaled_positions.py test.
    fractional %= 1.0
    fractional %= 1.0

    fractional[fractional > 0.5] -= 1
    return np.matmul(fractional, cell)


def calculate_pos_maes(traj_dir, nprocs):
    systems = glob.glob(
        traj_dir + "/*.traj",
    )
    steps = np.arange(0, 100, 1)
    per_step_distances = []
    for step in tqdm(steps):
        global global_step
        global_step = int(step)
        pool = mp.Pool(processes=nprocs)
        mean_distances = pool.map(get_mean_distances_from_traj, systems)
        per_step_distances.append(np.nanmean(mean_distances))
        print("Step: ", step, "MAE: ", np.nanmean(mean_distances))
    plt.plot(per_step_distances)
    plt.savefig(os.path.join(traj_dir, "mae.png"))
    plt.close()


def calculate_single_pos_maes(traj_dir):
    systems = glob.glob(
        traj_dir + "/*.traj",
    )
    # steps = np.linspace(0, 250, 5000-1)
    per_step_distances = []
    step = 0
    while True:
        global global_step
        global_step = int(step)
        mean_distance = get_mean_distances_from_traj_single(systems[1])
        if mean_distance == 0:
            break
        per_step_distances.append(mean_distance)
        print("Step: ", step, "MAE: ", mean_distance)
        step += 1
    plt.plot(per_step_distances)
    plt.title(traj_dir.split("/")[-1])
    plt.savefig(os.path.join(traj_dir, "mae.png"))
    plt.close()


def calculate_final_pos_maes(traj_dir):
    systems = glob.glob(
        traj_dir + "/*.traj",
    )
    # steps = np.linspace(0, 250, 5000-1)
    per_step_distances = []
    step = -1
    success = 0
    for system in systems:
        global global_step
        global_step = int(step)
        mean_distance = get_mean_distances_from_traj_single(system)
        if mean_distance == 0:
            break
        if mean_distance < 1:
            success += 1
        per_step_distances.append(mean_distance)
        sid = system.split("/")[-1].split(".")[0]
        print("Sid: ", sid, "MAE: ", mean_distance)
    print("success % - ", success / len(systems) * 100)
    print("overall mean: ", np.mean(per_step_distances))


def calculate_ads_com_maes(traj_dir, nprocs):
    systems = glob.glob(
        traj_dir + "/*.traj",
    )

    def get_mae(path):
        sid, fid = path.split("/")[-1].split(".")[0].rsplit("_", 1)
        pos_target = dft_pos_target[sid]
        traj_predicted = ase.io.read(path, ":")
        try:
            ml_relaxed = traj_predicted[global_step]
            pos_prediction = ml_relaxed.get_positions()
            cell = ml_relaxed.get_cell(complete=True)
            diff = pos_target - pos_prediction
            return np.mean(
                np.linalg.norm(min_diff(diff, cell)[tags_map[sid] > 1], axis=1)
            )
        except:
            return 0

    steps = np.linspace(0, 1000, 50)
    per_step_distances = []
    for step in tqdm(steps):
        global global_step
        global_step = int(step)
        pool = mp.Pool(processes=nprocs)
        mean_distances = pool.map(get_mean_distances_from_traj, systems)
        per_step_distances.append(np.nanmean(mean_distances))
        print("Step: ", step, "MAE: ", np.nanmean(mean_distances))
    plt.plot(per_step_distances)
    plt.savefig(os.path.join(traj_dir, "mae.png"))
    plt.close()


if __name__ == "__main__":
    # traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/com_sde/rel_only_z0_test"
    # traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/eqv2_relaxations/rand1_I0.1_fmax0.01_allmd"
    # traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/overfit_sde/pretrain-bysigma_lmdbcorr2-overfit_std0.1-10_numstep50_sample1"
    # traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/val44_baseline/noisybfgs/val44_std-schedule1_xyads_wrandrots_NS10"
    # traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/vasp/gnoc/rand1_I1"
    traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/valid_rerun/debug_painn_prev2/0/relaxations"
    # traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/valood_baseline/R1I0.1"
    # traj_path = "/home/jovyan/shared-scratch/adeesh/denoising/val44_baseline/w_rot/is2rs_conditional/relaxations"
    print(traj_path.split("/")[-1])

    npz_path = "/home/jovyan/shared-scratch/adeesh/denoising/lmdbs/randheur100_I0.1_fmax0.03/eq2_23M_2M/results/2023-10-13-16-53-20/s2ef_predictions.npz"
    # get DFT targets
    dft_target_path = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/dft_val_minE_new.pkl"
    if not Path(dft_target_path).exists():
        with open(
            "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_val_targets.pkl",
            "rb",
        ) as f:
            dft_targets = pickle.load(f)
        dft_targets = get_dft_data(dft_targets)
    else:
        with open(dft_target_path, "rb") as f:
            dft_targets = pickle.load(f)
    dft_target_pos_path = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/dft_val_minE_pos.pkl"
    tag_path = (
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_tags.pkl"
    )
    with open(os.path.join(tag_path), "rb") as h:
        tags_map = pickle.load(h)
    if not Path(dft_target_pos_path).exists():
        with open(
            "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_val_targets.pkl",
            "rb",
        ) as f:
            dft_targets = pickle.load(f)
        dft_pos_target = get_minE_pos(dft_targets)
    else:
        with open(dft_target_pos_path, "rb") as f:
            dft_pos_target = pickle.load(f)

    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/train_minE_pos.pkl",
        "rb",
    ) as f:
        train_minE_pos = pickle.load(f)

    get_success_from_train_trajs(traj_path, train_minE_pos)
    # get_success_from_train_trajs_nsite(traj_path, train_minE_pos)
    # get_success_from_pkl("/home/jovyan/shared-scratch/adeesh/data/oc20_dense/results.pkl", dft_targets)
    # calculate_pos_maes(traj_path, 1)
    # calculate_single_pos_maes(traj_path)
    # calculate_final_pos_maes(traj_path)
    # get_success_from_trajs(traj_path, dft_targets, 1)
    # parallelized_adwt("/home/jovyan/shared-scratch/adeesh/denoising/relaxations_fmax0.01/densetrain-diff_low0.01_high5_numstep20*50_lr1e-6", -1)
    # get_success_from_trajs_rewrite(traj_path, dft_targets)
    # get_success_from_noisy_relax_trajs(traj_path, dft_targets)
    # get_success_from_noisy_relax_train_trajs(traj_path, train_minE_pos)
    # get_success_from_npz_energies(npz_path, traj_path, dft_targets)
    # get_success_from_trajs_parallel(traj_path, dft_targets)

    # get_success_from_dft_train(traj_path, train_minE_pos)
    # get_success_from_dft_valood(traj_path, dft_targets)
