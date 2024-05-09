"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import lmdb
import numpy as np
import glob
import ase.io
import torch
from typing import Literal

from adsorbdiff.utils.atoms_to_graphs import AtomsToGraphs
from adsorbdiff.placement import Adsorbate, AdsorbateSlabConfig, Bulk, Slab


def adsorbate_placement(
    adslab,
    sid: str,
    interstitial_gap: float = 1,
    num_sites: int = 2,
    mode: Literal[
        "random_site_heuristic_placement", "random", "heuristic"
    ] = "random_site_heuristic_placement",
):

    ads_mask = tags_map[sid] == 2
    adslab.set_tags(tags_map[sid])

    adsorbate_atoms = Adsorbate(
        adsorbate_id_from_db=int(sid.split("_")[0]),
        # adsorbate_db_path="/home/jovyan/repos/Open-Catalyst-Dataset/ocdata/databases/pkls/adsorbates.pkl",
    )
    bulk = Bulk(
        bulk_id_from_db=int(sid.split("_")[1]),
        # bulk_db_path="/home/jovyan/repos/Open-Catalyst-Dataset/ocdata/databases/pkls/bulks.pkl",
    )
    slab_atoms = Slab(bulk=bulk, slab_atoms=adslab[~ads_mask])
    random_adslabs = AdsorbateSlabConfig(
        slab_atoms,
        adsorbate_atoms,
        mode=mode,
        interstitial_gap=interstitial_gap,
        num_sites=num_sites,
    )

    return random_adslabs.atoms_list


def write_images_to_lmdb(mp_arg):
    # def write_images_to_lmdb(a2g, samples):
    a2g, db_path, samples, sampled_ids, idx, pid, args = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    print(db_path)

    for sys_id in samples:
        # try:
        sid = sys_id.strip()
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

        energies -= energies[minE_idx]

        traj_path = traj_files[minE_idx]
        traj = ase.io.read(traj_path)

        relaxed_pos = traj.get_positions()
        # trajs = adsorbate_placement(traj, sid)

        image = a2g.convert(traj)

        image.sid = sid
        image.fid = -1
        image.tags = torch.tensor(tags_map[sid])
        image.y = torch.tensor(energies[minE_idx])

        # ads_mask = tags_map[sid] == 2
        # final_ads_pos = traj.get_positions()[ads_mask]
        # import pdb; pdb.set_trace()
        # random_pos = image.pos[ads_mask]
        # random_pos[:, :2] = torch.tensor(final_ads_pos[:, :2])
        # image.pos[ads_mask] = random_pos

        assert energies[minE_idx] == 0.0
        assert image.pos.shape[0] == image.tags.shape[0]
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(image, protocol=-1))
        txn.commit()
        idx += 1

        # for fid, traj in enumerate(trajs):
        #     image = a2g.convert(traj)
        #     image.sid = f"{sid}_{fid}"
        #     image.fid = fid
        #     image.relaxed_pos = torch.tensor(relaxed_pos)
        #     txn = db.begin(write=True)
        #     txn.put(f"{idx}".encode("ascii"), pickle.dumps(image, protocol=-1))
        #     txn.commit()
        #     idx += 1

        for i, traj_path in enumerate(traj_files):
            if i == minE_idx:
                continue
            traj = ase.io.read(traj_path)
            image = a2g.convert(traj)
            image.sid = sid
            image.fid = i
            image.tags = torch.tensor(tags_map[sid])
            image.y = torch.tensor(energies[i])

            #     final_ads_pos = traj.get_positions()[ads_mask]
            #     random_pos = image.pos[ads_mask]
            #     random_pos[:, :2] = torch.tensor(final_ads_pos[:, :2])
            #     image.pos[ads_mask] = random_pos

            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(image, protocol=-1))
            txn.commit()
            idx += 1
    # except:
    #     print("Error in traj_path: ", traj_path)
    #     pass

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txt-path",
        required=False,
        default="/home/jovyan/shared-scratch/adeesh/data/oc20_dense/unique_train_system_id.txt",
    )
    parser.add_argument(
        "--out-path",
        required=False,
        default="/home/jovyan/shared-scratch/adeesh/data/oc20_dense/lmdbs/val44_RH2I1",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )

    parser.add_argument(
        "--tags", action="store_true", default=True, help="Add atom tags"
    )
    parser.add_argument(
        "--chunk", default=1, type=int, help="Chunk to of inputs to preprocess"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="doesn't parallelize for easy debugging",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.tags:
        tag_path = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_tags.pkl"
        with open(os.path.join(tag_path), "rb") as h:
            tags_map = pickle.load(h)

    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_mapping.pkl",
        "rb",
    ) as f:
        oc20_dense_mapping = pickle.load(f)

    # open and read txt file
    with open(args.txt_path, "r") as f:
        system_ids = f.readlines()

    # shuffle system_ids
    np.random.seed(42)
    np.random.shuffle(system_ids)

    system_ids = system_ids[200:]
    print("### Found %d trajectories" % (len(system_ids)))

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Chunk the trajectories into args.num_workers splits
    chunked_traj_dirs = np.array_split(system_ids, args.num_workers)

    # Extract features
    sampled_ids, idx = [[]] * args.num_workers, [0] * args.num_workers

    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            chunked_traj_dirs[i],
            sampled_ids[i],
            idx[i],
            i,
            args,
        )
        for i in range(args.num_workers)
    ]
    if args.debug:
        op = []
        for i in range(args.num_workers):
            op.append(write_images_to_lmdb(mp_args[i]))
            op = list(zip(*op))
    else:
        op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
